# g2p_model.py
# Minimal, self-contained PyTorch scaffold for genotype->phenotype with
# sequence+priors encoder, phylogenetic tree encoder, optional GVP stub,
# generator/discriminator (GAN), and training utilities.

import math
from typing import List, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------
def mlp(sizes: List[int], act=nn.ReLU, dropout=0.0, bn=False) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            if bn:
                layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def to_device(x: Any, device: str):
    if isinstance(x, (list, tuple)):
        return [to_device(xx, device) for xx in x]
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=torch.float32, device=device)
    return x.to(device) if torch.is_tensor(x) else x


# -----------------------------
# Sequence encoding
# -----------------------------
DNA_ALPHABET = "ACGTN"
PROT_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"  # 20 AAs + X(unknown)


class SequenceEncoder:
    def __init__(self, kind: str = "protein"):
        assert kind in {"protein", "dna"}
        self.kind = kind
        self.alphabet = PROT_ALPHABET if kind == "protein" else DNA_ALPHABET
        self.vocab = {ch: i for i, ch in enumerate(self.alphabet)}
        self.dim = len(self.alphabet)

    def one_hot(self, seq: str) -> torch.Tensor:
        L = len(seq)
        x = torch.zeros(L, self.dim, dtype=torch.float32)
        unk = self.vocab[self.alphabet[-1]]
        for i, ch in enumerate(seq.upper()):
            x[i, self.vocab.get(ch, unk)] = 1.0
        return x  # [L, A]

    def embed_tokens(self, seq: str) -> torch.Tensor:
        unk = self.dim - 1
        return torch.tensor([self.vocab.get(ch.upper(), unk) for ch in seq], dtype=torch.long)


def build_prior_tensor(
    length: int,
    treesaap_props: Optional[np.ndarray] = None,  # [L, P1]
    polyphen_scores: Optional[np.ndarray] = None,  # [L, P2]
) -> torch.Tensor:
    parts = []
    if treesaap_props is not None:
        parts.append(torch.tensor(treesaap_props, dtype=torch.float32))
    if polyphen_scores is not None:
        parts.append(torch.tensor(polyphen_scores, dtype=torch.float32))
    if not parts:
        return torch.zeros(length, 0, dtype=torch.float32)
    return torch.cat(parts, dim=-1)


# -----------------------------
# Phylogenetic tree encoder (light GNN)
# -----------------------------
class TreeContextEncoder(nn.Module):
    """
    Nodes represent tips/ancestors; edges have branch lengths.
    Inputs:
      node_init: [N, F]
      edge_index: [2, E] (src,dst)
      edge_len: [E]
    Output:
      node_emb: [N, H]
    """

    def __init__(self, in_dim: int, hidden: int, layers: int = 2):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.bn = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])
        self.att_k = nn.Parameter(torch.randn(hidden))

    def forward(self, node_init: torch.Tensor, edge_index: torch.Tensor, edge_len: torch.Tensor):
        h = self.proj_in(node_init)  # [N, H]
        N = h.size(0)
        src, dst = edge_index
        for lin, bn in zip(self.layers, self.bn):
            key = (h[src] * self.att_k).sum(dim=-1) / math.sqrt(h.size(-1))  # [E]
            w = torch.exp(key) / (1.0 + edge_len)  # decay with branch length
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, w.unsqueeze(-1) * h[src])
            h = h + lin(F.relu(agg))  # residual
            h = bn(h)
        return h  # [N, H]


# -----------------------------
# (Optional) GVP-like stub for structure
# -----------------------------
class GVPStub(nn.Module):
    """
    Minimal GVP-style block.
    Inputs:
      s: [N, S_in] scalar node feats
      v: [N, V_in] vector feats already flattened (e.g., geometric descriptors)
      edge_index: [2, E]
    Output:
      node_emb: [N, S_h + V_h]
    """

    def __init__(self, s_in: int, v_in: int, s_hidden: int = 64, v_hidden: int = 64, iters: int = 2):
        super().__init__()
        self.iters = iters
        self.lin_s = nn.Linear(s_in, s_hidden)
        self.lin_v = nn.Linear(v_in, v_hidden)
        self.msg_s = nn.Linear(s_hidden, s_hidden)
        self.msg_v = nn.Linear(v_hidden, v_hidden)

    def forward(self, s: torch.Tensor, v: torch.Tensor, edge_index: torch.Tensor):
        s = F.relu(self.lin_s(s))
        v = F.relu(self.lin_v(v))
        src, dst = edge_index
        for _ in range(self.iters):
            agg_s = torch.zeros_like(s)
            agg_v = torch.zeros_like(v)
            agg_s.index_add_(0, dst, self.msg_s(s[src]))
            agg_v.index_add_(0, dst, self.msg_v(v[src]))
            s = F.relu(s + agg_s)
            v = F.relu(v + agg_v)
        return torch.cat([s, v], dim=-1)  # [N, s_h+v_h]


# -----------------------------
# Sequence + prior encoder (1D CNN)
# -----------------------------
class SeqVariantEncoder(nn.Module):
    def __init__(self, token_dim: int, prior_dim: int, hidden: int = 256, out_dim: int = 256):
        super().__init__()
        self.conv1 = nn.Conv1d(token_dim + prior_dim, hidden, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, onehot: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        # onehot: [L, A], priors: [L, P]
        x = torch.cat([onehot, priors], dim=-1).transpose(0, 1).unsqueeze(0)  # [1, A+P, L]
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))  # [1, H, L]
        h = h.max(dim=-1).values.squeeze(0)  # [H]
        return self.proj(h)  # [out_dim]


# -----------------------------
# Generator & Discriminator
# -----------------------------
class Generator(nn.Module):
    def __init__(self, es: int, et: int, ep: int, y_dim: int, hidden: int = 256):
        super().__init__()
        self.fuse = mlp([es + et + ep, hidden, hidden], bn=False)
        self.y_head = nn.Linear(hidden, y_dim)
        self.e_head = nn.Linear(hidden, 1)  # global effect score (extend if needed)

    def forward(self, seq_emb: torch.Tensor, tree_emb: torch.Tensor, struct_emb: torch.Tensor):
        h = torch.cat([seq_emb, tree_emb, struct_emb], dim=-1)
        h = self.fuse(h)
        y_hat = self.y_head(h)
        e_hat = self.e_head(h).squeeze(-1)
        return y_hat, e_hat


class Discriminator(nn.Module):
    def __init__(self, es: int, et: int, y_dim: int, hidden: int = 256):
        super().__init__()
        self.net = mlp([es + et + y_dim, hidden, hidden, 1], bn=False)

    def forward(self, seq_emb: torch.Tensor, tree_emb: torch.Tensor, y: torch.Tensor):
        x = torch.cat([seq_emb, tree_emb, y], dim=-1)
        return self.net(x).squeeze(-1)  # scalar


# -----------------------------
# Losses
# -----------------------------
def supervised_loss(y_hat: torch.Tensor, y_true: torch.Tensor, task: str = "regression"):
    if task == "regression":
        mask = ~torch.isnan(y_true)
        return F.mse_loss(y_hat[mask], y_true[mask])
    return F.cross_entropy(y_hat, y_true.long())


def gan_losses(d_real: torch.Tensor, d_fake: torch.Tensor, mode: str = "ls"):
    if mode == "ls":  # Least Squares GAN
        L_D = 0.5 * ((d_real - 1.0) ** 2).mean() + 0.5 * (d_fake**2).mean()
        L_G = 0.5 * ((d_fake - 1.0) ** 2).mean()
        return L_D, L_G
    if mode == "hinge":
        L_D = (F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean())
        L_G = (-d_fake).mean()
        return L_D, L_G
    # vanilla
    L_D = -(torch.log(torch.sigmoid(d_real) + 1e-9).mean() +
            torch.log(1 - torch.sigmoid(d_fake) + 1e-9).mean())
    L_G = -torch.log(torch.sigmoid(d_fake) + 1e-9).mean()
    return L_D, L_G


def phylo_contrastive(
    y_pred: torch.Tensor,
    tip_index: torch.Tensor,
    patristic_dist: np.ndarray,
    tau: float = 1.0,
):
    """
    y_pred: [B, Y]
    tip_index: [B] (indices into global tree)
    patristic_dist: [N, N] numpy
    """
    device = y_pred.device
    idx = tip_index
    D = torch.tensor(patristic_dist, dtype=torch.float32, device=device)
    D = D[idx][:, idx]  # [B, B]
    W = torch.exp(-D / (tau + 1e-8))  # similarity
    Y = y_pred
    diff = ((Y.unsqueeze(1) - Y.unsqueeze(0)) ** 2).mean(dim=-1)  # [B, B]
    mask = ~torch.eye(Y.size(0), dtype=torch.bool, device=device)
    return (W[mask] * diff[mask]).mean()


# -----------------------------
# Full model wrapper
# -----------------------------
class GenotypeToPhenotype(nn.Module):
    def __init__(
        self,
        token_dim: int,
        prior_dim: int,
        tree_in_dim: int,
        y_dim: int,
        use_structure: bool = False,
        struct_s_dim: int = 21,
        struct_v_dim: int = 6,
        seq_out: int = 256,
        tree_out: int = 256,
        struct_out: int = 128,
    ):
        super().__init__()
        self.seq_enc = SeqVariantEncoder(token_dim, prior_dim, out_dim=seq_out)
        self.tree_enc = TreeContextEncoder(in_dim=tree_in_dim, hidden=tree_out, layers=2)

        self.use_structure = use_structure
        if use_structure:
            self.gvp = GVPStub(struct_s_dim, struct_v_dim, s_hidden=64, v_hidden=64, iters=2)
            self.struct_pool = mlp([128, struct_out])
        else:
            self.gvp = None
            self.struct_pool = None

        self.G = Generator(seq_out, tree_out, struct_out if use_structure else 0, y_dim=y_dim)
        self.D = Discriminator(seq_out, tree_out, y_dim=y_dim)

    # Generator pass for a single sample
    def forward_G(
        self,
        onehot: torch.Tensor,       # [L, A]
        priors: torch.Tensor,       # [L, P]
        tip_id: int,                # int
        node_init: torch.Tensor,    # [N, F_t]
        edge_index: torch.Tensor,   # [2, E]
        edge_len: torch.Tensor,     # [E]
        struct_s: Optional[torch.Tensor] = None,   # [R, S]
        struct_v: Optional[torch.Tensor] = None,   # [R, V]
        struct_edges: Optional[torch.Tensor] = None,  # [2, E_p]
    ):
        seq_emb = self.seq_enc(onehot, priors)  # [E_s]
        node_emb = self.tree_enc(node_init, edge_index, edge_len)  # [N, E_t]
        tree_emb = node_emb[tip_id]  # [E_t]

        if self.use_structure:
            res_emb = self.gvp(struct_s, struct_v, struct_edges)  # [R, 128]
            struct_emb = self.struct_pool(res_emb.mean(dim=0))    # [E_p]
        else:
            struct_emb = torch.zeros(0, device=seq_emb.device)

        y_hat, e_hat = self.G(seq_emb, tree_emb, struct_emb)
        return (seq_emb, tree_emb), (y_hat, e_hat)

    # Discriminator pass for a batch
    def forward_D(self, seq_emb: torch.Tensor, tree_emb: torch.Tensor, y: torch.Tensor):
        return self.D(seq_emb, tree_emb, y)


# -----------------------------
# Training loop (one epoch)
# -----------------------------
def train_one_epoch(
    model: GenotypeToPhenotype,
    loader,
    optim_G,
    optim_D,
    device: str = "cpu",
    task: str = "regression",
    gan_mode: str = "ls",
    λ_phylo: float = 0.1,
):
    model.train()
    log = {"L_D": 0.0, "L_G": 0.0, "L_sup": 0.0, "L_phylo": 0.0}
    steps = 0

    for batch in loader:
        steps += 1
        batch = to_device(batch, device)

        seq_onehot = batch["onehot"]            # list[Tensor [L,A]]
        seq_priors = batch["priors"]            # list[Tensor [L,P]]
        tip_ids = batch["tip_ids"]              # [B]
        y_true = batch["y_true"]                # [B, y_dim]
        node_init = batch["tree_node_init"]     # [N, F_t]
        edge_index = batch["tree_edge_index"]   # [2, E]
        edge_len = batch["tree_edge_len"]       # [E]
        pat_dist = batch["patristic_dist"]      # [N, N] (numpy or tensor)

        struct_s = batch.get("struct_s", [None] * len(seq_onehot))
        struct_v = batch.get("struct_v", [None] * len(seq_onehot))
        struct_edges = batch.get("struct_edges", [None] * len(seq_onehot))

        seq_embs, tree_embs, y_hats = [], [], []
        for i in range(len(seq_onehot)):
            (se, te), (y_hat, _e) = model.forward_G(
                onehot=seq_onehot[i],
                priors=seq_priors[i],
                tip_id=int(tip_ids[i].item()) if torch.is_tensor(tip_ids[i]) else int(tip_ids[i]),
                node_init=node_init,
                edge_index=edge_index,
                edge_len=edge_len,
                struct_s=struct_s[i],
                struct_v=struct_v[i],
                struct_edges=struct_edges[i],
            )
            seq_embs.append(se)
            tree_embs.append(te)
            y_hats.append(y_hat)

        seq_embs = torch.stack(seq_embs, dim=0)  # [B, E_s]
        tree_embs = torch.stack(tree_embs, dim=0)  # [B, E_t]
        y_hat = torch.stack(y_hats, dim=0)  # [B, y_dim]

        # --- Discriminator ---
        optim_D.zero_grad(set_to_none=True)
        d_real = model.forward_D(seq_embs.detach(), tree_embs.detach(), y_true)
        d_fake = model.forward_D(seq_embs.detach(), tree_embs.detach(), y_hat.detach())
        L_D, L_Gadv = gan_losses(d_real, d_fake, mode=gan_mode)
        L_D.backward()
        optim_D.step()

        # --- Generator ---
        optim_G.zero_grad(set_to_none=True)
        L_sup = supervised_loss(y_hat, y_true, task=task)
        L_phy = phylo_contrastive(y_hat, tip_ids if torch.is_tensor(tip_ids) else torch.tensor(tip_ids, device=device),
                                  pat_dist)
        L_G = L_sup + L_Gadv + λ_phylo * L_phy
        L_G.backward()
        optim_G.step()

        log["L_D"] += float(L_D.item())
        log["L_G"] += float(L_G.item())
        log["L_sup"] += float(L_sup.item())
        log["L_phylo"] += float(L_phy.item())

    for k in log:
        log[k] /= max(1, steps)
    return log
