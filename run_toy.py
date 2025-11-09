# run_toy.py
# Runs a tiny forward pass using the scaffold (no training).
from g2p_model import (
    SequenceEncoder,
    build_prior_tensor,
    GenotypeToPhenotype,
)
import numpy as np
import torch


def main():
    # ----- 1) toy sequence + priors -----
    enc = SequenceEncoder(kind="protein")
    seq = "MKTFFVLLL"
    onehot = enc.one_hot(seq)  # [L, 21]
    L = onehot.size(0)

    priors = build_prior_tensor(
        length=L,
        treesaap_props=np.random.randn(L, 8),   # dummy TreeSAAP-like features
        polyphen_scores=np.random.rand(L, 1),   # dummy PolyPhen-like probability
    )  # [L, 9]

    # ----- 2) toy phylogenetic tree with 3 tips (0-1-2 chain) -----
    N, F_t = 3, 8
    node_init = torch.zeros(N, F_t)  # could hold tip summaries; zeros are fine for demo
    edge_index = torch.tensor([[0, 1], [1, 2]])  # directed
    # make it bidirectional:
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # [2, 4]
    edge_len = torch.tensor([0.1, 0.2, 0.1, 0.2], dtype=torch.float32)

    # ----- 3) build model and forward a single sample -----
    y_dim = 1
    model = GenotypeToPhenotype(
        token_dim=onehot.size(1),
        prior_dim=priors.size(1),
        tree_in_dim=F_t,
        y_dim=y_dim,
        use_structure=False,
    )

    model.eval()  # inference mode
    with torch.no_grad():  # no gradients during inference
        (seq_emb, tree_emb), (y_hat, e_hat) = model.forward_G(
            onehot=onehot,
            priors=priors,
            tip_id=1,
            node_init=node_init,
            edge_index=edge_index,
            edge_len=edge_len,
        )

    print("OK — forward pass completed.")
    print("seq_emb shape:", tuple(seq_emb.shape))
    print("tree_emb shape:", tuple(tree_emb.shape))
    print("y_hat shape:", tuple(y_hat.shape), "value:", y_hat.item())
    print("effect score (toy):", e_hat.item())


if __name__ == "__main__":
    main()
