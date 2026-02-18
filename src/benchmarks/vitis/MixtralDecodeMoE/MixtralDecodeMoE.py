import torch
import torch.nn as nn
import torch.nn.functional as F


class MixtralDecodeMoE(nn.Module):
    """Mixtral 8x7B decode-phase MoE feedforward block (single token).

    Structurally identical to MixtralPrefillMoE but operates on a single
    token (batch, 1, hidden).  Compiled separately for single-token input.

    Real Mixtral 8x7B dimensions:
      hidden=4096, ffn_intermediate=14336, n_experts=8, top_k=2.
    """

    def __init__(self, hidden=4096, ffn_intermediate=14336, n_experts=8, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        self.gate = nn.Linear(hidden, n_experts, bias=False)
        self.w1 = nn.Linear(hidden, ffn_intermediate * top_k, bias=False)
        self.w3 = nn.Linear(hidden, ffn_intermediate * top_k, bias=False)
        self.w2 = nn.Linear(ffn_intermediate * top_k, hidden, bias=False)

    def forward(self, x):
        """
        Args:
            x: (batch, 1, hidden)
        Returns:
            (batch, 1, hidden)
        """
        _ = self.gate(x)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
