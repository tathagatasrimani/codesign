import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSeekDecodeMoE(nn.Module):
    """DeepSeek-V3 decode-phase MoE feedforward block (single token).

    Structurally identical to DeepSeekPrefillMoE but operates on a single
    token (batch, 1, hidden) during autoregressive decoding.  Compiled
    separately so StreamHLS optimises for the single-token input shape.

    Real DeepSeek-V3 dimensions:
      hidden=7168, moe_intermediate=2048, n_routed_experts=256, top_k=8.
    """

    def __init__(self, hidden=7168, moe_intermediate=2048,
                 n_routed_experts=256, top_k=8):
        super().__init__()
        self.n_routed_experts = n_routed_experts
        self.top_k = top_k

        # Router
        self.gate = nn.Linear(hidden, n_routed_experts, bias=False)

        # Shared expert (SwiGLU)
        self.shared_w1 = nn.Linear(hidden, moe_intermediate, bias=False)
        self.shared_w3 = nn.Linear(hidden, moe_intermediate, bias=False)
        self.shared_w2 = nn.Linear(moe_intermediate, hidden, bias=False)

        # top_k routed expert FFNs (fused)
        self.routed_w1 = nn.Linear(hidden, moe_intermediate * top_k, bias=False)
        self.routed_w3 = nn.Linear(hidden, moe_intermediate * top_k, bias=False)
        self.routed_w2 = nn.Linear(moe_intermediate * top_k, hidden, bias=False)

    def forward(self, x):
        """
        Args:
            x: (batch, 1, hidden)
        Returns:
            (batch, 1, hidden)
        """
        _ = self.gate(x)

        shared_out = self.shared_w2(
            F.silu(self.shared_w1(x)) * self.shared_w3(x)
        )
        routed_out = self.routed_w2(
            F.silu(self.routed_w1(x)) * self.routed_w3(x)
        )
        return shared_out + routed_out
