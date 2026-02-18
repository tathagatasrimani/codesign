import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSeekPrefillMoE(nn.Module):
    """DeepSeek-V3 prefill-phase Mixture-of-Experts feedforward block.

    DeepSeek-V3 uses a fine-grained MoE with:
      - 256 routed experts + 1 shared expert (always active)
      - top-8 routing among routed experts per token
      - SwiGLU activation in each expert

    Real DeepSeek-V3 dimensions:
      hidden=7168, moe_intermediate=2048, n_routed_experts=256, top_k=8.

    StreamHLS simplification: dynamic top-k routing cannot be traced by
    torch-mlir.  Instead, we model the dominant compute as top_k separate
    expert FFNs (expert_0 … expert_{top_k-1}) plus one shared expert.
    Each expert has the correct weight dimensions.  A static gate linear
    is kept to model the router overhead (Linear(hidden, n_routed_experts)).
    Expert outputs are summed with uniform weights to keep all weights live
    in the computation graph.
    """

    def __init__(self, hidden=7168, moe_intermediate=2048,
                 n_routed_experts=256, top_k=8):
        super().__init__()
        self.hidden = hidden
        self.top_k = top_k
        self.n_routed_experts = n_routed_experts

        # Router (kept for weight / FLOPs accounting)
        self.gate = nn.Linear(hidden, n_routed_experts, bias=False)

        # Shared expert (always active)
        self.shared_w1 = nn.Linear(hidden, moe_intermediate, bias=False)
        self.shared_w3 = nn.Linear(hidden, moe_intermediate, bias=False)
        self.shared_w2 = nn.Linear(moe_intermediate, hidden, bias=False)

        # top_k routed expert FFNs (represent average active compute)
        self.routed_w1 = nn.Linear(hidden, moe_intermediate * top_k, bias=False)
        self.routed_w3 = nn.Linear(hidden, moe_intermediate * top_k, bias=False)
        self.routed_w2 = nn.Linear(moe_intermediate * top_k, hidden, bias=False)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden)
        Returns:
            (batch, seq_len, hidden)
        """
        # Router (not used to index — kept for weight accounting)
        _ = self.gate(x)  # (batch, seq, n_routed_experts)

        # Shared expert (SwiGLU)
        shared_out = self.shared_w2(
            F.silu(self.shared_w1(x)) * self.shared_w3(x)
        )

        # top_k routed experts (all computed in a single fused linear for
        # StreamHLS traceability, then summed back to hidden)
        routed_out = self.routed_w2(
            F.silu(self.routed_w1(x)) * self.routed_w3(x)
        )

        return shared_out + routed_out
