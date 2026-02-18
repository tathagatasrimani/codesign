import torch
import torch.nn as nn
import torch.nn.functional as F


class MixtralPrefillMoE(nn.Module):
    """Mixtral 8x7B prefill-phase Mixture-of-Experts feedforward block.

    Mixtral uses a sparse MoE with:
      - 8 experts, top-2 active per token
      - SwiGLU activation in each expert (w1, w2, w3)
      - Router: Linear(hidden, n_experts)

    Real Mixtral 8x7B dimensions:
      hidden=4096, ffn_intermediate=14336, n_experts=8, top_k=2.

    StreamHLS simplification: compute both active experts as two separate
    SwiGLU FFNs and sum their outputs.  A static router linear is kept for
    FLOPs/weight accounting.  This accurately models the per-token compute
    (2 × SwiGLU at full intermediate width) without dynamic routing.
    """

    def __init__(self, hidden=4096, ffn_intermediate=14336, n_experts=8, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        # Router (kept for weight / FLOPs accounting)
        self.gate = nn.Linear(hidden, n_experts, bias=False)

        # top_k expert SwiGLU FFNs (fused into single linear for traceability)
        # Each expert: w1(hidden→ffn_intermediate), w3(hidden→ffn_intermediate),
        #              w2(ffn_intermediate→hidden).
        # Fuse top_k experts along the intermediate dimension.
        self.w1 = nn.Linear(hidden, ffn_intermediate * top_k, bias=False)
        self.w3 = nn.Linear(hidden, ffn_intermediate * top_k, bias=False)
        self.w2 = nn.Linear(ffn_intermediate * top_k, hidden, bias=False)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden)
        Returns:
            (batch, seq_len, hidden)
        """
        # Router (not used for dynamic dispatch — kept for accounting)
        _ = self.gate(x)  # (batch, seq, n_experts)

        # top_k expert compute: SwiGLU over fused intermediate
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
