import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSeekPrefillDenseFFN(nn.Module):
    """DeepSeek-V3 dense SwiGLU feedforward block for prefill (layers 1-3).

    DeepSeek-V3 uses dense FFN (not MoE) in the first `first_k_dense_replace=3`
    transformer layers.  This avoids MoE routing overhead in the early layers
    where token representations are still being built up.

    The dense FFN uses the same SwiGLU gating as the Llama FFN but with a
    significantly wider intermediate dimension than the per-expert MoE intermediate:

    Real DeepSeek-V3 dimensions:
      hidden=7168, ffn_intermediate=18432 (dense, ~2.57× expansion).

    Compiled separately from DeepSeekDecodeDenseFFN to expose the full
    prompt sequence length to StreamHLS.
    """

    def __init__(self, hidden=7168, ffn_intermediate=18432):
        super().__init__()
        self.w1 = nn.Linear(hidden, ffn_intermediate, bias=False)
        self.w3 = nn.Linear(hidden, ffn_intermediate, bias=False)
        self.w2 = nn.Linear(ffn_intermediate, hidden, bias=False)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden)
        Returns:
            (batch, seq_len, hidden)
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
