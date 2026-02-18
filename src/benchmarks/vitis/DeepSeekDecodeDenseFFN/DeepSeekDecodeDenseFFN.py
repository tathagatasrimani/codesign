import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSeekDecodeDenseFFN(nn.Module):
    """DeepSeek-V3 dense SwiGLU feedforward block for decode (layers 1-3).

    Structurally identical to DeepSeekPrefillDenseFFN but compiled separately
    so StreamHLS optimises for the single-token (batch, 1, hidden) input shape
    used during autoregressive decoding.

    Real DeepSeek-V3 dimensions:
      hidden=7168, ffn_intermediate=18432.
    """

    def __init__(self, hidden=7168, ffn_intermediate=18432):
        super().__init__()
        self.w1 = nn.Linear(hidden, ffn_intermediate, bias=False)
        self.w3 = nn.Linear(hidden, ffn_intermediate, bias=False)
        self.w2 = nn.Linear(ffn_intermediate, hidden, bias=False)

    def forward(self, x):
        """
        Args:
            x: (batch, 1, hidden)
        Returns:
            (batch, 1, hidden)
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
