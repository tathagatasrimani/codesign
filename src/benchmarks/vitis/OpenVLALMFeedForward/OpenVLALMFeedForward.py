import torch
import torch.nn as nn
import torch.nn.functional as F


class OpenVLALMFeedForward(nn.Module):
    """OpenVLA language model SwiGLU feedforward block (Llama 2 7B).

    Shared between prefill and decode phases; compiled separately for each
    phase to expose the correct input sequence length to StreamHLS.
    This version targets the prefill phase (batch, seq_len, hidden).

    Real Llama 2 7B dimensions (OpenVLA LM):
      hidden=4096, ffn_intermediate=11008.
    """

    def __init__(self, hidden=4096, ffn_intermediate=11008):
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
