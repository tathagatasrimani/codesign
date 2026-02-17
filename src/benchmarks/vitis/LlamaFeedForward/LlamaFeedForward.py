import torch
import torch.nn as nn
from torch.nn import functional as F


class LlamaFeedForward(nn.Module):
    """LLaMA 3 SwiGLU feedforward block."""
    def __init__(self, dim=256, ffn_dim=1024):
        super().__init__()
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w3 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
