import torch
import torch.nn as nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """Pre-normalization as used in LLaMA 3."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class LlamaFeedForward(nn.Module):
    """LLaMA 3 SwiGLU feedforward block with pre-RMSNorm."""
    def __init__(self, dim=256, ffn_dim=1024):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w3 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
