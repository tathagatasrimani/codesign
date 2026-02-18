import torch
import torch.nn as nn
import torch.nn.functional as F


class OpenVLAVisionFeedForward(nn.Module):
    """OpenVLA vision encoder feedforward block (SigLIP-SO400M).

    SigLIP uses a gated feedforward (SwiGLU-like) with GELU activation:
      - fc1: Linear(hidden, ffn_intermediate)
      - fc2: Linear(ffn_intermediate, hidden)

    Real SigLIP-SO400M (ViT-L/14) dimensions:
      hidden=1152, ffn_intermediate=4304 (MLP ratio ≈ 3.74).
    """

    def __init__(self, hidden=1152, ffn_intermediate=4304):
        super().__init__()
        self.fc1 = nn.Linear(hidden, ffn_intermediate, bias=False)
        self.fc2 = nn.Linear(ffn_intermediate, hidden, bias=False)

    def forward(self, x):
        """
        Args:
            x: (batch, VISION_PATCHES, hidden)
        Returns:
            (batch, VISION_PATCHES, hidden)
        """
        return self.fc2(F.gelu(self.fc1(x)))
