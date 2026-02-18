import torch
import torch.nn as nn
import torch.nn.functional as F


class VJEPAFeedForward(nn.Module):
    """V-JEPA (ViT-H/16) MLP feedforward block.

    Standard two-layer MLP with GELU activation, as used in ViT-H:
      - fc1: Linear(hidden, ffn_intermediate)
      - fc2: Linear(ffn_intermediate, hidden)

    Operates on all video tokens (batch, total_patches, hidden) after
    temporal attention.

    Real ViT-H/16 dimensions (V-JEPA encoder):
      hidden=1280, ffn_intermediate=5120 (MLP ratio 4:1).
    """

    def __init__(self, hidden=1280, ffn_intermediate=5120):
        super().__init__()
        self.fc1 = nn.Linear(hidden, ffn_intermediate, bias=False)
        self.fc2 = nn.Linear(ffn_intermediate, hidden, bias=False)

    def forward(self, x):
        """
        Args:
            x: (batch, total_patches, hidden)
        Returns:
            (batch, total_patches, hidden)
        """
        return self.fc2(F.gelu(self.fc1(x)))
