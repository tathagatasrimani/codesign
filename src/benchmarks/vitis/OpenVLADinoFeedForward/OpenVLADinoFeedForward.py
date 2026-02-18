import torch
import torch.nn as nn
import torch.nn.functional as F


class OpenVLADinoFeedForward(nn.Module):
    """OpenVLA DINOv2 vision encoder feedforward block (ViT-L/14).

    Two-layer MLP with GELU activation in each DINOv2 encoder layer.

    Real DINOv2 ViT-L/14 dimensions:
      hidden=1024, ffn_intermediate=4096 (MLP ratio 4:1).
    """

    def __init__(self, hidden=1024, ffn_intermediate=4096):
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
