import torch
import torch.nn as nn
import torch.nn.functional as F


class OpenVLAVisionProjection(nn.Module):
    """OpenVLA vision-to-language projection MLP.

    After both DINOv2 and SigLIP encoders process the image patches, their
    per-patch features are concatenated (dino_hidden + siglip_hidden = 2176)
    and projected into the language model embedding space (lm_hidden = 4096)
    via a two-layer MLP.  This produces 256 vision tokens directly compatible
    with the LM's embedding space for the subsequent prefill phase.

    Real OpenVLA dimensions:
      proj_in  = dino_hidden + siglip_hidden = 1024 + 1152 = 2176
      proj_mid = lm_hidden = 4096  (hidden layer = output dim)
      proj_out = lm_hidden = 4096
    """

    def __init__(self, proj_in=2176, lm_hidden=4096):
        super().__init__()
        self.fc1 = nn.Linear(proj_in, lm_hidden, bias=False)
        self.fc2 = nn.Linear(lm_hidden, lm_hidden, bias=False)

    def forward(self, x):
        """
        Args:
            x: (batch, n_patches, proj_in)  — concatenated DINOv2 + SigLIP features
        Returns:
            (batch, n_patches, lm_hidden)   — projected vision tokens for LM
        """
        return self.fc2(F.gelu(self.fc1(x)))
