import torch
import torch.nn as nn
import torch.nn.functional as F


# Fixed image patch count for torch-mlir traceability.
# DINOv2 ViT-L/14 at 224×224: 224/14 = 16 → 16 × 16 = 256 patches.
VISION_PATCHES = 256


class OpenVLADinoAttention(nn.Module):
    """OpenVLA DINOv2 vision encoder self-attention (ViT-L/14).

    OpenVLA uses a dual vision encoder: DINOv2 ViT-L/14 captures semantic
    and structural features while SigLIP captures vision-language aligned
    features.  Patch features from both encoders are concatenated per patch
    and projected into the language model embedding space.

    This block covers the DINOv2 encoder:
      - 256 patch tokens (16×16 patches for 224×224 input)
      - 24 encoder layers
      - Standard bidirectional multi-head self-attention (no causal mask)

    Real DINOv2 ViT-L/14 dimensions:
      hidden=1024, n_heads=16, head_dim=64, n_layers=24.
    """

    def __init__(self, hidden=1024, n_heads=16, head_dim=64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim

        self.wq = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, hidden, bias=False)

    def forward(self, x):
        """
        Args:
            x: (batch, VISION_PATCHES, hidden)
        Returns:
            (batch, VISION_PATCHES, hidden)
        """
        bsz, seqlen, _ = x.shape

        q = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / (self.head_dim ** 0.5)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, v)

        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)
