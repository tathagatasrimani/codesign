import torch
import torch.nn as nn
import torch.nn.functional as F


# Fixed spatial patch count for torch-mlir traceability.
# For a 224×224 video frame with 16×16 patches: 14 × 14 = 196 patches/frame.
SPATIAL_PATCHES = 196


class VJEPASpatialAttention(nn.Module):
    """V-JEPA (ViT-H/16) spatial self-attention over patches in one frame.

    V-JEPA uses factorised space-time attention: spatial attention first
    processes the 196 patch tokens within each video frame independently,
    then temporal attention relates patches across frames.

    This block implements the spatial attention sub-step:
      - Input: (batch * n_frames, SPATIAL_PATCHES, hidden)
      - Standard multi-head self-attention (no causal mask)
      - Bias-free linear projections

    Real ViT-H/16 dimensions (V-JEPA encoder):
      hidden=1280, n_heads=16, head_dim=80.
    """

    def __init__(self, hidden=1280, n_heads=16, head_dim=80):
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
            x: (batch, SPATIAL_PATCHES, hidden)
               (caller collapses batch × frames dimension before calling)
        Returns:
            (batch, SPATIAL_PATCHES, hidden)
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
