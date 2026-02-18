import torch
import torch.nn as nn
import torch.nn.functional as F


# Total video patches = n_frames × patches_per_frame = 16 × 196 = 3136.
TOTAL_PATCHES = 3136


class VJEPATemporalAttention(nn.Module):
    """V-JEPA (ViT-H/16) temporal self-attention across all video patches.

    After spatial attention within each frame, temporal attention is applied
    over all tokens from all frames jointly (3136 = 16 frames × 196 patches).
    This captures long-range spatio-temporal dependencies.

    This block implements the temporal attention sub-step:
      - Input: (batch, TOTAL_PATCHES, hidden)  — all frames flattened
      - Standard multi-head self-attention (no causal mask; encoder context)
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
            x: (batch, TOTAL_PATCHES, hidden)
        Returns:
            (batch, TOTAL_PATCHES, hidden)
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
