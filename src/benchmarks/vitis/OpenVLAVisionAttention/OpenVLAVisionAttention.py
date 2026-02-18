import torch
import torch.nn as nn
import torch.nn.functional as F


# Fixed image patch count for torch-mlir traceability.
# SigLIP-SO400M processes 224×224 images with 14×14 patches: 16 × 16 = 256 patches.
VISION_PATCHES = 256


class OpenVLAVisionAttention(nn.Module):
    """OpenVLA vision encoder self-attention (SigLIP-SO400M ViT-L/14).

    OpenVLA uses a SigLIP-SO400M vision encoder based on ViT-L/14 at
    224×224 resolution, producing 256 patch tokens.  Each encoder layer
    applies standard multi-head self-attention without causal masking.

    Real SigLIP-SO400M (ViT-L/14) dimensions:
      hidden=1152, n_heads=16, head_dim=72, n_layers=27.
    """

    def __init__(self, hidden=1152, n_heads=16, head_dim=72):
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
