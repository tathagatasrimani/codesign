import torch
import torch.nn as nn
import torch.nn.functional as F


# Fixed predictor sequence length for torch-mlir traceability.
# Context patches (~25% unmasked) + mask tokens for target positions.
# 16 frames × 196 patches = 3136 total video positions.
PRED_SEQ_LEN = 3136


class VJEPAPredictorAttention(nn.Module):
    """V-JEPA predictor self-attention block.

    V-JEPA is trained with a joint-embedding predictive objective.  After
    the context encoder produces representations for the visible (unmasked)
    patches, a narrower predictor transformer predicts the representations
    of the masked target patches.

    Predictor input: encoded context tokens projected to pred_hidden +
    learnable mask tokens at target positions — all 3136 video patch positions.

    The predictor uses standard bidirectional (non-causal) self-attention.
    RoPE is not used in ViT-based predictors.

    Real V-JEPA predictor dimensions:
      pred_hidden=384, n_heads=12, head_dim=32, n_layers=12.
    """

    def __init__(self, pred_hidden=384, n_heads=12, head_dim=32):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim

        self.wq = nn.Linear(pred_hidden, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(pred_hidden, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(pred_hidden, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, pred_hidden, bias=False)

    def forward(self, x):
        """
        Args:
            x: (batch, PRED_SEQ_LEN, pred_hidden)
               — context embeddings (projected to pred_hidden) + mask tokens
        Returns:
            (batch, PRED_SEQ_LEN, pred_hidden)
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
