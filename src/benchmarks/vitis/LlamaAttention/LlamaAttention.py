import torch
import torch.nn as nn
import torch.nn.functional as F


# Fixed sequence length for torch-mlir traceability
SEQ_LEN = 64


class LlamaAttention(nn.Module):
    """LLaMA 3 Attention (simplified for StreamHLS compatibility).

    Captures the dominant compute structure of LLaMA attention:
    - Bias-free linear projections (wq, wk, wv, wo)
    - Multi-head scaled dot-product attention
    - Causal masking

    RoPE and GQA are omitted because they generate MLIR ops
    (strided slicing, torch.stack, repeat) that StreamHLS cannot lower.
    Their compute cost is negligible compared to the matmuls.
    """
    def __init__(self, dim=256, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        # Pre-compute causal mask (registered buffer, not traced dynamically)
        mask = torch.zeros(SEQ_LEN, SEQ_LEN)
        for i in range(SEQ_LEN):
            for j in range(i + 1, SEQ_LEN):
                mask[i, j] = -1e9
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        bsz, seqlen, _ = x.shape

        queries = self.wq(x)
        keys = self.wk(x)
        values = self.wv(x)

        queries = queries.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        values = values.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        scale = 1.0 / (self.head_dim ** 0.5)
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale
        attn_scores = attn_scores + self.causal_mask[:seqlen, :seqlen]
        attn_probs = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, values)

        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)
