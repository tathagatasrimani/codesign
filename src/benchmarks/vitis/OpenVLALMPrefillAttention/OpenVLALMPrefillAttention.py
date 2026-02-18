import torch
import torch.nn as nn
import torch.nn.functional as F


# Fixed context length for torch-mlir traceability.
# Projected vision tokens (256) + language instruction tokens (256) = 512.
SEQ_LEN = 512


class OpenVLALMPrefillAttention(nn.Module):
    """OpenVLA language model prefill-phase attention (Llama 2 7B backbone).

    OpenVLA's language model processes the concatenation of projected vision
    tokens and language instruction tokens during prefill.  The attention is
    standard multi-head self-attention with a causal mask.  RoPE is omitted
    for StreamHLS traceability.

    Real Llama 2 7B dimensions (OpenVLA LM):
      hidden=4096, n_heads=32, head_dim=128, n_layers=32.
    """

    def __init__(self, hidden=4096, n_heads=32, head_dim=128):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim

        self.wq = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, hidden, bias=False)

        # Pre-compute causal mask
        mask = torch.zeros(SEQ_LEN, SEQ_LEN)
        for i in range(SEQ_LEN):
            for j in range(i + 1, SEQ_LEN):
                mask[i, j] = -1e9
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden)  — vision + language tokens
        Returns:
            (batch, seq_len, hidden)
        """
        bsz, seqlen, _ = x.shape

        q = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / (self.head_dim ** 0.5)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_scores = attn_scores + self.causal_mask[:seqlen, :seqlen]
        attn_probs = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, v)

        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)
