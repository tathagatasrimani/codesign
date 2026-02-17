import torch
import torch.nn as nn
import torch.nn.functional as F


class LlamaDecodeAttention(nn.Module):
    """LLaMA 3 decode-phase attention with KV cache.

    During autoregressive decoding, each generated token attends to all
    previously computed keys and values (the KV cache).  This block models
    a single decode step:

      - Input: single new token embedding  x  (batch, 1, dim)
      - KV cache: pre-computed keys/values  k_cache, v_cache
                  each (batch, n_heads, cache_len, head_dim)
      - Projects x → q, then attends q against the full KV cache
      - No causal mask needed (single query token, all cache positions visible)

    The cache_len parameter sets the fixed KV cache size for StreamHLS
    traceability (defaults to the prompt length from prefill).
    """
    def __init__(self, dim=256, n_heads=4, cache_len=64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.cache_len = cache_len

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x, k_cache, v_cache):
        # x:       (batch, 1, dim)
        # k_cache: (batch, n_heads, cache_len, head_dim)
        # v_cache: (batch, n_heads, cache_len, head_dim)
        bsz = x.shape[0]

        # Project the new token
        q = self.wq(x)                          # (batch, 1, dim)
        k_new = self.wk(x)                      # (batch, 1, dim)
        v_new = self.wv(x)                      # (batch, 1, dim)

        # Reshape to multi-head layout
        q = q.view(bsz, 1, self.n_heads, self.head_dim).transpose(1, 2)
        # q: (batch, n_heads, 1, head_dim)

        k_new = k_new.view(bsz, 1, self.n_heads, self.head_dim).transpose(1, 2)
        v_new = v_new.view(bsz, 1, self.n_heads, self.head_dim).transpose(1, 2)
        # k_new, v_new: (batch, n_heads, 1, head_dim)

        # Append new K/V to cache: (batch, n_heads, cache_len+1, head_dim)
        k_full = torch.cat([k_cache, k_new], dim=2)
        v_full = torch.cat([v_cache, v_new], dim=2)

        # Scaled dot-product attention (no causal mask — single query token)
        scale = 1.0 / (self.head_dim ** 0.5)
        attn_scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale
        # attn_scores: (batch, n_heads, 1, cache_len+1)

        attn_probs = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, v_full)
        # out: (batch, n_heads, 1, head_dim)

        out = out.transpose(1, 2).contiguous().view(bsz, 1, -1)
        return self.wo(out)
