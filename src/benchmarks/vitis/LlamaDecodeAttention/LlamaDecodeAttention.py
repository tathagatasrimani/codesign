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
        # k_cache: (batch, n_heads, cache_len, head_dim)  — pre-sized to include new token slot
        # v_cache: (batch, n_heads, cache_len, head_dim)
        bsz = x.shape[0]

        # Project the new token
        q = self.wq(x)                          # (batch, 1, dim)
        k_new = self.wk(x)                      # (batch, 1, dim)
        v_new = self.wv(x)                      # (batch, 1, dim)

        # Reshape to multi-head layout
        q = q.view(bsz, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k_new = k_new.view(bsz, 1, self.n_heads, self.head_dim).transpose(1, 2)
        v_new = v_new.view(bsz, 1, self.n_heads, self.head_dim).transpose(1, 2)
        # q/k_new/v_new: (batch, n_heads, 1, head_dim)

        # Attend over KV cache and the new token separately, then sum.
        # This avoids torch.cat (which triggers bufferization.to_tensor in
        # MLIR, unsupported by StreamHLS) while keeping wk/wv in the live
        # computation graph for correct area/energy estimation.
        scale = 1.0 / (self.head_dim ** 0.5)

        # Attention over cached K/V
        cache_scores = torch.matmul(q, k_cache.transpose(-2, -1)) * scale
        # cache_scores: (batch, n_heads, 1, cache_len)
        cache_probs = F.softmax(cache_scores, dim=-1)
        cache_out = torch.matmul(cache_probs, v_cache)

        # Attention over new token's K/V (keeps wk, wv live)
        new_scores = torch.matmul(q, k_new.transpose(-2, -1)) * scale
        # new_scores: (batch, n_heads, 1, 1)
        new_probs = F.softmax(new_scores, dim=-1)
        new_out = torch.matmul(new_probs, v_new)

        out = cache_out + new_out
        # out: (batch, n_heads, 1, head_dim)

        out = out.transpose(1, 2).contiguous().view(bsz, 1, -1)
        return self.wo(out)
