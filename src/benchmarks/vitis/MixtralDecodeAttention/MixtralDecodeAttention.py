import torch
import torch.nn as nn
import torch.nn.functional as F


class MixtralDecodeAttention(nn.Module):
    """Mixtral 8x7B decode-phase GQA with KV cache.

    Single-token decode step.  The new query (1 token) attends over the
    cached K and V from the n_kv_heads key-value heads.

    Real Mixtral 8x7B dimensions:
      hidden=4096, n_heads=32, n_kv_heads=8, head_dim=128, cache_len=512.

    Cache layout: (batch, n_kv_heads, cache_len, head_dim).

    To avoid torch.cat for cache update, the new token's K/V contribution is
    computed as a separate attention term and added to the cache attention
    result — keeping wk and wv in the live computation graph.
    """

    def __init__(self, hidden=4096, n_heads=32, n_kv_heads=8,
                 head_dim=128, cache_len=512):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads
        self.cache_len = cache_len

        self.wq = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, hidden, bias=False)

    def forward(self, x, k_cache, v_cache):
        """
        Args:
            x:       (batch, 1, hidden)
            k_cache: (batch, n_kv_heads, cache_len, head_dim)
            v_cache: (batch, n_kv_heads, cache_len, head_dim)
        Returns:
            (batch, 1, hidden)
        """
        bsz = x.shape[0]

        # Query: all n_heads
        q = self.wq(x).view(bsz, 1, self.n_heads, self.head_dim).transpose(1, 2)
        # q: (batch, n_heads, 1, head_dim)

        # New token K/V (n_kv_heads)
        k_new = self.wk(x).view(bsz, 1, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v_new = self.wv(x).view(bsz, 1, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # k_new/v_new: (batch, n_kv_heads, 1, head_dim)

        # GQA with 4D tensors: merge batch and n_kv_heads to avoid 5D tensors,
        # which are not supported by streamhls-opt.
        # q: (batch, n_heads=n_kv_heads*n_rep, 1, head_dim)
        #  → reshape to (batch * n_kv_heads, n_rep, 1, head_dim)
        q_flat = q.reshape(bsz * self.n_kv_heads, self.n_rep, 1, self.head_dim)
        k_cache_flat = k_cache.reshape(bsz * self.n_kv_heads, 1, self.cache_len, self.head_dim)
        v_cache_flat = v_cache.reshape(bsz * self.n_kv_heads, 1, self.cache_len, self.head_dim)
        k_new_flat = k_new.reshape(bsz * self.n_kv_heads, 1, 1, self.head_dim)
        v_new_flat = v_new.reshape(bsz * self.n_kv_heads, 1, 1, self.head_dim)

        scale = 1.0 / (self.head_dim ** 0.5)

        # Attention over cached K/V
        cache_scores = torch.matmul(q_flat, k_cache_flat.transpose(-2, -1)) * scale
        # cache_scores: (batch * n_kv_heads, n_rep, 1, cache_len)
        cache_probs = F.softmax(cache_scores, dim=-1)
        cache_out = torch.matmul(cache_probs, v_cache_flat)
        # cache_out: (batch * n_kv_heads, n_rep, 1, head_dim)

        # Attention over new token K/V (keeps wk, wv live)
        new_scores = torch.matmul(q_flat, k_new_flat.transpose(-2, -1)) * scale
        new_probs = F.softmax(new_scores, dim=-1)
        new_out = torch.matmul(new_probs, v_new_flat)

        out = cache_out + new_out  # (batch * n_kv_heads, n_rep, 1, head_dim)
        out = out.reshape(bsz, self.n_heads, 1, self.head_dim)
        out = out.transpose(1, 2).contiguous().view(bsz, 1, -1)
        return self.wo(out)
