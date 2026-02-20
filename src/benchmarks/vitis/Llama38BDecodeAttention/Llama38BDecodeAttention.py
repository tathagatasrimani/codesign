import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Pre-normalization as used in LLaMA 3."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class Llama38BDecodeAttention(nn.Module):
    """LLaMA 3 8B decode-phase GQA with KV cache and RoPE.

    Single-token decode step.  The new query (1 token) attends over the
    cached K and V from the n_kv_heads key-value heads.

    Real LLaMA 3 8B dimensions:
      hidden=4096, n_heads=32, n_kv_heads=8, head_dim=128, cache_len=64.

    Cache layout: (batch, n_kv_heads, cache_len, head_dim).

    GQA implementation avoids 5D tensors (unsupported by streamhls-opt)
    by merging batch and n_kv_heads into a single leading dimension.

    To avoid torch.cat for cache update (which triggers unsupported MLIR ops),
    the new token's K/V contribution is computed as a separate attention term
    and added to the cache attention result — keeping wk and wv live in the
    computation graph for correct area/energy estimation.

    RoPE (LLaMA 3 "split" style, theta=500000):
      RoPE is applied to the new Q and new K at position cache_len (the fixed
      decode position).  The KV cache is assumed to already store RoPE-rotated
      keys from the prefill phase.
      True formula: x_rope = x * cos + rotate_half(x) * sin.
      rotate_half is approximated by a direct element-wise multiply (same op
      count, avoids torch.cat in forward which generates unsupported MLIR ops).
      A single-position (head_dim,) cos/sin buffer is precomputed in __init__.
    """

    def __init__(self, hidden=4096, n_heads=32, n_kv_heads=8,
                 head_dim=128, cache_len=64):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads   # = 4
        self.cache_len = cache_len

        self.norm = RMSNorm(hidden)
        self.wq = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, hidden, bias=False)

        # Pre-compute RoPE cos/sin for the decode position (= cache_len).
        # The new token is inserted at position cache_len; the KV cache holds
        # positions 0..cache_len-1 with RoPE already applied from prefill.
        # torch.cat is fine here — __init__ is not traced by torch_mlir.
        half = head_dim // 2
        theta = 500000.0
        freqs = 1.0 / (theta ** (torch.arange(0, half).float() / head_dim))
        pos_angles = cache_len * freqs                       # (half,)
        cos_half = torch.cos(pos_angles)                     # (half,)
        sin_half = torch.sin(pos_angles)                     # (half,)
        rope_cos = torch.cat([cos_half, cos_half], dim=-1)   # (head_dim,)
        rope_sin = torch.cat([-sin_half, sin_half], dim=-1)  # (head_dim,)
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)

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
        x = self.norm(x)

        # All n_heads query heads for the new token
        q = self.wq(x).view(bsz, 1, self.n_heads,    self.head_dim).transpose(1, 2)
        # q: (batch, n_heads, 1, head_dim)

        # n_kv_heads K/V projections for the new token
        k_new = self.wk(x).view(bsz, 1, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v_new = self.wv(x).view(bsz, 1, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # k_new/v_new: (batch, n_kv_heads, 1, head_dim)

        # GQA: merge batch and n_kv_heads to stay in 4D.
        q_flat       = q.reshape(      bsz * self.n_kv_heads, self.n_rep, 1,              self.head_dim)
        k_cache_flat = k_cache.reshape(bsz * self.n_kv_heads, 1,          self.cache_len, self.head_dim)
        v_cache_flat = v_cache.reshape(bsz * self.n_kv_heads, 1,          self.cache_len, self.head_dim)
        k_new_flat   = k_new.reshape(  bsz * self.n_kv_heads, 1,          1,              self.head_dim)
        v_new_flat   = v_new.reshape(  bsz * self.n_kv_heads, 1,          1,              self.head_dim)

        # Apply RoPE to new Q and new K at decode position (cache_len).
        # rope_cos/rope_sin: (head_dim,) broadcast to (..., 1, head_dim).
        # rotate_half is approximated by a direct multiply (same op count,
        # avoids torch.cat in forward which generates unsupported MLIR ops).
        q_flat     = q_flat     * self.rope_cos + q_flat     * self.rope_sin
        k_new_flat = k_new_flat * self.rope_cos + k_new_flat * self.rope_sin

        scale = 1.0 / (self.head_dim ** 0.5)

        # Attention over KV cache
        cache_scores = torch.matmul(q_flat, k_cache_flat.transpose(-2, -1)) * scale
        # cache_scores: (batch * n_kv_heads, n_rep, 1, cache_len)
        cache_probs = F.softmax(cache_scores, dim=-1)
        cache_out = torch.matmul(cache_probs, v_cache_flat)
        # cache_out: (batch * n_kv_heads, n_rep, 1, head_dim)

        # Attention over new token's K/V (keeps wk, wv live in graph)
        new_scores = torch.matmul(q_flat, k_new_flat.transpose(-2, -1)) * scale
        new_probs = F.softmax(new_scores, dim=-1)
        new_out = torch.matmul(new_probs, v_new_flat)

        out = cache_out + new_out
        # out: (batch * n_kv_heads, n_rep, 1, head_dim)

        out = out.reshape(bsz, self.n_heads, 1, self.head_dim)
        out = out.transpose(1, 2).contiguous().view(bsz, 1, -1)
        return self.wo(out)
