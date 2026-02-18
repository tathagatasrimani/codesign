import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSeekDecodeAttention(nn.Module):
    """DeepSeek-V3 decode-phase MLA with compressed KV cache.

    In MLA, the KV cache stores the low-rank latent c_kv (kv_lora_rank dims)
    rather than full K/V tensors.  During decode, a single new token is
    projected down to c_kv, then K/V are reconstructed.  The query for the
    new token attends over the reconstructed K from the entire cache.

    StreamHLS simplification: cache K/V in reconstructed form (n_heads *
    head_dim) to avoid the torch.cat needed for cache update.  Two separate
    attention terms (over cached K/V and new token's K/V) are summed using
    the same trick as LlamaDecodeAttention.

    Real DeepSeek-V3 dimensions:
      hidden=7168, n_heads=128, head_dim=128,
      q_lora_rank=1536, kv_lora_rank=512, cache_len=512.
    """

    def __init__(self, hidden=7168, n_heads=128, head_dim=128,
                 q_lora_rank=1536, kv_lora_rank=512, cache_len=512):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.cache_len = cache_len

        # Query low-rank projection
        self.q_down = nn.Linear(hidden, q_lora_rank, bias=False)
        self.q_up = nn.Linear(q_lora_rank, n_heads * head_dim, bias=False)

        # KV compression + reconstruction (same weights as prefill)
        self.kv_down = nn.Linear(hidden, kv_lora_rank, bias=False)
        self.k_up = nn.Linear(kv_lora_rank, n_heads * head_dim, bias=False)
        self.v_up = nn.Linear(kv_lora_rank, n_heads * head_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(n_heads * head_dim, hidden, bias=False)

    def forward(self, x, k_cache, v_cache):
        """
        Args:
            x:       (batch, 1, hidden)  — new token
            k_cache: (batch, n_heads, cache_len, head_dim)  — cached keys
            v_cache: (batch, n_heads, cache_len, head_dim)  — cached values
        Returns:
            (batch, 1, hidden)
        """
        bsz = x.shape[0]

        # New token query (via low-rank projection)
        q = self.q_up(self.q_down(x))  # (batch, 1, n_heads * head_dim)
        q = q.view(bsz, 1, self.n_heads, self.head_dim).transpose(1, 2)

        # New token KV (reconstructed from compressed latent)
        c_kv = self.kv_down(x)          # (batch, 1, kv_lora_rank)
        k_new = self.k_up(c_kv)         # (batch, 1, n_heads * head_dim)
        v_new = self.v_up(c_kv)
        k_new = k_new.view(bsz, 1, self.n_heads, self.head_dim).transpose(1, 2)
        v_new = v_new.view(bsz, 1, self.n_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / (self.head_dim ** 0.5)

        # Attend over KV cache (prefix)
        cache_scores = torch.matmul(q, k_cache.transpose(-2, -1)) * scale
        cache_probs = F.softmax(cache_scores, dim=-1)
        cache_out = torch.matmul(cache_probs, v_cache)

        # Attend over new token's K/V (keeps kv_down, k_up, v_up live)
        new_scores = torch.matmul(q, k_new.transpose(-2, -1)) * scale
        new_probs = F.softmax(new_scores, dim=-1)
        new_out = torch.matmul(new_probs, v_new)

        out = cache_out + new_out  # (batch, n_heads, 1, head_dim)
        out = out.transpose(1, 2).contiguous().view(bsz, 1, -1)
        return self.o_proj(out)
