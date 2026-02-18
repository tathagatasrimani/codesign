import torch
import torch.nn as nn
import torch.nn.functional as F


class OpenVLALMDecodeAttention(nn.Module):
    """OpenVLA language model decode-phase attention with KV cache.

    After prefill (vision + instruction tokens), OpenVLA autoregressively
    decodes 7 discrete action tokens.  Each decode step attends over the
    KV cache populated during prefill.

    Uses the split-attention trick (no torch.cat for cache update):
    cache and new-token attention terms computed separately and summed.

    Real Llama 2 7B dimensions (OpenVLA LM):
      hidden=4096, n_heads=32, head_dim=128, cache_len=512.
    """

    def __init__(self, hidden=4096, n_heads=32, head_dim=128, cache_len=512):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.cache_len = cache_len

        self.wq = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, hidden, bias=False)

    def forward(self, x, k_cache, v_cache):
        """
        Args:
            x:       (batch, 1, hidden)  — new action token
            k_cache: (batch, n_heads, cache_len, head_dim)
            v_cache: (batch, n_heads, cache_len, head_dim)
        Returns:
            (batch, 1, hidden)
        """
        bsz = x.shape[0]

        q = self.wq(x).view(bsz, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k_new = self.wk(x).view(bsz, 1, self.n_heads, self.head_dim).transpose(1, 2)
        v_new = self.wv(x).view(bsz, 1, self.n_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / (self.head_dim ** 0.5)

        # Attend over KV cache (vision + instruction context)
        cache_scores = torch.matmul(q, k_cache.transpose(-2, -1)) * scale
        cache_probs = F.softmax(cache_scores, dim=-1)
        cache_out = torch.matmul(cache_probs, v_cache)

        # Attend over new token K/V (keeps wk, wv live)
        new_scores = torch.matmul(q, k_new.transpose(-2, -1)) * scale
        new_probs = F.softmax(new_scores, dim=-1)
        new_out = torch.matmul(new_probs, v_new)

        out = cache_out + new_out
        out = out.transpose(1, 2).contiguous().view(bsz, 1, -1)
        return self.wo(out)
