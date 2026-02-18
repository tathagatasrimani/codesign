import torch
import torch.nn as nn
import torch.nn.functional as F


# Fixed sequence length for torch-mlir traceability
SEQ_LEN = 512


class DeepSeekPrefillAttention(nn.Module):
    """DeepSeek-V3 prefill-phase Multi-head Latent Attention (MLA).

    DeepSeek-V3 uses MLA to compress the KV cache: rather than caching full
    K and V projections, it caches a low-rank latent vector c_kv from which
    K and V are reconstructed.  The Q projection is also rank-compressed.

    Simplified for StreamHLS traceability (RoPE omitted):
      - q_down: Linear(hidden, q_lora_rank)   — query compression
      - q_up:   Linear(q_lora_rank, n_heads * head_dim)  — query reconstruction
      - kv_down: Linear(hidden, kv_lora_rank)  — joint KV compression
      - k_up:   Linear(kv_lora_rank, n_heads * head_dim)
      - v_up:   Linear(kv_lora_rank, n_heads * head_dim)
      - o_proj: Linear(n_heads * head_dim, hidden)

    Real DeepSeek-V3 dimensions:
      hidden=7168, n_heads=128, head_dim=128,
      q_lora_rank=1536, kv_lora_rank=512.
    """

    def __init__(self, hidden=7168, n_heads=128, head_dim=128,
                 q_lora_rank=1536, kv_lora_rank=512):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.kv_lora_rank = kv_lora_rank

        # Query low-rank projection
        self.q_down = nn.Linear(hidden, q_lora_rank, bias=False)
        self.q_up = nn.Linear(q_lora_rank, n_heads * head_dim, bias=False)

        # Joint KV low-rank compression + per-head expansion
        self.kv_down = nn.Linear(hidden, kv_lora_rank, bias=False)
        self.k_up = nn.Linear(kv_lora_rank, n_heads * head_dim, bias=False)
        self.v_up = nn.Linear(kv_lora_rank, n_heads * head_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(n_heads * head_dim, hidden, bias=False)

        # Pre-compute causal mask
        mask = torch.zeros(SEQ_LEN, SEQ_LEN)
        for i in range(SEQ_LEN):
            for j in range(i + 1, SEQ_LEN):
                mask[i, j] = -1e9
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden)
        Returns:
            (batch, seq_len, hidden)
        """
        bsz, seqlen, _ = x.shape

        # Compressed query
        q = self.q_up(self.q_down(x))  # (batch, seq, n_heads * head_dim)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        # Compressed KV
        c_kv = self.kv_down(x)  # (batch, seq, kv_lora_rank)
        k = self.k_up(c_kv)     # (batch, seq, n_heads * head_dim)
        v = self.v_up(c_kv)

        k = k.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        scale = 1.0 / (self.head_dim ** 0.5)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_scores = attn_scores + self.causal_mask[:seqlen, :seqlen]
        attn_probs = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, v)

        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.o_proj(out)
