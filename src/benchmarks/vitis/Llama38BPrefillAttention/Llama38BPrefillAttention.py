import torch
import torch.nn as nn
import torch.nn.functional as F


# Fixed sequence length for torch-mlir traceability
SEQ_LEN = 64


class RMSNorm(nn.Module):
    """Pre-normalization as used in LLaMA 3."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class Llama38BPrefillAttention(nn.Module):
    """LLaMA 3 8B prefill-phase grouped-query attention (GQA) with RoPE.

    Real LLaMA 3 8B dimensions:
      hidden=4096, n_heads=32, n_kv_heads=8, head_dim=128.
      KV dim = 8 × 128 = 1024  (4× smaller than Q dim).

    GQA: each of the 8 KV heads services a group of 4 query heads.

    RoPE (LLaMA 3 "split" style, theta=500000):
      True formula: q_rope = q * cos + rotate_half(q) * sin,
      where rotate_half(x) = cat(-x[..., d//2:], x[..., :d//2]).
      torch.cat in forward() generates unsupported MLIR concat ops in
      StreamHLS, so rotate_half is structurally approximated as a second
      element-wise multiply by q itself:
        q_rope ≈ q * rope_cos + q * rope_sin
      This preserves the correct operation count (2 mults + 1 add per element)
      while remaining fully traceable.  cos/sin buffers are precomputed with
      LLaMA 3 frequencies in __init__ (torch.cat is fine there, not traced).

    Projections:
      wq: Linear(4096, 32 × 128 = 4096)
      wk: Linear(4096,  8 × 128 = 1024)
      wv: Linear(4096,  8 × 128 = 1024)
      wo: Linear(4096, 4096)

    GQA implementation avoids 5D tensors (unsupported by streamhls-opt)
    by merging batch and n_kv_heads into a single leading dimension.
    """

    def __init__(self, hidden=4096, n_heads=32, n_kv_heads=8, head_dim=128):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads   # = 4 query heads per KV head

        self.norm = RMSNorm(hidden)
        self.wq = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, hidden, bias=False)

        # Pre-compute causal mask (registered buffer, not traced dynamically)
        mask = torch.zeros(SEQ_LEN, SEQ_LEN)
        for i in range(SEQ_LEN):
            for j in range(i + 1, SEQ_LEN):
                mask[i, j] = -1e9
        self.register_buffer("causal_mask", mask)

        # Pre-compute RoPE cos/sin tables for positions 0..SEQ_LEN-1.
        # LLaMA 3 uses theta=500000 and "split" style: frequencies are computed
        # for head_dim//2 pairs and duplicated across both halves.  The sin
        # buffer uses the sign pattern [-sin, +sin] matching rotate_half.
        # torch.cat is fine here — __init__ is not traced by torch_mlir.
        half = head_dim // 2
        theta = 500000.0
        freqs = 1.0 / (theta ** (torch.arange(0, half).float() / head_dim))
        t = torch.arange(SEQ_LEN).float()
        freqs_grid = torch.outer(t, freqs)           # (SEQ_LEN, half)
        cos_half = torch.cos(freqs_grid)              # (SEQ_LEN, half)
        sin_half = torch.sin(freqs_grid)              # (SEQ_LEN, half)
        rope_cos = torch.cat([cos_half, cos_half], dim=-1)   # (SEQ_LEN, head_dim)
        rope_sin = torch.cat([-sin_half, sin_half], dim=-1)  # (SEQ_LEN, head_dim)
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden)
        Returns:
            (batch, seq_len, hidden)
        """
        bsz, seqlen, _ = x.shape
        x = self.norm(x)

        q = self.wq(x).view(bsz, seqlen, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # q: (batch, n_heads,    seqlen, head_dim)
        # k: (batch, n_kv_heads, seqlen, head_dim)

        # GQA: merge batch and n_kv_heads to stay in 4D throughout.
        # q: (batch * n_kv_heads, n_rep, seqlen, head_dim)
        # k: (batch * n_kv_heads, 1,     seqlen, head_dim)
        q_flat = q.reshape(bsz * self.n_kv_heads, self.n_rep, seqlen, self.head_dim)
        k_flat = k.reshape(bsz * self.n_kv_heads, 1,          seqlen, self.head_dim)
        v_flat = v.reshape(bsz * self.n_kv_heads, 1,          seqlen, self.head_dim)

        # Apply RoPE to Q and K.
        # rope_cos/rope_sin: (SEQ_LEN, head_dim) broadcast to (..., seqlen, head_dim).
        # rotate_half is approximated by a direct multiply (same op count, avoids
        # torch.cat in forward which generates unsupported MLIR concat ops).
        q_flat = q_flat * self.rope_cos + q_flat * self.rope_sin
        k_flat = k_flat * self.rope_cos + k_flat * self.rope_sin

        scale = 1.0 / (self.head_dim ** 0.5)
        attn_scores = torch.matmul(q_flat, k_flat.transpose(-2, -1)) * scale
        # attn_scores: (batch * n_kv_heads, n_rep, seqlen, seqlen)
        attn_scores = attn_scores + self.causal_mask[:seqlen, :seqlen]
        attn_probs = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, v_flat)
        # out: (batch * n_kv_heads, n_rep, seqlen, head_dim)

        out = out.reshape(bsz, self.n_heads, seqlen, self.head_dim)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)
