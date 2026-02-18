import torch
import torch.nn as nn
import torch.nn.functional as F


# Fixed sequence length for torch-mlir traceability
SEQ_LEN = 512


class MixtralPrefillAttention(nn.Module):
    """Mixtral 8x7B prefill-phase grouped-query attention (GQA).

    Mixtral uses GQA with 32 query heads and 8 key/value heads (4:1 ratio).
    Each KV head services a group of 4 query heads.  RoPE is omitted for
    StreamHLS traceability.

    Real Mixtral 8x7B dimensions:
      hidden=4096, n_heads=32, n_kv_heads=8, head_dim=128.

    Compute structure:
      - wq: Linear(hidden, n_heads * head_dim)
      - wk: Linear(hidden, n_kv_heads * head_dim)
      - wv: Linear(hidden, n_kv_heads * head_dim)
      - wo: Linear(n_heads * head_dim, hidden)
      - KV heads are repeated (n_heads // n_kv_heads) times for the dot
        product (modelled as a scaled matmul over repeated KV).
    """

    def __init__(self, hidden=4096, n_heads=32, n_kv_heads=8, head_dim=128):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads  # repetitions per KV head

        self.wq = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
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
            x: (batch, seq_len, hidden)
        Returns:
            (batch, seq_len, hidden)
        """
        bsz, seqlen, _ = x.shape

        q = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # q: (batch, n_heads, seqlen, head_dim)
        # k: (batch, n_kv_heads, seqlen, head_dim)

        # GQA with 4D tensors: merge batch and n_kv_heads to avoid 5D tensors,
        # which are not supported by streamhls-opt.
        # q: (batch, n_kv_heads * n_rep, seqlen, head_dim)
        #  → reshape to (batch * n_kv_heads, n_rep, seqlen, head_dim)
        q_flat = q.reshape(bsz * self.n_kv_heads, self.n_rep, seqlen, self.head_dim)
        k_flat = k.reshape(bsz * self.n_kv_heads, 1, seqlen, self.head_dim)
        v_flat = v.reshape(bsz * self.n_kv_heads, 1, seqlen, self.head_dim)

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
