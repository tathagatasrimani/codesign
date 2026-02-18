import os
import sys
import torch
import torch.nn as nn

# Add parent vitis directory so sibling sub-block packages are importable.
_here = os.path.dirname(os.path.abspath(__file__))
_vitis = os.path.dirname(_here)
if _vitis not in sys.path:
    sys.path.insert(0, _vitis)

from MixtralPrefillAttention.MixtralPrefillAttention import MixtralPrefillAttention
from MixtralPrefillMoE.MixtralPrefillMoE import MixtralPrefillMoE
from MixtralDecodeAttention.MixtralDecodeAttention import MixtralDecodeAttention
from MixtralDecodeMoE.MixtralDecodeMoE import MixtralDecodeMoE


class MixtralInference(nn.Module):
    """Mixtral 8x7B full inference: prefill phase + autoregressive decode.

    Mixtral 8x7B is a sparse mixture-of-experts transformer:
      - 32 transformer layers
      - Grouped-query attention (GQA): 32 Q-heads, 8 KV-heads, head_dim=128
      - Sparse MoE FFN: 8 experts, top-2 active, ffn_intermediate=14336

    Sub-block attributes matched to block_types in the system YAML:
      prefill_attention -> MixtralPrefillAttention  (GQA prefill)
      prefill_moe       -> MixtralPrefillMoE        (2-of-8 MoE prefill)
      decode_attention  -> MixtralDecodeAttention   (GQA decode + KV cache)
      decode_moe        -> MixtralDecodeMoE         (2-of-8 MoE decode)
    """

    def __init__(self, hidden=4096, n_heads=32, n_kv_heads=8, head_dim=128,
                 ffn_intermediate=14336, n_experts=8, top_k=2,
                 n_layers=32, n_decode=64, cache_len=512):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_layers = n_layers
        self.n_decode = n_decode
        self.cache_len = cache_len

        self.prefill_attention = MixtralPrefillAttention(
            hidden, n_heads, n_kv_heads, head_dim)
        self.prefill_moe = MixtralPrefillMoE(
            hidden, ffn_intermediate, n_experts, top_k)
        self.decode_attention = MixtralDecodeAttention(
            hidden, n_heads, n_kv_heads, head_dim, cache_len)
        self.decode_moe = MixtralDecodeMoE(
            hidden, ffn_intermediate, n_experts, top_k)

    def forward(self, x):
        """
        Args:
            x: (batch, prompt_len, hidden)
        Returns:
            (batch, 1, hidden) — last decoded token embedding
        """
        bsz = x.shape[0]

        # === Prefill phase ===
        for _ in range(self.n_layers):
            x = self.prefill_attention(x)
            x = self.prefill_moe(x)

        # Initialise KV cache (n_kv_heads for GQA)
        k_cache = torch.zeros(bsz, self.n_kv_heads, self.cache_len, self.head_dim)
        v_cache = torch.zeros(bsz, self.n_kv_heads, self.cache_len, self.head_dim)

        # === Decode phase ===
        token_x = x[:, -1:, :]

        for _ in range(self.n_decode):
            for _ in range(self.n_layers):
                token_x = self.decode_attention(token_x, k_cache, v_cache)
                token_x = self.decode_moe(token_x)

        return token_x
