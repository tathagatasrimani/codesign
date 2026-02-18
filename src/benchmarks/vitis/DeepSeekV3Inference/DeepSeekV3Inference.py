import os
import sys
import torch
import torch.nn as nn

# Add parent vitis directory so sibling sub-block packages are importable.
_here = os.path.dirname(os.path.abspath(__file__))
_vitis = os.path.dirname(_here)
if _vitis not in sys.path:
    sys.path.insert(0, _vitis)

from DeepSeekPrefillAttention.DeepSeekPrefillAttention import DeepSeekPrefillAttention
from DeepSeekPrefillDenseFFN.DeepSeekPrefillDenseFFN import DeepSeekPrefillDenseFFN
from DeepSeekPrefillMoE.DeepSeekPrefillMoE import DeepSeekPrefillMoE
from DeepSeekDecodeAttention.DeepSeekDecodeAttention import DeepSeekDecodeAttention
from DeepSeekDecodeDenseFFN.DeepSeekDecodeDenseFFN import DeepSeekDecodeDenseFFN
from DeepSeekDecodeMoE.DeepSeekDecodeMoE import DeepSeekDecodeMoE


class DeepSeekV3Inference(nn.Module):
    """DeepSeek-V3 full inference: prefill phase + autoregressive decode.

    DeepSeek-V3 is a 671B-parameter (37B active) Mixture-of-Experts model.
    The transformer consists of two structurally distinct layer types:

      Layers 1-3   (first_k_dense_replace=3):
        MLA attention + DENSE SwiGLU FFN (intermediate=18432)

      Layers 4-61  (n_layers - n_dense = 58):
        MLA attention + SPARSE MoE FFN
          256 routed + 1 shared expert, top-8 active (intermediate=2048)

    Real dimensions:
      hidden=7168, n_heads=128, head_dim=128,
      q_lora_rank=1536, kv_lora_rank=512,
      dense_intermediate=18432,
      moe_intermediate=2048, n_routed_experts=256, top_k=8.

    Sub-block attributes matched to block_types in the system YAML:
      prefill_attention -> DeepSeekPrefillAttention  (MLA prefill, all layers)
      prefill_dense_ffn -> DeepSeekPrefillDenseFFN   (dense FFN, layers 1-3)
      prefill_moe       -> DeepSeekPrefillMoE        (MoE FFN, layers 4-61)
      decode_attention  -> DeepSeekDecodeAttention   (MLA decode, all layers)
      decode_dense_ffn  -> DeepSeekDecodeDenseFFN    (dense FFN decode, layers 1-3)
      decode_moe        -> DeepSeekDecodeMoE         (MoE FFN decode, layers 4-61)
    """

    def __init__(self, hidden=7168, n_heads=128, head_dim=128,
                 q_lora_rank=1536, kv_lora_rank=512,
                 dense_intermediate=18432,
                 moe_intermediate=2048, n_routed_experts=256, top_k=8,
                 n_layers=61, n_dense=3, n_decode=64, cache_len=512):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_layers = n_layers
        self.n_dense = n_dense
        self.n_decode = n_decode
        self.cache_len = cache_len

        # Attention is MLA for all layers
        self.prefill_attention = DeepSeekPrefillAttention(
            hidden, n_heads, head_dim, q_lora_rank, kv_lora_rank)
        self.decode_attention = DeepSeekDecodeAttention(
            hidden, n_heads, head_dim, q_lora_rank, kv_lora_rank, cache_len)

        # Dense FFN (layers 1–3) vs MoE FFN (layers 4–61)
        self.prefill_dense_ffn = DeepSeekPrefillDenseFFN(hidden, dense_intermediate)
        self.prefill_moe = DeepSeekPrefillMoE(
            hidden, moe_intermediate, n_routed_experts, top_k)
        self.decode_dense_ffn = DeepSeekDecodeDenseFFN(hidden, dense_intermediate)
        self.decode_moe = DeepSeekDecodeMoE(
            hidden, moe_intermediate, n_routed_experts, top_k)

    def forward(self, x):
        """
        Args:
            x: (batch, prompt_len, hidden)
        Returns:
            (batch, 1, hidden) — last decoded token embedding
        """
        bsz = x.shape[0]

        # === Prefill phase ===
        # First n_dense layers: MLA + dense FFN
        for _ in range(self.n_dense):
            x = self.prefill_attention(x)
            x = self.prefill_dense_ffn(x)

        # Remaining layers: MLA + MoE FFN
        for _ in range(self.n_layers - self.n_dense):
            x = self.prefill_attention(x)
            x = self.prefill_moe(x)

        # Initialise K/V cache
        k_cache = torch.zeros(bsz, self.n_heads, self.cache_len, self.head_dim)
        v_cache = torch.zeros(bsz, self.n_heads, self.cache_len, self.head_dim)

        # === Decode phase ===
        token_x = x[:, -1:, :]

        for _ in range(self.n_decode):
            for _ in range(self.n_dense):
                token_x = self.decode_attention(token_x, k_cache, v_cache)
                token_x = self.decode_dense_ffn(token_x)
            for _ in range(self.n_layers - self.n_dense):
                token_x = self.decode_attention(token_x, k_cache, v_cache)
                token_x = self.decode_moe(token_x)

        return token_x
