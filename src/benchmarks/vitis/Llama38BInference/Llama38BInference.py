import os
import sys
import torch
import torch.nn as nn

# Add parent vitis directory so sibling sub-block packages are importable.
_here = os.path.dirname(os.path.abspath(__file__))
_vitis = os.path.dirname(_here)
if _vitis not in sys.path:
    sys.path.insert(0, _vitis)

from Llama38BPrefillAttention.Llama38BPrefillAttention import Llama38BPrefillAttention
from LlamaFeedForward.LlamaFeedForward import LlamaFeedForward
from Llama38BDecodeAttention.Llama38BDecodeAttention import Llama38BDecodeAttention
from LlamaDecodeFeedForward.LlamaDecodeFeedForward import LlamaDecodeFeedForward


class Llama38BInference(nn.Module):
    """LLaMA 3 8B full inference: prefill phase followed by autoregressive decode.

    Uses grouped-query attention (GQA) with 32 Q-heads and 8 KV-heads, and
    a dense SwiGLU feedforward with intermediate dim 14336.

    This module defines the end-to-end inference dataflow by composing
    sub-block modules.  System codesign traces a forward pass through this
    module and records which sub-blocks are called to build the system DFG.

    Sub-block attributes matched to block_types in the system YAML:
        prefill_attention   -> Llama38BPrefillAttention  (GQA, full-seq)
        prefill_feedforward -> LlamaFeedForward           (SwiGLU, full-seq)
        decode_attention    -> Llama38BDecodeAttention    (GQA, single-token + KV cache)
        decode_feedforward  -> LlamaDecodeFeedForward     (SwiGLU, single-token)

    Real LLaMA 3 8B dimensions:
        hidden=4096, n_heads=32, n_kv_heads=8, head_dim=128,
        ffn_dim=14336, n_layers=32.
    """

    def __init__(self, hidden=4096, n_heads=32, n_kv_heads=8, head_dim=128,
                 ffn_dim=14336, n_layers=32, n_decode=64, cache_len=64):
        super().__init__()
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_layers = n_layers
        self.n_decode = n_decode
        self.cache_len = cache_len

        self.prefill_attention   = Llama38BPrefillAttention(hidden, n_heads, n_kv_heads, head_dim)
        self.prefill_feedforward = LlamaFeedForward(hidden, ffn_dim)
        self.decode_attention    = Llama38BDecodeAttention(hidden, n_heads, n_kv_heads, head_dim, cache_len)
        self.decode_feedforward  = LlamaDecodeFeedForward(hidden, ffn_dim)

    def forward(self, x):
        """Run full inference: prefill then autoregressive decode.

        Args:
            x: prompt tensor (batch, prompt_len, hidden)
        Returns:
            Last decoded token embedding (batch, 1, hidden)
        """
        bsz = x.shape[0]

        # === Prefill phase ===
        for _ in range(self.n_layers):
            x = self.prefill_attention(x)
            x = self.prefill_feedforward(x)
        
        # KV cache uses n_kv_heads (GQA), not n_heads
        k_cache = torch.zeros(bsz, self.n_kv_heads, self.cache_len, self.head_dim)
        v_cache = torch.zeros(bsz, self.n_kv_heads, self.cache_len, self.head_dim)

        # === Decode phase ===
        token_x = x[:, -1:, :]   # seed: last prefill token (batch, 1, hidden)

        for _ in range(self.n_decode):
            for _ in range(self.n_layers):
                token_x = self.decode_attention(token_x, k_cache, v_cache)
                token_x = self.decode_feedforward(token_x)

        return token_x
