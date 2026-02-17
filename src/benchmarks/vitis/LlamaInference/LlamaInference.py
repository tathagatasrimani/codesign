import torch
import torch.nn as nn

from LlamaPrefillAttention import LlamaPrefillAttention
from LlamaFeedForward import LlamaFeedForward
from LlamaDecodeAttention import LlamaDecodeAttention
from LlamaDecodeFeedForward import LlamaDecodeFeedForward


class LlamaInference(nn.Module):
    """Full LLaMA inference: prefill phase followed by autoregressive decode.

    This module defines the end-to-end inference dataflow by composing
    sub-block modules.  System codesign traces a forward pass through
    this module and records which sub-blocks are called (and in what
    order) to automatically build the system-level DFG.

    Sub-block attributes correspond to block_types in the system YAML:
        prefill_attention   -> LlamaPrefillAttention  (full-seq attention)
        prefill_feedforward -> LlamaFeedForward       (full-seq SwiGLU FF)
        decode_attention    -> LlamaDecodeAttention    (single-token + KV cache)
        decode_feedforward  -> LlamaDecodeFeedForward  (single-token SwiGLU FF)
    """

    def __init__(self, dim=256, n_heads=4, ffn_dim=1024,
                 n_layers=32, n_decode=64, cache_len=64):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.n_layers = n_layers
        self.n_decode = n_decode
        self.cache_len = cache_len

        # Sub-blocks — attribute names are matched to block_types via module_attr
        self.prefill_attention = LlamaPrefillAttention(dim, n_heads)
        self.prefill_feedforward = LlamaFeedForward(dim, ffn_dim)
        self.decode_attention = LlamaDecodeAttention(dim, n_heads, cache_len)
        self.decode_feedforward = LlamaDecodeFeedForward(dim, ffn_dim)

    def forward(self, x):
        """Run full inference: prefill then decode.

        Args:
            x: prompt tensor (batch, prompt_len, dim)

        Returns:
            Last decoded token embedding (batch, 1, dim)
        """
        bsz = x.shape[0]

        # === Prefill phase ===
        # Process entire prompt through all transformer layers.
        for layer in range(self.n_layers):
            x = self.prefill_attention(x)
            x = self.prefill_feedforward(x)

        # Initialize KV cache from prefill output
        # (In practice, prefill populates the cache; here we use zeros
        # since we only need the call graph structure, not exact values.)
        k_cache = torch.zeros(bsz, self.n_heads, self.cache_len, self.head_dim)
        v_cache = torch.zeros(bsz, self.n_heads, self.cache_len, self.head_dim)

        # === Decode phase ===
        # Generate tokens one at a time, each attending to the KV cache.
        token_x = x[:, -1:, :]  # seed: last prefill token (batch, 1, dim)

        for token in range(self.n_decode):
            for layer in range(self.n_layers):
                token_x = self.decode_attention(token_x, k_cache, v_cache)
                token_x = self.decode_feedforward(token_x)

        return token_x
