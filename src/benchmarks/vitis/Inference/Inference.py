from csv import QUOTE_ALL
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class MultiHeadSelfAttentionKV(nn.Module):
    """
    Multi-Head Self-Attention with KV caching support.
    
    This class provides separate methods for prefill (processing entire sequence)
    and decode (processing single tokens with cached K/V) phases.
    """
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttentionKV, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        use_kv_cache: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim) - new tokens
            k_cache: Cached keys of shape (batch_size, cached_len, num_heads, head_dim)
            v_cache: Cached values of shape (batch_size, cached_len, num_heads, head_dim)
            use_kv_cache: Whether to use cached K/V or not
            
        Returns:
            Tuple of (output, new_k_cache, new_v_cache) where:
            - output: Attention output of shape (batch_size, seq_len, embed_dim)
            - new_k_cache: Updated keys cache of shape (batch_size, cached_len + 1, num_heads, head_dim)
            - new_v_cache: Updated values cache of shape (batch_size, cached_len + 1, num_heads, head_dim)
        """
        batch_size, seq_len, embed_dim = x.size()
        
        # Project new token to Q, K, V
        Q = self.query(x)  # (batch_size, seq_len, embed_dim)
        K_new = self.key(x)    # (batch_size, seq_len, embed_dim)
        V_new = self.value(x)  # (batch_size, seq_len, embed_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        K_new = K_new.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        V_new = V_new.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        
        # Concatenate cached K/V with new token's K/V
        # k_cache: (batch, cached_len, num_heads, head_dim) -> transpose to (batch, num_heads, cached_len, head_dim)
        # NOTE: this is a workaround to avoid bufferization dialect. I instead just increased the sequence dimension of K_cache and V_cache by 1.
        if use_kv_cache:
            k_cache_transposed = k_cache.transpose(1, 2)  # (batch, num_heads, cached_len, head_dim)
            v_cache_transposed = v_cache.transpose(1, 2)  # (batch, num_heads, cached_len, head_dim)
            K = k_cache_transposed
            V = v_cache_transposed
        else:
            K = K_new
            V = V_new
        
        # Compute attention scores: Q @ K^T
        # Q: (batch, num_heads, seq_len, head_dim)
        # K: (batch, num_heads, cached_len + 1, head_dim)
        # scores: (batch, num_heads, seq_len, cached_len + 1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)  # (batch, num_heads, seq_len, head_dim)
        
        # Reshape back: (batch, num_heads, 1, head_dim) -> (batch, 1, embed_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out(context)
        
        # Update cache: concatenate new K/V to cached K/V
        # Transpose back to (batch, cached_len + 1, num_heads, head_dim) for storage
        new_k_cache = K.transpose(1, 2)  # (batch, cached_len + 1, num_heads, head_dim)
        new_v_cache = V.transpose(1, 2)  # (batch, cached_len + 1, num_heads, head_dim)
        
        return output, new_k_cache, new_v_cache

class TransformerLayerKV(nn.Module):
    """
    A single transformer layer that combines MultiHeadSelfAttentionKV and FeedForward.
    This custom layer is needed because nn.Sequential can't handle multiple inputs/outputs.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerLayerKV, self).__init__()
        self.attention = MultiHeadSelfAttentionKV(embed_dim, num_heads)
        self.feedforward = FeedForward(embed_dim, ff_dim, dropout)
    
    def forward(self, x, k_cache, v_cache, use_kv_cache=True):
        # Attention with KV cache
        x, new_k_cache, new_v_cache = self.attention(x, k_cache, v_cache, use_kv_cache=use_kv_cache)
        # Feedforward (only takes x)
        x = self.feedforward(x)
        return x, new_k_cache, new_v_cache

class TransformerKV(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, num_tokens, ff_dim, dropout=0.1):
        super(TransformerKV, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_tokens = num_tokens
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.embed = nn.Embedding(num_tokens, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList([
            TransformerLayerKV(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.out = nn.Linear(embed_dim, num_tokens)

    def forward(self, x, k_cache, v_cache, use_kv_cache=True): # assume x is already embedded
        #x = self.embed(x)
        #x = self.norm(x)
        for layer in self.layers:
            x, new_k_cache, new_v_cache = layer(x, k_cache, v_cache, use_kv_cache=use_kv_cache)
            k_cache = new_k_cache
            v_cache = new_v_cache
        #x = self.out(x)
        return x, k_cache, v_cache

class Inference(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, num_tokens, ff_dim, dropout=0.1):
        super(Inference, self).__init__()
        self.transformer_kv = TransformerKV(embed_dim, num_heads, num_layers, num_tokens, ff_dim, dropout)

    def forward(self, x, num_decode_steps=2): # runs 1 prefill and 2 decodes
        x, k, v = self.transformer_kv(x, [], [], use_kv_cache=False)
        print(f"x shape: {x.shape}")
        print(f"k shape: {k.shape}")
        print(f"v shape: {v.shape}")
        x = x[:, -1:]
        for _ in range(num_decode_steps):
            x, k, v = self.transformer_kv(x, k, v, use_kv_cache=True)
        return x