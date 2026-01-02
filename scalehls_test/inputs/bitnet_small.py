import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torch_mlir
import re

from typing import Optional, Tuple, Callable, Any


class MinimalAttention(nn.Module):
    """Minimal model to reproduce dominance error: matmul + softmax pattern"""
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        # Simple linear layers for query and key
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.key_proj = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x: Tensor) -> Tensor:
        # x shape: [batch, seq_len, dim]
        batch, seq_len, _ = x.shape
        
        # Project to query and key
        query = self.query_proj(x)  # [batch, seq_len, dim]
        key = self.key_proj(x)      # [batch, seq_len, dim]
        
        # Matmul: query @ key^T
        # This creates a tensor [batch, seq_len, seq_len]
        attn = torch.matmul(query, key.transpose(-2, -1))  # [batch, seq_len, seq_len]
        
        # Softmax over the last dimension (reduction dimension)
        # This is the pattern that creates the linalg.generic with reduction
        attn = F.softmax(attn, dim=-1)  # [batch, seq_len, seq_len]
        
        return attn


# Create minimal model
model = MinimalAttention(dim=64)
model.train(False)

# Create input: [batch=1, seq_len=32, dim=64]
input_tensor = torch.randn(1, 32, 64)

# Compile with torch_mlir
module = torch_mlir.compile(model, input_tensor, 
                           output_type=torch_mlir.OutputType.LINALG_ON_TENSORS)
print(module)
# mlir_text = str(module)
# mlir_text = rewrite_maximumf_to_cmp_select_text(mlir_text)

# print(mlir_text)
