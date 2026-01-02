import torch
import torch.nn as nn
import torch_mlir
from torch import Tensor


class ScaledDotProductAttention(nn.Module):
    """
    Minimal scaled dot-product attention layer.

    It expects query/key/value tensors shaped as
    (batch, heads, seq_len, head_dim).
    """

    def __init__(self, head_dim: int):
        super().__init__()
        self.scale = head_dim ** -0.5

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        attn_scores = torch.matmul(query * self.scale, key.transpose(-2, -1))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, value)


class SimpleAttention(nn.Module):
    """
    Toy multi-head self-attention block intended for MLIR export tests.
    """

    def __init__(self, embed_dim: int = 128, num_heads: int = 4):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn = ScaledDotProductAttention(self.head_dim)

    def _shape(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _unshape(self, x: Tensor) -> Tensor:
        batch, _, seq_len, _ = x.shape
        return x.transpose(1, 2).reshape(batch, seq_len, self.embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        query = self._shape(self.q_proj(x))
        key = self._shape(self.k_proj(x))
        value = self._shape(self.v_proj(x))
        attn_out = self.attn(query, key, value)
        attn_out = self._unshape(attn_out)
        return self.out_proj(attn_out)


def attention_model(**kwargs) -> SimpleAttention:
    return SimpleAttention(**kwargs)


class TinyAttention(nn.Module):
    """
    Replica of SimpleAttention but drastically reduced width and heads.
    """

    def __init__(self, embed_dim: int = 32, num_heads: int = 2):
        super().__init__()
        self.attn = SimpleAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, x: Tensor) -> Tensor:
        return self.attn(x)


def tiny_attention_model(**kwargs) -> TinyAttention:
    return TinyAttention(**kwargs)


if __name__ == "__main__":
    big_model = attention_model(embed_dim=128, num_heads=4)
    big_model.eval()
    big_sample = torch.randn(1, 32, big_model.embed_dim)
    big_module = torch_mlir.compile(
        big_model,
        big_sample,
        output_type=torch_mlir.OutputType.LINALG_ON_TENSORS,
    )
    print(big_module)

    tiny_model = tiny_attention_model(embed_dim=32, num_heads=2)
    tiny_model.eval()
    tiny_sample = torch.randn(1, 8, tiny_model.attn.embed_dim)
    tiny_module = torch_mlir.compile(
        tiny_model,
        tiny_sample,
        output_type=torch_mlir.OutputType.LINALG_ON_TENSORS,
    )
    print(tiny_module)

