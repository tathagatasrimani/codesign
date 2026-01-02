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

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        fixed_batch: int | None = None,
        fixed_seq: int | None = None,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.fixed_batch = fixed_batch
        self.fixed_seq = fixed_seq

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn = ScaledDotProductAttention(self.head_dim)

    def _shape(self, x: Tensor) -> Tensor:
        batch = self.fixed_batch if self.fixed_batch is not None else x.size(0)
        seq_len = self.fixed_seq if self.fixed_seq is not None else x.size(1)
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _unshape(self, x: Tensor) -> Tensor:
        batch = self.fixed_batch if self.fixed_batch is not None else x.size(0)
        seq_len = self.fixed_seq if self.fixed_seq is not None else x.size(2)
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
    Replica of SimpleAttention but with fixed static dimensions to
    avoid dynamic shapes in the emitted MLIR.
    """

    def __init__(
        self,
        *,
        batch_size: int = 1,
        seq_len: int = 8,
        embed_dim: int = 32,
        num_heads: int = 2,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.attn = SimpleAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            fixed_batch=batch_size,
            fixed_seq=seq_len,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape(self.batch_size, self.seq_len, self.embed_dim)
        return self.attn(x)


def tiny_attention_model(**kwargs) -> TinyAttention:
    return TinyAttention(**kwargs)


if __name__ == "__main__":

    tiny_model = tiny_attention_model(
        batch_size=1,
        seq_len=8,
        embed_dim=32,
        num_heads=2,
    )
    tiny_model.eval()
    tiny_sample = torch.randn(1, 8, tiny_model.attn.embed_dim)
    tiny_module = torch_mlir.compile(
        tiny_model,
        tiny_sample,
        output_type=torch_mlir.OutputType.LINALG_ON_TENSORS,
    )
    print(tiny_module)