import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torch_mlir

from typing import Optional


class SimpleRMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** -0.5

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
        return x / rms * self.scale


def activation_quant(x: Tensor) -> Tensor:
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


def weight_quant(w: Tensor) -> Tensor:
    scale = w.abs().mean()
    return w.sign() * scale


class BitLinear(nn.Linear):
    def __init__(self, dim_in: int, dim_out: int, bias: bool):
        super().__init__(dim_in, dim_out, bias)
        self.rms = SimpleRMSNorm(dim_in)

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        x_norm = self.rms(x)
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        return F.linear(x_quant, w_quant, self.bias)


def gqa_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    query_heads: int,
    kv_heads: int,
) -> Tensor:
    batch_size, seq_len, _ = query.shape
    head_dim = query.size(-1) // query_heads

    query = query.view(batch_size, seq_len, query_heads, head_dim)
    key = key.view(batch_size, seq_len, kv_heads, head_dim)
    value = value.view(batch_size, seq_len, kv_heads, head_dim)

    if query_heads > kv_heads:
        repeat = query_heads // kv_heads
        key = (
            key.unsqueeze(3)
            .expand(batch_size, seq_len, kv_heads, repeat, head_dim)
            .reshape(batch_size, seq_len, query_heads, head_dim)
        )
        value = (
            value.unsqueeze(3)
            .expand(batch_size, seq_len, kv_heads, repeat, head_dim)
            .reshape(batch_size, seq_len, query_heads, head_dim)
        )

    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)

    scale = head_dim ** -0.5
    attn = torch.matmul(query * scale, key.transpose(-2, -1))
    attn = F.softmax(attn, dim=-1)
    out = torch.matmul(attn, value)
    return out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)


class BitMGQA(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        query_heads: int = 8,
        kv_heads: int = 4,
        dropout: float = 0.0,
        bias: bool = True,
        layer_norm: bool = False,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.dropout = dropout
        self.layer_norm = layer_norm

        if self.query_heads % self.kv_heads != 0:
            raise ValueError("query_heads must be divisible by kv_heads")
        if (embed_dim % self.query_heads != 0) or (embed_dim % self.kv_heads != 0):
            raise ValueError("embed_dim must be divisible by head counts")

        head_dim = embed_dim // query_heads
        if head_dim % 8 != 0:
            raise ValueError("head_dim must be divisible by 8")
        if head_dim > 128:
            raise ValueError("head_dim must be <= 128")

        self.q_proj = BitLinear(embed_dim, embed_dim, bias=bias)
        self.kv_embed_dim = embed_dim // query_heads * kv_heads
        self.k_proj = BitLinear(embed_dim, self.kv_embed_dim, bias=bias)
        self.v_proj = BitLinear(embed_dim, self.kv_embed_dim, bias=bias)
        self.norm: Optional[nn.LayerNorm] = None
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.out_proj = BitLinear(embed_dim, embed_dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
        nn.init.xavier_normal_(self.k_proj.weight)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0)
        nn.init.xavier_normal_(self.v_proj.weight)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0)
        nn.init.xavier_normal_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        x = gqa_attention(q, k, v, self.query_heads, self.kv_heads)
        if self.layer_norm:
            x = self.norm(x)
        return self.out_proj(x)


class BitNetAttentionOnly(nn.Module):
    """
    Benchmark containing a single BitNet grouped-query attention invocation.
    """

    def __init__(
        self,
        *,
        embed_dim: int = 1024,
        seq_len: int = 1024,
        query_heads: int = 8,
        kv_heads: int = 4,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.input_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = BitMGQA(
            embed_dim=embed_dim,
            query_heads=query_heads,
            kv_heads=kv_heads,
            dropout=0.0,
            layer_norm=True,
            layer_norm_eps=1e-5,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.input_proj(x))
        return self.attn(x, x, x)


def bitnet_attention_only(**kwargs) -> BitNetAttentionOnly:
    return BitNetAttentionOnly(**kwargs)


if __name__ == "__main__":
    model = bitnet_attention_only()
    model.eval()
    sample = torch.randn(1, model.seq_len, model.embed_dim)
    module = torch_mlir.compile(
        model,
        sample,
        output_type=torch_mlir.OutputType.LINALG_ON_TENSORS,
    )
    print(module)

