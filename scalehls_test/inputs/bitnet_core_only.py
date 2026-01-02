import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torch_mlir
import re

from typing import Optional, Tuple, Callable, Any




class OutputHead(nn.Module):
    """
    Simple output head for language modeling.
    Replaces the zeta.OutputHead to avoid dependency issues.
    """
    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, vocab_size)
    
    def forward(self, x):
        x = self.norm(x)
        return self.linear(x)



class SimpleRMSNorm(nn.Module):
    """
    SimpleRMSNorm

    Args:
        dim (int): dimension of the embedding
rear
    Usage:
    We can use SimpleRMSNorm as a layer in a neural network as follows:
        >>> x = torch.randn(1, 10, 512)
        >>> simple_rms_norm = SimpleRMSNorm(dim=512)
        >>> simple_rms_norm(x).shape
        torch.Size([1, 10, 512])

    """

    def __init__(self, dim):
        super().__init__()
        self.scale = dim**-0.5

    def forward(self, x):
        """Forward method of SimpleRMSNorm using RMS computation"""
        # Compute RMS along last dimension
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
        # Normalize and scale
        return x / rms * self.scale

def activation_quant(x: Tensor):
    """Per token quantization to 8bits. No grouping is needed for quantization

    Args:
        x (Tensor): _description_

    Returns:
        _type_: _description_
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y

def weight_quant(w: Tensor):
    """Binary weight quantization following BitNet paper.
    w → sign(w) * E[|w|]
    """
    scale = w.abs().mean()
    return w.sign() * scale

class BitLinear(nn.Linear):
    """
    Custom linear layer with bit quantization following BitNet paper.
    Key features:
    1. RMSNorm for input normalization
    2. Binary weight quantization
    3. 8-bit activation quantization
    4. Straight-through estimator (STE) for gradients

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        bias (bool): Whether to use bias
    """
    def __init__(self, dim_in, dim_out, bias): 
        super().__init__(dim_in, dim_out, bias)
        self.rms = SimpleRMSNorm(dim_in)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the BitLinear layer implementing BitNet operations.
        1. RMSNorm the input
        2. Quantize input to 8-bit
        3. Binarize weights
        4. Use STE for both quantizations
        """
        w = self.weight
        x_norm = self.rms(x)

        # Apply quantization with straight-through estimator
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        
        # Linear transformation with quantized values
        return F.linear(x_quant, w_quant, self.bias)

def default(val, d):
    return val if val is not None else d


def init_zero_(tensor):
    nn.init.constant_(tensor, 0.0)


# [GLU]
class GLU(nn.Module):
    """
    Gated Linear Unit (GLU) module.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension.
        activation (Callable): Activation function to be applied to the gate.
        mult_bias (bool, optional): Whether to multiply the bias term. Defaults to False.
        linear (Callable, optional): Linear function to be used for projection. Defaults to False.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        activation: Callable,
        mult_bias: bool = False,
        linear: Callable = False,
        *args,
        **kwargs
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.activation = activation
        self.mult_bias = mult_bias

        if linear:
            self.proj = linear(dim_in, dim_out * 2)
        else:
            self.proj = BitLinear(dim_in, dim_out * 2, *args, **kwargs)

        self.mult_bias = nn.Parameter(torch.ones(dim_out)) if mult_bias else 1.0

    def forward(self, x: Tensor):
        proj_out = self.proj(x)
        chunks = proj_out.shape[-1] // 2
        x, gate = proj_out[..., :chunks], proj_out[..., chunks:]
        return x * self.activation(gate) * self.mult_bias


# [FEATURE] Add type hints to the forward method
class BitFeedForward(nn.Module):
    """
    BitFeedForward module performs feed-forward operations on the input tensor.

    Args:
        dim (int): The input dimension.
        dim_out (int, optional): The output dimension. If not provided, it is set to the input dimension.
        mult (int, optional): The multiplier for the inner dimension. Default is 4.
        glu (bool, optional): Whether to use Gated Linear Unit (GLU) activation. Default is False.
        glu_mult_bias (bool, optional): Whether to apply bias to the GLU activation. Default is False.
        swish (bool, optional): Whether to use Swish activation. Default is False.
        relu_squared (bool, optional): Whether to use squared ReLU activation. Default is False.
        post_act_ln (bool, optional): Whether to apply Layer Normalization after activation. Default is False.
        dropout (float, optional): The dropout probability. Default is 0.0.
        no_bias (bool, optional): Whether to exclude bias in linear layers. Default is False.
        zero_init_output (bool, optional): Whether to initialize the last linear layer to 0. Default is False.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        glu: bool = False,
        glu_mult_bias: bool = False,
        swish: bool = False,
        post_act_ln: bool = False,
        dropout: float = 0.0,
        no_bias: bool = False,
        zero_init_output: bool = False,
        *args,
        **kwargs
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        if swish:
            activation = nn.SiLU()
        else:
            activation = nn.GELU()

        if glu:
            project_in = GLU(dim, inner_dim, activation, mult_bias=glu_mult_bias)
        else:
            project_in = nn.Sequential(
                BitLinear(dim, inner_dim, bias=not no_bias, *args, **kwargs), activation
            )
        if post_act_ln:
            self.ff = nn.Sequential(
                project_in,
                nn.LayerNorm(inner_dim),
                nn.Dropout(dropout),
                BitLinear(inner_dim, dim_out, bias=not no_bias, *args, **kwargs),
            )
        else:
            self.ff = nn.Sequential(
                project_in,
                nn.Dropout(dropout),
                BitLinear(inner_dim, dim_out, bias=not no_bias, *args, **kwargs),
            )

        # init last linear layer to 0
        if zero_init_output:
            init_zero_(self.ff[-1])

    def forward(self, x):
        """
        Forward pass of the BitFeedForward module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.ff(x)

def rearrange_qkv(qkv: Tensor, num_heads: int) -> Tensor:
    """Split heads for query/key/value tensors."""
    batch_size, seq_len = qkv.size(0), qkv.size(1)
    head_dim = qkv.size(-1) // num_heads
    qkv = qkv.view(batch_size, seq_len, num_heads, head_dim)
    return qkv.permute(0, 2, 1, 3)  # [batch, heads, seq, head_dim]

def rearrange_output(out: Tensor) -> Tensor:
    """Combine heads for output tensor."""
    batch_size = out.size(0)
    seq_len = out.size(2)
    hidden_dim = out.size(1) * out.size(3)
    out = out.permute(0, 2, 1, 3)  # [batch, seq, heads, head_dim]
    return out.reshape(batch_size, seq_len, hidden_dim)

def gqa_attention(query: Tensor, key: Tensor, value: Tensor, query_heads: int, kv_heads: int) -> Tensor:
    """Compute grouped query attention with static shapes."""
    # Get dimensions
    batch_size, seq_len, _ = query.shape
    head_dim = query.size(-1) // query_heads
    
    # Reshape inputs
    query = query.view(batch_size, seq_len, query_heads, head_dim)
    key = key.view(batch_size, seq_len, kv_heads, head_dim)
    value = value.view(batch_size, seq_len, kv_heads, head_dim)
    
    # Handle different head counts by expanding k/v
    if query_heads > kv_heads:
        # Expand k/v along head dimension to match query heads
        key = key.unsqueeze(3).expand(batch_size, seq_len, kv_heads, query_heads//kv_heads, head_dim).reshape(batch_size, seq_len, query_heads, head_dim)
        value = value.unsqueeze(3).expand(batch_size, seq_len, kv_heads, query_heads//kv_heads, head_dim).reshape(batch_size, seq_len, query_heads, head_dim)
    
    # Transpose to [batch, heads, seq, head_dim]
    query = query.permute(0, 2, 1, 3)  # [b, h, s, d]
    key = key.permute(0, 2, 1, 3)  # [b, h, s, d]
    value = value.permute(0, 2, 1, 3)  # [b, h, s, d]
    
    # Compute attention with matching head dimensions
    scale = head_dim ** -0.5
    attn = torch.matmul(query * scale, key.transpose(-2, -1))  # [b, h, s, s]
    attn = F.softmax(attn, dim=-1)
    
    # Apply attention to values and reshape
    out = torch.matmul(attn, value)  # [b, h, s, d]
    return out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

def scaled_dot_product_gqa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout: float = 0.0,
    scale: Optional[float] = None,
    mask: Optional[Tensor] = None,
    is_causal: Optional[bool] = None,
    need_weights: bool = False,
    average_attn_weights: bool = False,
    force_grouped: bool = False,
) -> Tensor:  # Changed return type to single Tensor
    """Scaled dot product attention with support for grouped queries.

    Einstein notation:
    - b: batch size
    - n / s: sequence length
    - h: number of heads
    - g: number of groups
    - d: dimension of query/key/value

    Args:
        query: Query tensor of shape (b, n, h, d)
        key: Key tensor of shape (b, s, h, d)
        value: Value tensor of shape (b, s, h, d)
        dropout: Dropout probability (default: 0.0)
        scale: Scale factor for query (default: d_query ** 0.5)
        mask: Mask tensor of shape (b, n, s) or (b, s). If 'ndim == 2', the mask is
            applied to all 'n' rows of the attention matrix. (default: None)
        force_grouped: If True, apply grouped-query attention even if the number of
            heads is equal for query, key, and value. (default: False)

    Returns:
        Attention output with shape (b, n, h, d)
    """
    if (mask is not None) and (is_causal is not None):
        raise ValueError(
            "Only one of 'mask' and 'is_causal' should be provided, but got both."
        )
    elif not query.ndim == key.ndim == value.ndim == 4:
        raise ValueError(
            f"Expected query, key, and value to be 4-dimensional, but got shapes "
            f"{query.shape}, {key.shape}, and {value.shape}."
        )

    # Move sequence length dimension to axis 2.
    # This makes the attention operations below *much* faster.
    query = _rearrange_to_heads(query, "b n h d -> b h n d")
    key = _rearrange_to_heads(key, "b s h d -> b h s d")
    value = _rearrange_to_heads(value, "b s h d -> b h s d")

    bq, hq, nq, dq = query.shape
    bk, hk, nk, dk = key.shape
    bv, hv, nv, dv = value.shape
    if not (bq == bk == bv and dq == dk == dv):
        raise ValueError(
            "Expected query, key, and value to have the same batch size (dim=0) and "
            f"embedding dimension (dim=3), but got query: {query.shape}, "
            f"key: {key.shape}, and value: {value.shape}."
        )
    elif (hk != hv) or (nk != nv):
        raise ValueError(
            "Expected key and value to have the same size in dimensions 1 and 2, but "
            f"got key: {key.shape} and value: {value.shape}."
        )
    elif hq % hk != 0:
        raise ValueError(
            "Expected query heads to be a multiple of key/value heads, but got "
            f"query: {query.shape} and key/value: {key.shape}."
        )

    if scale is None:
        scale = query.size(-1) ** 0.5
    query = query / scale

    num_head_groups = hq // hk
    if num_head_groups > 1 or force_grouped:
        # Separate the query heads into 'num_head_groups' chunks, and fold the group
        # dimension into the batch dimension.  This allows us to compute the attention
        # for each head in parallel, then sum over all of the groups at the end.
        query = _rearrange_to_heads_with_g(query, "b (h g) n d -> b g h n d", g=num_head_groups)
        similarity = _einsum_similarity(query, key, "b g h n d, b h s d -> b h n s")
    else:
        # If the number of query/key heads is equal, we can skip grouping the queries,
        # and just use the standard sdot product attention.
        similarity = _einsum_similarity(query, key, "b h n d, b h s d -> b h n s")

    if is_causal is True:
        # Mask out the upper triangular portion of the attention matrix. This prevents
        # the model from attending to tokens in the future.
        mask = torch.ones(
            (bq, nq, nk),
            device=query.device,
            dtype=torch.bool,
        ).tril_()

    if mask is not None:
        # Expand mask to match the shape of the attention matrix.
        # If mask is 2D, assume that it is applied to the key/value sequence dimension.
        # Else if mask is 3D, assume that it is applied to the query/key/value sequence
        # dimension for all attention heads.
        #
        # Users could also provide a 4D mask, which is applied to the query/key/value
        # sequence dimension for each attention head (though I don't have a particular
        # use case in mind for that).
        if mask.ndim == 2:
            mask = _rearrange_to_heads(mask, "b s -> b () () s")
        elif mask.ndim == 3:
            mask = _rearrange_to_heads(mask, "b n s -> b () n s")
        # Mask similarity values by setting them to negative infinity.  This guarantees
        # that they will not contribute to the softmax computation below.
        similarity = similarity.masked_fill(~mask, -1e9)


    attention = F.softmax(similarity / scale, dim=-1)
    if dropout > 0.0:
        attention = F.dropout(attention, p=dropout)

    # Apply attention matrix to the value Tensor.
    out = _einsum_similarity(attention, value, "b h n s, b h s d -> b h n d")
    # Move head dimension back to axis 2 and return only output tensor
    return _rearrange_to_heads(out, "b h n d -> b n h d")

class BitMGQA(nn.Module):
    """Multi-head grouped query attention (GQA) layer.

    Reference:
        "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
        https://arxiv.org/pdf/2305.13245v1.pdf

    GQA is a variant of multihead attention (MHA) that uses fewer write heads
    (key / value) than query heads.  GQA can be viewed as a generalization of
    multi-query attention (MQA), which uses a single write head. GQA and MQA give
    significant speedups over standard MHA in decoder layers, with minimal loss in
    accuracy. In the paper, GQA is shown to be more accurate than MQA, while still
    having a significant speedup over MHA.

    NOTE: The original authors only benchmark GQA by adapting the T5 (XL or XXL) model
    from MHA to GQA.  As a result, they do not mention parameter initialization or
    layer normalization strategies.  I follow the best practices laid out in the
    MAGNETO paper, which improves Transformer performance through better parameter
    initialization and layer norm placement.  See:
        https://arxiv.org/pdf/2210.06423.pdf, Fig. 2
    """

    def __init__(
        self,
        embed_dim: int,
        query_heads: int = 8,
        kv_heads: int = 4,
        dropout: float = 0.1,
        bias: bool = True,
        layer_norm: bool = True,
        layer_norm_eps: float = 1e-5,
        gamma_init: float = 1.0,
        linear_groups: int = 1
    ):
        super().__init__()
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.gamma_init = gamma_init

        if self.query_heads % self.kv_heads != 0:
            raise ValueError(
                f"query_heads ({query_heads}) must be divisible by "
                f"kv_heads ({kv_heads})"
            )
        elif (embed_dim % self.query_heads != 0) or (embed_dim % self.kv_heads != 0):
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"query_heads ({query_heads}) and kv_heads ({kv_heads})"
            )

        head_dim = embed_dim // query_heads
        if not head_dim % 8 == 0:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {head_dim}) must be divisible by 8"
            )
        if not head_dim <= 128:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {head_dim}) must be <= 128"
            )

        # Query projection layer is the same as in vanilla MHA.
        self.q_proj = BitLinear(
            embed_dim,
            embed_dim,
            bias=bias
        )
        # Key/value projection layers have a smaller output dimension, so that
        # the we have fewer key/value attention heads after reshaping.
        self.kv_embed_dim = embed_dim // query_heads * kv_heads
        self.k_proj = BitLinear(
            embed_dim,
            self.kv_embed_dim,
            bias=bias
        )
        self.v_proj = BitLinear(
            embed_dim,
            self.kv_embed_dim,
            bias=bias
        )
        self.norm: Optional[nn.LayerNorm] = None
        if layer_norm:
            self.norm = nn.LayerNorm(
                embed_dim,  # Use full embedding dim for normalization
                eps=layer_norm_eps
            )
        # Grouped attention output will have the same embedding dimension as the
        # key/value Tensors.  So the output projection layer needs to accept the
        # same dimension (kv_embed_dim).
        self.out_proj = BitLinear(
            embed_dim,  # Use full embedding dim for output projection
            embed_dim,
            bias=bias
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
        nn.init.xavier_normal_(self.k_proj.weight)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0)

        # NOTE: We follow the initialization strategy from MAGNETO.  See:
        # https://arxiv.org/pdf/2210.06423.pdf, Fig. 2
        # Gain (self.gamma_init) should be provided as a keyword argument when
        # initializing the larger Transformer model, since it requires knowledge
        # of the number of encoder/decoder layers in the model.

        nn.init.xavier_normal_(self.v_proj.weight, gain=self.gamma_init)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0)
        nn.init.xavier_normal_(self.out_proj.weight, gain=self.gamma_init)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        need_weights: bool = False,
        is_causal: bool = False,
        average_attn_weights: bool = False,
    ) -> Tensor:
        # Project and compute attention
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        x = gqa_attention(q, k, v, self.query_heads, self.kv_heads)
        
        # Apply layer norm and output projection
        if self.layer_norm:
            x = self.norm(x)
        return self.out_proj(x)

class Transformer(nn.Module):
    """
    Transformer module that applies multi-head attention and feed-forward layers.

    Args:
        dim (int): The dimension of the input and output tensors.
        heads (int): The number of attention heads.
        depth (int): The number of transformer layers.
        ff_mult (int, optional): The multiplier for the hidden dimension in the feed-forward layers.
            Defaults to 2.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        layers (nn.ModuleList): List of multi-head attention layers.
        ffn_layers (nn.ModuleList): List of feed-forward layers.

    """

    def __init__(
        self, dim: int, heads: int, depth: int, ff_mult: int = 2, *args, **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(BitMGQA(dim, heads))

            self.ffn_layers.append(
                BitFeedForward(
                    dim,
                    dim,
                    ff_mult,
                    swish=True,
                    post_act_ln=True,
                    dropout=0.1,
                ),
            )
            
        # Norm
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        for attn, ffn in zip(self.layers, self.ffn_layers):
            skip = x
            # Get attention output only, no weights needed
            x = attn(x, x, x, is_causal=True, need_weights=False)
            #x = self.norm(x + skip)
            #x = ffn(x) + x
            return x
        return x


# [MAIN MODEL] BitNetTransformer
class BitNetTransformer(nn.Module):
    """
    BitNetTransformer is a transformer-based model for BitNet.

    Args:
        dim (int): The dimension of the token embeddings.
        depth (int): The number of transformer layers.
        num_tokens (int): The number of tokens in the vocabulary.
        heads (int, optional): The number of attention heads in the transformer. Defaults to 8.
        ff_mult (int, optional): The multiplier for the feed-forward layer dimension. Defaults to 4.

    Examples:
    >>> import torch
    >>> from bitnet import BitNetTransformer
    >>> x = torch.randint(0, 20000, (1, 1024))
    >>> bitnet = BitNetTransformer(
    ...     num_tokens=20000,
    ...     dim=1024,
    ...     depth=6,
    ...     heads=8,
    ...     ff_mult=4,
    ... )
    >>> logits = bitnet(x)
    >>> print(logits)
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_tokens: int,
        heads: int = 8,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.emb = nn.Embedding(num_tokens, dim)

        self.transformer = Transformer(
            dim=dim, depth=depth, heads=heads, ff_mult=ff_mult
        )

        # self.to_logits = nn.Sequential(RMSNorm(dim), nn.Linear(dim, num_tokens))
        self.to_logits = OutputHead(
            dim,
            vocab_size=num_tokens,
        )
        
        # Norm
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.emb(x)
        # Post emb norm
        x = self.norm(x)
        
        x = self.transformer(x)
        return x


def _bitnet(num_tokens: int, dim: int, depth: int, heads: int, ff_mult: int, **kwargs: Any) -> BitNetTransformer:
    """
    Generic factory function to create a BitNetTransformer model.
    """
    model = BitNetTransformer(
        num_tokens=num_tokens,
        dim=dim,
        depth=depth,
        heads=heads,
        ff_mult=ff_mult,
        **kwargs
    )
    return model

def bitnet_base(**kwargs: Any) -> BitNetTransformer:
    """
    Creates a 'base' configuration for BitNetTransformer.
    You must provide `num_tokens` as a keyword argument.
    
    Example:
        >>> model = bitnet_base(num_tokens=20000)
    """
    if 'num_tokens' not in kwargs:
        raise ValueError("You must specify 'num_tokens' for the model configuration.")
    
    return _bitnet(
        dim=1024,
        depth=6,
        heads=8,
        ff_mult=4,
        **kwargs
    )

# Create and compile BitNet model
bitnet_model = bitnet_base(num_tokens=20000)
bitnet_model.train(False)

def rewrite_maximumf_to_cmp_select_text(mlir_text: str) -> str:
    pat = re.compile(r'(\s*)(%[\w\d_]+)\s*=\s*arith\.maximumf\s+([^,]+),\s*([^\s:]+)\s*:\s*([^\n]+)')
    def repl(m):
        i,res,a,b,t = m.groups()
        c = res + "_cmp"
        return f"{i}{c} = arith.cmpf ogt, {a}, {b} : {t}\n{i}{res}  = arith.select {c}, {a}, {b} : {t}"
    return pat.sub(repl, mlir_text)

# Compile with torch_mlir
module = torch_mlir.compile(bitnet_model, torch.randint(0, 20000, (1, 1024)), 
                           output_type=torch_mlir.OutputType.LINALG_ON_TENSORS)
print(module)
# mlir_text = str(module)
# mlir_text = rewrite_maximumf_to_cmp_select_text(mlir_text)

# print(mlir_text)

