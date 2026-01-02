import torch
from torch import nn, Tensor
import torch_mlir


class OutputHead(nn.Module):
    """
    Simple output head for language modeling.
    Re-declared here so the minimal model is self-contained.
    """

    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        return self.linear(x)


class BitNetHeadOnly(nn.Module):
    """
    Minimal BitNet variant that keeps only embedding, LayerNorm, and output head.
    Useful for isolating ScaleHLS / MLIR issues without the transformer stack.
    """

    def __init__(
        self,
        dim: int,
        num_tokens: int,
    ):
        super().__init__()
        self.emb = nn.Embedding(num_tokens, dim)
        self.norm = nn.LayerNorm(dim)
        self.to_logits = OutputHead(
            dim,
            vocab_size=num_tokens,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.emb(x)
        x = self.norm(x)
        return self.to_logits(x)


def bitnet_head_only(
    *,
    num_tokens: int,
    dim: int = 1024,
) -> BitNetHeadOnly:
    return BitNetHeadOnly(dim=dim, num_tokens=num_tokens)


if __name__ == "__main__":
    model = bitnet_head_only(num_tokens=20000)
    model.eval()
    module = torch_mlir.compile(
        model,
        torch.randint(0, 20000, (1, 1024)),
        output_type=torch_mlir.OutputType.LINALG_ON_TENSORS,
    )
    print(module)

