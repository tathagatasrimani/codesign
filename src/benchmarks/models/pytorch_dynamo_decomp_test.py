import torch
import torch._decomp
import os
from torch import nn

from torch._ops import ops
from torch._inductor.decomposition import register_decomposition
from torch.fx import symbolic_trace

# os.environ["TORCH_COMPILE_DEBUG"] = '1'
device = "cpu"
    
def native_relu(x):
    print("Custom ReLU decomposition is called")
    return (x > 0) * x

@register_decomposition(ops.aten.relu)
def relu_decomposition(input):
    return native_relu(input)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.ReLU(),
        )

    @torch.compile()
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

def read_weights_from_file(x, y, z):
    W_q = torch.zeros((x, y, z))
    W_k = torch.zeros((x, y, z))
    return W_q, W_k

if __name__ == "__main__":
    model = NeuralNetwork()
    W_q, W_k = read_weights_from_file(12, 28, 28)
    z = W_k + W_q
    x = torch.rand(1, 28, 28, device=device, requires_grad=True)
    model.forward(x)
    symbolic_traced : torch.fx.GraphModule = symbolic_trace(model, concrete_args={'x': x})
    print(symbolic_traced.code)