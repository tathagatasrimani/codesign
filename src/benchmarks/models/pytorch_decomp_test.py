import torch
import torch._decomp
import os
from torch import nn

from torch._ops import ops
#from torch._inductor.decomposition import register_decomposition, remove_decompositions, get_decompositions
from torch._decomp import register_decomposition, global_decomposition_table
from torch.fx import symbolic_trace

# os.environ["TORCH_COMPILE_DEBUG"] = '1'
device = "cpu"
aten = ops.aten
def native_view(x, stride):
    return [row[::stride[0]] for row in x[::stride[1]]]

global_decomposition_table[aten.view] = native_view

def native_permute(x, stride):
    return [row[::stride[0]] for row in x[::stride[1]]]

global_decomposition_table[aten.permute] = native_permute

def native_relu(x):
    print("run native relu")
    return (x > 0) * x

global_decomposition_table[aten.relu] = native_relu

def native_addmm(input, mat1, mat2):
    rows_A, cols_A = len(mat1), len(mat1[0])
    rows_B, cols_B = len(mat2), len(mat2[0])
    result = input
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += mat1[i][k] * mat2[k][j]
    return result

global_decomposition_table[aten.addmm] = native_addmm

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
    symbolic_traced : torch.fx.GraphModule = symbolic_trace(model)
    print(symbolic_traced.code)