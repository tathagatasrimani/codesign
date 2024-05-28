# https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
import os
os.environ["TORCH_COMPILE_DEBUG"] = '1'
from torch._inductor.select_algorithm import extern_kernels

import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch._inductor.decomposition import register_decomposition

device = "cpu"
aten = torch.ops.aten

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    @torch._dynamo.optimize()
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def read_weights_from_file(x, y, z):
    W_q = np.zeros((x, y, z))
    W_k = np.zeros((x, y, z))
    return W_q, W_k

@register_decomposition(aten.view)
def native_view(x, stride):
    return [row[::stride[0]] for row in x[::stride[1]]]

@register_decomposition(aten.permute)
def native_permute(x, stride):
    return [row[::stride[0]] for row in x[::stride[1]]]

@register_decomposition(aten.relu)
def native_relu(x):
    return (x > 0) * x

@register_decomposition(aten.addmm)
def native_addmm(input, mat1, mat2):
    rows_A, cols_A = len(mat1), len(mat1[0])
    rows_B, cols_B = len(mat2), len(mat2[0])
    result = input
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += mat1[i][k] * mat2[k][j]
    return result

if __name__ == "__main__":
    model = NeuralNetwork().to(device)
    W_q, W_k = read_weights_from_file(12, 28, 28)
    z = W_k + W_q
    x = torch.rand(1, 28, 28, device=device, requires_grad=True)
    model.forward(x)
