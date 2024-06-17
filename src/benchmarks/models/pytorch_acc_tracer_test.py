import os
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.fx.experimental.proxy_tensor import make_fx

device = "cpu"

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

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def read_weights_from_file(x, y, z):
    W_q = np.zeros((x, y, z))
    W_k = np.zeros((x, y, z))
    return W_q, W_k

def native_mm(mtx1, mtx2):
    result = torch.zeros((mtx1.shape[0], mtx2.shape[1]))
    for i in range(mtx1.shape[0]):
        for j in range(mtx2.shape[1]):
            for k in range(mtx2.shape[0]):
                result[i, j] += mtx1[i, k] * mtx2[k, j]
    return result

if __name__ == "__main__":
    model = NeuralNetwork().to(device)
    W_q, W_k = read_weights_from_file(12, 28, 28)
    z = W_k + W_q
    x = torch.rand(1, 28, 28, device=device)
    model.forward(x)

    decomposition_table = torch._decomp.get_decompositions([
        torch.ops.aten.addmm,
        torch.ops.aten.permute,
        torch.ops.aten.mul,
        torch.ops.aten.relu,
        torch.ops.aten.le,
        torch.ops.aten.mm,
        torch.ops.aten.scalar_tensor,  
        torch.ops.aten.add,
        torch.ops.aten.expand,
        torch.ops.aten.alias,
        torch.ops.aten.view,
        torch.ops.aten.as_strided
    ])
    # decomposition_table.update({torch.ops.aten.mm.default: native_mm})

    traced = make_fx(model, decomposition_table=decomposition_table)(x)
    traced.print_readable()
    traced.graph.print_tabular()