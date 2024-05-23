import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import List

device = "cpu"

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward  # return a python callable

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

if __name__ == "__main__":
    x_count=0
    def increase_count():
        x_count=x_count+1
    model = NeuralNetwork().to(device)
    W_q, W_k = read_weights_from_file(12, 28, 28)
    z = W_k + W_q
    x = torch.rand(1, 28, 28, device=device, requires_grad=True)
    h = x[0][0][0].register_hook(increase_count)
    model.forward(x)
    print(x_count)