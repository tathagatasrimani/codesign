import os
import torch
from torch.fx import symbolic_trace
import operator
from torch import nn
import numpy as np
from torch._ops import ops
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cpu"
torch.fx.wrap('len')

def native_relu(x):
    print("Custom ReLU decomposition is called")
    return (x > 0) * x

def native_linear(x):
    D = len(x)
    weights = torch.ones((D, D))
    bias = torch.ones((D,))
    return x * weights + bias

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(28*28, 28*28)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear(x)
        x = self.relu(x)
        logits = self.linear(x)
        return logits

def read_weights_from_file(x, y, z):
    W_q = np.zeros((x, y, z))
    W_k = np.zeros((x, y, z))
    return W_q, W_k

if __name__ == "__main__":
    model = NeuralNetwork().to(device)
    W_q, W_k = read_weights_from_file(12, 28, 28)
    z = W_k + W_q
    x = torch.rand(1, 28, 28, device=device)
    #model.forward(x)

    traced = symbolic_trace(NeuralNetwork(), concrete_args={'x': x})
    traced.graph.print_tabular()

    relu_patterns = set(['relu'])
    linear_patterns = set(['linear'])

    for n in traced.graph.nodes:
        # If the target matches one of the patterns
        if any(n.target == pattern for pattern in relu_patterns):
            # Set the insert point, add the new node, and replace all uses
            # of `n` with the new node
            with traced.graph.inserting_after(n):
                new_node = traced.graph.call_function(native_relu, n.args, n.kwargs)
                n.replace_all_uses_with(new_node)
            # Remove the old node from the graph
            traced.graph.erase_node(n)
        if any(n.target == pattern for pattern in linear_patterns):
            # Set the insert point, add the new node, and replace all uses
            # of `n` with the new node
            with traced.graph.inserting_after(n):
                new_node = traced.graph.call_function(native_linear, n.args, n.kwargs)
                n.replace_all_uses_with(new_node)
            # Remove the old node from the graph
            traced.graph.erase_node(n)

    traced.recompile()
    traced.graph.print_tabular()

    