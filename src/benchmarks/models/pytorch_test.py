import os
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.fx import symbolic_trace, Tracer, GraphModule

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

    @torch.compile()
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class MyTracer(Tracer):
    def call_module(self, module, forward, args, kwargs):
        if isinstance(module, nn.Sequential):
            for layer in module:
                args = self.call_module(layer, layer.forward, args, kwargs)
            return args
        else:
            return super().call_module(module, forward, args, kwargs)
        
def read_weights_from_file(x, y, z):
    W_q = np.zeros((x, y, z))
    W_k = np.zeros((x, y, z))
    return W_q, W_k

if __name__ == "__main__":
    model = NeuralNetwork().to(device)
    tracer = MyTracer()
    graph = tracer.trace(model)
    symbolic_traced : torch.fx.GraphModule = symbolic_trace(model, graph)
    print(symbolic_traced.code)
    W_q, W_k = read_weights_from_file(12, 28, 28)
    z = W_k + W_q
    x = torch.rand(1, 28, 28, device=device)
    model.forward(x)