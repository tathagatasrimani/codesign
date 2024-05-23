# https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import List
from torch.profiler import profile, record_function, ProfilerActivity
import torch.autograd.profiler as profiler
from torch.profiler.profiler import ExecutionTraceObserver
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
    model = NeuralNetwork().to(device)
    W_q, W_k = read_weights_from_file(12, 28, 28)
    z = W_k + W_q
    x = torch.rand(1, 28, 28, device=device, requires_grad=True)
    with profile(activities=[ProfilerActivity.CPU], 
                profile_memory=True,
                record_shapes=True,
                with_modules=True, 
                execution_trace_observer=(
                ExecutionTraceObserver().register_callback("./execution_trace.json")
            )) as prof:
        with record_function("model_inference"):
            model.forward(x)
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total"))

