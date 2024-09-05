import os
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cpu"

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_soft_stack = nn.Sequential(
            nn.Conv2d(1, 4, 3, 1),
            nn.Conv2d(4, 16, 5, 1),
            nn.Flatten(),
            nn.Linear(484, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        logits = self.conv_soft_stack(x)
        return logits

if __name__ == "__main__":
    model = NeuralNetwork().to(device)
    x = torch.rand(1, 28, 28, device=device)
    model.forward(x)