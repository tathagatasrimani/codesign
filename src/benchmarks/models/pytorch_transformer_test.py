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
        self.transformer_model = nn.Transformer(d_model= 28, nhead=2, num_encoder_layers=2, dim_feedforward=64)

    def forward(self, src, tgt):
        logits = self.transformer_model(src, tgt)
        return logits

def read_weights_from_file(x, y, z):
    W_q = np.rand((x, y, z))
    W_k = np.rand((x, y, z))
    return W_q, W_k

if __name__ == "__main__":
    model = NeuralNetwork().to(device)
    src = torch.rand((5,28), device=device)
    tgt = torch.rand((5,28), device=device)
    model.forward(src, tgt)
