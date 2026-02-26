import torch

class dot_prod(torch.nn.Module):
  def __init__(self):
    super(dot_prod, self).__init__()
  def forward(self, a, b):
    return a @ b
