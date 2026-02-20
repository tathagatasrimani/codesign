import torch
import torch.nn as nn


class ResNet18Stem(nn.Module):
    """ResNet-18 stem: initial 7x7 conv + two max-pool ops.

    Takes a pre-padded 230x230 input (224 + 2*3 border) so that conv1
    can use padding=0 and avoid MLIR pad ops at the module boundary.

    Input:  (batch, 3, 230, 230)   # pre-padded; semantic size is 224x224
    Output: (batch, 64, 56, 56)

    Dimension trace (no internal padding ops):
      conv1  (7x7, stride=1, pad=0): 230 → 224
      pool1  (2x2, stride=2):        224 → 112
      relu
      maxpool(2x2, stride=2):        112 →  56

    Weights:
      conv1: 64 × 3 × 7 × 7 (no bias) = 9,408 params → 37,632 bytes
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=0, bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Args:
            x: (batch, 3, 230, 230)
        Returns:
            (batch, 64, 56, 56)
        """
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x
