import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet18BasicLayer(nn.Module):
    """One ResNet-18 stage: two stacked BasicBlocks.

    The first block applies the given stride and, when in_channels != out_channels,
    a 1x1 downsample projection.  The second block is always stride=1 with no
    downsample.  This is reused for all four ResNet-18 stages by varying
    in_channels, out_channels, and stride:

      Layer 1: in=64,  out=64,  stride=1  → (batch, 64,  56, 56)
      Layer 2: in=64,  out=128, stride=2  → (batch, 128, 28, 28)
      Layer 3: in=128, out=256, stride=2  → (batch, 256, 14, 14)
      Layer 4: in=256, out=512, stride=2  → (batch, 512,  7,  7)

    Weights (per layer):
      Layer 1: 2 × (64×64×9 + 64×64×9)         × 4B =    589,824 bytes
      Layer 2: (128×64×9 + 128×128×9 + 128×64)
             + (128×128×9 + 128×128×9)           × 4B =  2,097,152 bytes
      Layer 3: (256×128×9 + 256×256×9 + 256×128)
             + (256×256×9 + 256×256×9)           × 4B =  8,388,608 bytes
      Layer 4: (512×256×9 + 512×512×9 + 512×256)
             + (512×512×9 + 512×512×9)           × 4B = 33,554,432 bytes
    """

    def __init__(self, in_channels: int = 64, out_channels: int = 64,
                 stride: int = 1) -> None:
        super().__init__()

        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
            )

        self.block1 = BasicBlock(in_channels, out_channels, stride, downsample)
        self.block2 = BasicBlock(out_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, in_channels, H, W)
        Returns:
            (batch, out_channels, H // stride, W // stride)
        """
        x = self.block1(x)
        x = self.block2(x)
        return x
