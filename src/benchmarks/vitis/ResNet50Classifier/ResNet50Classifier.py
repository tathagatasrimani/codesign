import torch
import torch.nn as nn


class ResNet50Classifier(nn.Module):
    """ResNet-50 classification head: global average pool + fully-connected.

    Input:  (batch, 2048, 7, 7)
    Output: (batch, num_classes)

    Weights:
      fc: 2048 × num_classes + num_classes bias
      Default (num_classes=1000): 2,049,000 params → 8,196,000 bytes
    """

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, 2048, 7, 7)
        Returns:
            (batch, num_classes)
        """
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
