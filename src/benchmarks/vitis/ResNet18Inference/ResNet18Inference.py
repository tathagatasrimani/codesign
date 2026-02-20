import os
import sys
import torch.nn as nn
import torch.nn.functional as F

# Add parent vitis directory so sibling sub-block packages are importable.
_here = os.path.dirname(os.path.abspath(__file__))
_vitis = os.path.dirname(_here)
if _vitis not in sys.path:
    sys.path.insert(0, _vitis)

from ResNet18Stem.ResNet18Stem import ResNet18Stem
from ResNet18Block.ResNet18Block import ResNet18Block
from ResNet18Classifier.ResNet18Classifier import ResNet18Classifier


class ResNet18Inference(nn.Module):
    """ResNet-18 image classification inference.

    Each individual BasicBlock is a separate sub-module so that StreamHLS
    can optimize each one independently (stacking two blocks in one module
    makes the optimization space too large).

    Sub-block attributes matched to block_types in the system YAML:
      stem         -> ResNet18Stem
      layer1_block -> ResNet18Block(64,  64,  stride=1)  called 2×
      layer2_down  -> ResNet18Block(64,  128, stride=2)  called 1×
      layer2_block -> ResNet18Block(128, 128, stride=1)  called 1×
      layer3_down  -> ResNet18Block(128, 256, stride=2)  called 1×
      layer3_block -> ResNet18Block(256, 256, stride=1)  called 1×
      layer4_down  -> ResNet18Block(256, 512, stride=2)  called 1×
      layer4_block -> ResNet18Block(512, 512, stride=1)  called 1×
      classifier   -> ResNet18Classifier
    """

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.stem = ResNet18Stem()

        # Layer 1: two identical BasicBlocks (64→64, no downsample)
        self.layer1_block = ResNet18Block(64, 64, stride=1)

        # Layer 2: first block downsizes, second is plain
        self.layer2_down  = ResNet18Block(64,  128, stride=2)
        self.layer2_block = ResNet18Block(128, 128, stride=1)

        # Layer 3: first block downsizes, second is plain
        self.layer3_down  = ResNet18Block(128, 256, stride=2)
        self.layer3_block = ResNet18Block(256, 256, stride=1)

        # Layer 4: first block downsizes, second is plain
        self.layer4_down  = ResNet18Block(256, 512, stride=2)
        self.layer4_block = ResNet18Block(512, 512, stride=1)

        self.classifier = ResNet18Classifier(num_classes=num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, 3, 224, 224)
        Returns:
            (batch, num_classes)
        """
        # Stem: pre-pad 224→230 (3 per side) so the 7×7 conv runs with pad=0
        x = self.stem(F.pad(x, (3, 3, 3, 3)))          # → (batch, 64, 56, 56)

        # Layer 1 – two identical stride-1 blocks (64→64, 56→56)
        identity = x
        x = self.layer1_block(F.pad(identity, (2, 2, 2, 2)), identity)  # 60→56
        identity = x
        x = self.layer1_block(F.pad(identity, (2, 2, 2, 2)), identity)  # 60→56

        # Layer 2 – stride-2 down-block (64→128, 56→28), then stride-1 (128→128, 28→28)
        identity = x
        x = self.layer2_down(F.pad(identity, (3, 3, 3, 3)), identity)   # 62→28
        identity = x
        x = self.layer2_block(F.pad(identity, (2, 2, 2, 2)), identity)  # 32→28

        # Layer 3 – stride-2 down-block (128→256, 28→14), then stride-1 (256→256, 14→14)
        identity = x
        x = self.layer3_down(F.pad(identity, (3, 3, 3, 3)), identity)   # 34→14
        identity = x
        x = self.layer3_block(F.pad(identity, (2, 2, 2, 2)), identity)  # 18→14

        # Layer 4 – stride-2 down-block (256→512, 14→7), then stride-1 (512→512, 7→7)
        identity = x
        x = self.layer4_down(F.pad(identity, (3, 3, 3, 3)), identity)   # 20→7
        identity = x
        x = self.layer4_block(F.pad(identity, (2, 2, 2, 2)), identity)  # 11→7

        x = self.classifier(x)
        return x
