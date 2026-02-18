import torch
import torch.nn as nn
import torch.nn.functional as F


class VJEPAPredictorFeedForward(nn.Module):
    """V-JEPA predictor MLP feedforward block.

    Two-layer MLP with GELU activation in the predictor transformer.
    Operates at the narrower predictor hidden dimension (384) over all
    3136 video positions.

    Real V-JEPA predictor dimensions:
      pred_hidden=384, ffn_intermediate=1536 (MLP ratio 4:1).
    """

    def __init__(self, pred_hidden=384, ffn_intermediate=1536):
        super().__init__()
        self.fc1 = nn.Linear(pred_hidden, ffn_intermediate, bias=False)
        self.fc2 = nn.Linear(ffn_intermediate, pred_hidden, bias=False)

    def forward(self, x):
        """
        Args:
            x: (batch, PRED_SEQ_LEN, pred_hidden)
        Returns:
            (batch, PRED_SEQ_LEN, pred_hidden)
        """
        return self.fc2(F.gelu(self.fc1(x)))
