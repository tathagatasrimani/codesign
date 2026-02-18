import os
import sys
import torch
import torch.nn as nn

# Add parent vitis directory so sibling sub-block packages are importable.
_here = os.path.dirname(os.path.abspath(__file__))
_vitis = os.path.dirname(_here)
if _vitis not in sys.path:
    sys.path.insert(0, _vitis)

from VJEPASpatialAttention.VJEPASpatialAttention import VJEPASpatialAttention
from VJEPATemporalAttention.VJEPATemporalAttention import VJEPATemporalAttention
from VJEPAFeedForward.VJEPAFeedForward import VJEPAFeedForward
from VJEPAPredictorAttention.VJEPAPredictorAttention import VJEPAPredictorAttention
from VJEPAPredictorFeedForward.VJEPAPredictorFeedForward import VJEPAPredictorFeedForward


class VJEPAInference(nn.Module):
    """V-JEPA full inference: context encoder + predictor.

    V-JEPA (Video Joint Embedding Predictive Architecture) has two networks:

    1. Context encoder (ViT-H/16, 32 layers): processes all video tokens
       with factorised space-time attention:
         - Spatial attention: 196 patches/frame, per-frame (16 × 196 = 3136 total)
         - Temporal attention: all 3136 tokens jointly
         - FFN: MLP over all tokens

    2. Predictor (narrow transformer, 12 layers): takes encoder outputs
       (projected to pred_hidden=384) plus learnable mask tokens for the
       target positions, and predicts target patch representations.
       Input: 3136 positions (context + target) at pred_hidden=384.

    The predictor input projection (enc_hidden → pred_hidden) is a single
    Linear layer in the inference module (not a separately tracked block).

    Real dimensions:
      Encoder:   hidden=1280, n_heads=16, head_dim=80, ffn=5120, n_layers=32.
      Predictor: pred_hidden=384, n_heads=12, head_dim=32, ffn=1536, n_layers=12.

    Sub-block attributes matched to block_types in the system YAML:
      spatial_attention    -> VJEPASpatialAttention    (encoder, per-frame)
      temporal_attention   -> VJEPATemporalAttention   (encoder, all tokens)
      feedforward          -> VJEPAFeedForward         (encoder FFN)
      predictor_attention  -> VJEPAPredictorAttention  (predictor SA)
      predictor_feedforward -> VJEPAPredictorFeedForward (predictor FFN)
    """

    def __init__(self, hidden=1280, n_heads=16, head_dim=80,
                 ffn_intermediate=5120, n_frames=16, spatial_patches=196,
                 n_layers=32,
                 pred_hidden=384, pred_n_heads=12, pred_head_dim=32,
                 pred_ffn=1536, n_pred_layers=12):
        super().__init__()
        self.n_frames = n_frames
        self.spatial_patches = spatial_patches
        self.total_patches = n_frames * spatial_patches
        self.n_layers = n_layers
        self.hidden = hidden
        self.pred_hidden = pred_hidden
        self.n_pred_layers = n_pred_layers

        # Context encoder sub-blocks
        self.spatial_attention = VJEPASpatialAttention(hidden, n_heads, head_dim)
        self.temporal_attention = VJEPATemporalAttention(hidden, n_heads, head_dim)
        self.feedforward = VJEPAFeedForward(hidden, ffn_intermediate)

        # Encoder-to-predictor projection (called once, not tracked per-layer)
        self.pred_proj = nn.Linear(hidden, pred_hidden, bias=False)

        # Predictor sub-blocks
        self.predictor_attention = VJEPAPredictorAttention(
            pred_hidden, pred_n_heads, pred_head_dim)
        self.predictor_feedforward = VJEPAPredictorFeedForward(
            pred_hidden, pred_ffn)

    def forward(self, x):
        """
        Args:
            x: (batch, total_patches, hidden)
               — all n_frames × spatial_patches tokens
        Returns:
            (batch, total_patches, pred_hidden) — predicted patch representations
        """
        bsz = x.shape[0]

        # === Context encoder ===
        for _ in range(self.n_layers):
            # Spatial: (batch * n_frames, spatial_patches, hidden)
            x_spatial = x.view(bsz * self.n_frames, self.spatial_patches, self.hidden)
            x_spatial = self.spatial_attention(x_spatial)
            x = x_spatial.view(bsz, self.total_patches, self.hidden)

            x = self.temporal_attention(x)
            x = self.feedforward(x)

        # Project encoder output to predictor hidden dim
        # (batch, total_patches, pred_hidden)
        pred_x = self.pred_proj(x)

        # === Predictor ===
        # Input: encoded context + mask tokens, all total_patches positions
        for _ in range(self.n_pred_layers):
            pred_x = self.predictor_attention(pred_x)
            pred_x = self.predictor_feedforward(pred_x)

        return pred_x
