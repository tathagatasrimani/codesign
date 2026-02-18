import os
import sys
import torch
import torch.nn as nn

# Add parent vitis directory so sibling sub-block packages are importable.
_here = os.path.dirname(os.path.abspath(__file__))
_vitis = os.path.dirname(_here)
if _vitis not in sys.path:
    sys.path.insert(0, _vitis)

from OpenVLADinoAttention.OpenVLADinoAttention import OpenVLADinoAttention
from OpenVLADinoFeedForward.OpenVLADinoFeedForward import OpenVLADinoFeedForward
from OpenVLAVisionAttention.OpenVLAVisionAttention import OpenVLAVisionAttention
from OpenVLAVisionFeedForward.OpenVLAVisionFeedForward import OpenVLAVisionFeedForward
from OpenVLAVisionProjection.OpenVLAVisionProjection import OpenVLAVisionProjection
from OpenVLALMPrefillAttention.OpenVLALMPrefillAttention import OpenVLALMPrefillAttention
from OpenVLALMFeedForward.OpenVLALMFeedForward import OpenVLALMFeedForward
from OpenVLALMDecodeAttention.OpenVLALMDecodeAttention import OpenVLALMDecodeAttention


class OpenVLAInference(nn.Module):
    """OpenVLA full inference: dual vision encoding + projection + LM prefill + action decode.

    OpenVLA (Open Vision-Language-Action) is a 7B-parameter VLA model for
    robot manipulation.  It uses a Prismatic dual-encoder architecture:

      1. DINOv2 ViT-L/14 encoder  (24 layers): semantic/structural features
      2. SigLIP-SO400M encoder     (27 layers): vision-language aligned features
      Both encoders process the same 256 image patches (224×224 at 14px patch size).

      3. Vision projection MLP: concatenates DINOv2 (1024) + SigLIP (1152) =
         2176 features per patch, then projects to lm_hidden=4096 via 2-layer MLP.

      4. LM prefill (Llama 2 7B, 32 layers): processes 256 vision tokens +
         256 language instruction tokens = 512 total.

      5. LM decode: autoregressively outputs 7 discrete action tokens (7-DoF).

    Sub-block attributes matched to block_types in the system YAML:
      dino_attention      -> OpenVLADinoAttention       (DINOv2 ViT-L/14 SA)
      dino_feedforward    -> OpenVLADinoFeedForward     (DINOv2 FFN)
      vision_attention    -> OpenVLAVisionAttention     (SigLIP SA)
      vision_feedforward  -> OpenVLAVisionFeedForward   (SigLIP FFN)
      vision_projection   -> OpenVLAVisionProjection    (concat 2176→4096 MLP)
      lm_prefill_attention -> OpenVLALMPrefillAttention (LM prefill SA)
      lm_feedforward       -> OpenVLALMFeedForward      (LM SwiGLU FFN)
      lm_decode_attention  -> OpenVLALMDecodeAttention  (LM decode + KV cache)
    """

    def __init__(self,
                 # DINOv2 ViT-L/14
                 dino_hidden=1024, dino_n_heads=16, dino_head_dim=64,
                 dino_ffn=4096, n_dino_layers=24,
                 # SigLIP-SO400M
                 vis_hidden=1152, vis_n_heads=16, vis_head_dim=72,
                 vis_ffn=4304, n_vision_layers=27,
                 # Vision projection
                 n_patches=256,
                 # Language model (Llama 2 7B)
                 lm_hidden=4096, lm_n_heads=32, lm_head_dim=128,
                 lm_ffn=11008, n_lm_layers=32,
                 # Action decode
                 n_decode=7, cache_len=512):
        super().__init__()
        self.n_dino_layers = n_dino_layers
        self.n_vision_layers = n_vision_layers
        self.n_lm_layers = n_lm_layers
        self.n_decode = n_decode
        self.lm_n_heads = lm_n_heads
        self.lm_head_dim = lm_head_dim
        self.cache_len = cache_len

        # DINOv2 encoder sub-blocks
        self.dino_attention = OpenVLADinoAttention(dino_hidden, dino_n_heads, dino_head_dim)
        self.dino_feedforward = OpenVLADinoFeedForward(dino_hidden, dino_ffn)

        # SigLIP encoder sub-blocks
        self.vision_attention = OpenVLAVisionAttention(vis_hidden, vis_n_heads, vis_head_dim)
        self.vision_feedforward = OpenVLAVisionFeedForward(vis_hidden, vis_ffn)

        # Vision projection (concatenated DINOv2 + SigLIP → LM space)
        self.vision_projection = OpenVLAVisionProjection(
            proj_in=dino_hidden + vis_hidden, lm_hidden=lm_hidden)

        # Language model sub-blocks
        self.lm_prefill_attention = OpenVLALMPrefillAttention(
            lm_hidden, lm_n_heads, lm_head_dim)
        self.lm_feedforward = OpenVLALMFeedForward(lm_hidden, lm_ffn)
        self.lm_decode_attention = OpenVLALMDecodeAttention(
            lm_hidden, lm_n_heads, lm_head_dim, cache_len)

    def forward(self, x_dino, x_siglip, x_lang):
        """
        Args:
            x_dino:   (batch, n_patches, dino_hidden)  — DINOv2 patch embeddings
            x_siglip: (batch, n_patches, vis_hidden)   — SigLIP patch embeddings
            x_lang:   (batch, lang_len,  lm_hidden)    — language instruction tokens
        Returns:
            (batch, 1, lm_hidden) — last decoded action token embedding
        """
        bsz = x_dino.shape[0]

        # === DINOv2 encoding ===
        for _ in range(self.n_dino_layers):
            x_dino = self.dino_attention(x_dino)
            x_dino = self.dino_feedforward(x_dino)

        # === SigLIP encoding ===
        for _ in range(self.n_vision_layers):
            x_siglip = self.vision_attention(x_siglip)
            x_siglip = self.vision_feedforward(x_siglip)

        # === Vision projection ===
        # Concatenate per-patch features from both encoders, then project to LM space.
        # torch.cat is in the top-level module so StreamHLS doesn't compile it.
        x_vis_cat = torch.cat([x_dino, x_siglip], dim=-1)  # (batch, n_patches, 2176)
        vis_tokens = self.vision_projection(x_vis_cat)      # (batch, n_patches, lm_hidden)

        # === LM prefill ===
        x_lm = torch.cat([vis_tokens, x_lang], dim=1)       # (batch, n_patches + lang_len, lm_hidden)
        for _ in range(self.n_lm_layers):
            x_lm = self.lm_prefill_attention(x_lm)
            x_lm = self.lm_feedforward(x_lm)

        # === Action decode ===
        k_cache = torch.zeros(bsz, self.lm_n_heads, self.cache_len, self.lm_head_dim)
        v_cache = torch.zeros(bsz, self.lm_n_heads, self.cache_len, self.lm_head_dim)
        token_x = x_lm[:, -1:, :]

        for _ in range(self.n_decode):
            for _ in range(self.n_lm_layers):
                token_x = self.lm_decode_attention(token_x, k_cache, v_cache)
                token_x = self.lm_feedforward(token_x)

        return token_x
