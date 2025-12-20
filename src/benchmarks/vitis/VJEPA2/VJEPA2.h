// Copyright (c) Meta Platforms, Inc. and affiliates.
// Converted to C scaffold by GitHub Copilot.
// This file is MIT-licensed under the original project's terms.

#ifndef VJEPA2_H
#define VJEPA2_H

#include <stdint.h>

typedef struct {
    int dim;
    int num_heads;
    int mlp_ratio_num; /* use int for simple ratio representation */
    float drop_rate;
    int use_rope;
    /* Placeholder params for attention/MLP weights */
    float *attn_proj_weight;
    float *mlp_fc2_weight;
} Block;

typedef struct {
    int img_height;
    int img_width;
    int patch_size;
    int num_frames;
    int tubelet_size;
    int in_chans;
    int embed_dim;
    int depth;
    int num_heads;
    int num_patches;
    int is_video;
    int use_rope;
    float *pos_embed; /* [1 x num_patches x embed_dim] flattened */
    Block *blocks;    /* length == depth */
    float *norm_bias; /* placeholder for layer norm bias */
    float *norm_weight; /* placeholder for layer norm weight */
} VisionTransformer;

/* Create / destroy */
VisionTransformer *vit_create(int img_h, int img_w, int patch_size, int num_frames, int in_chans, int embed_dim, int depth, int num_heads, int use_rope);
void vit_free(VisionTransformer *vit);

/* Forward: input: x (B x C x T x H x W) or (B x C x H x W) depending on is_video
   This implementation expects a flattened float buffer and returns output in 'out' buffer
   with shape (B x num_patches x embed_dim). The function is a placeholder and will simply
   project patches with a naive average and add positional embedding. */
void vit_forward(VisionTransformer *vit, const float *x, int B, int C, int T, int H, int W, float *out);

#endif /* VJEPA2_H */