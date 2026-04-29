import torch.nn as nn
from functools import partial


def crossvit_tiny_224():
    return {
        "img_size": [240, 224],
        "patch_size": [12, 16],
        "embed_dim": [96, 192],
        "depth": [[1, 4, 0], [1, 4, 0], [1, 4, 0]],
        "num_heads": [3, 3],
        "mlp_ratio": [4, 4, 1],
        "qkv_bias": True,
        "norm_layer": partial(nn.LayerNorm, eps=1e-6),
    }


def crossvit_small_224():
    return {
        "img_size": [240, 224],
        "patch_size": [12, 16],
        "embed_dim": [192, 384],
        "depth": [[1, 4, 0], [1, 4, 0], [1, 4, 0]],
        "num_heads": [6, 6],
        "mlp_ratio": [4, 4, 1],
        "qkv_bias": True,
        "norm_layer": partial(nn.LayerNorm, eps=1e-6),
    }


def crossvit_base_224():
    return {
        "img_size": [240, 224],
        "patch_size": [12, 16],
        "embed_dim": [384, 768],
        "depth": [[1, 4, 0], [1, 4, 0], [1, 4, 0]],
        "num_heads": [12, 12],
        "mlp_ratio": [4, 4, 1],
        "qkv_bias": True,
        "norm_layer": partial(nn.LayerNorm, eps=1e-6),
    }


def crossvit_15_224():
    return {
        "img_size": [240, 224],
        "patch_size": [12, 16],
        "embed_dim": [192, 384],
        "depth": [[1, 5, 0], [1, 5, 0], [1, 5, 0]],
        "num_heads": [6, 6],
        "mlp_ratio": [3, 3, 1],
        "qkv_bias": True,
        "norm_layer": partial(nn.LayerNorm, eps=1e-6),
    }


def crossvit_18_224():
    return {
        "img_size": [240, 224],
        "patch_size": [12, 16],
        "embed_dim": [224, 448],
        "depth": [[1, 6, 0], [1, 6, 0], [1, 6, 0]],
        "num_heads": [7, 7],
        "mlp_ratio": [3, 3, 1],
        "qkv_bias": True,
        "norm_layer": partial(nn.LayerNorm, eps=1e-6),
    }


def crossvit_15_dagger_224():
    return {
        "img_size": [240, 224],
        "patch_size": [12, 16],
        "embed_dim": [192, 384],
        "depth": [[1, 5, 0], [1, 5, 0], [1, 5, 0]],
        "num_heads": [6, 6],
        "mlp_ratio": [3, 3, 1],
        "qkv_bias": True,
        "norm_layer": partial(nn.LayerNorm, eps=1e-6),
        "multi_conv": True,
    }


crossvit_kwargs_map = {
    "crossvit_tiny_224": crossvit_tiny_224(),
    "crossvit_small_224": crossvit_small_224(),
    "crossvit_base_224": crossvit_base_224(),
    "crossvit_15_224": crossvit_15_224(),
    "crossvit_18_224": crossvit_18_224(),
    "crossvit_15_dagger_224": crossvit_15_dagger_224(),
}
