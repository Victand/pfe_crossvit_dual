import torch
import torch.nn as nn

from CrossViT.models import crossvit
from pfe_crossvit_dual.training.model.recording_layers import MultiScaleBlock


class DualCrossVitYolo(crossvit.VisionTransformer):
    """
    Custom CrossVit class to accept two images in input : Original/detoured herb image.
    Inherits from the standard VisionTransformer (CrossViT implementation).
    Adds custom Blocks to record Attentions/cross-attention to perform Attention rollout.
    """

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=(8, 16),
        embed_dim=(192, 384),
        depth=([1, 3, 1], [1, 3, 1], [1, 3, 1]),
        num_heads=(6, 12),
        mlp_ratio=(2.0, 2.0, 4.0),
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        **kwargs,
    ):
        super().__init__(
            img_size,
            patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=False,
            qk_scale=None,
            drop_rate=drop_rate,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            hybrid_backbone=None,
            norm_layer=nn.LayerNorm,
            multi_conv=False,
            **kwargs,
        )
        self.kwargs = kwargs
        num_patches = crossvit._compute_num_patches(img_size, patch_size)

        self.patch_size = patch_size

        self.default_weights = [
            torch.ones((num_patches[i], 1)) for i in range(len(self.img_size))
        ]

        total_depth = sum([sum(x[-2:]) for x in depth])
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule
        dpr_ptr = 0
        self.blocks = nn.ModuleList()
        for idx, block_cfg in enumerate(depth):
            curr_depth = max(block_cfg[:-1]) + block_cfg[-1]
            dpr_ = dpr[dpr_ptr : dpr_ptr + curr_depth]
            blk = MultiScaleBlock(
                embed_dim,
                block_cfg,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_,
                norm_layer=norm_layer,
            )
            dpr_ptr += curr_depth
            self.blocks.append(blk)

    def forward_features(self, x_small, x_large, weights, num_patches):  # pyright: ignore[reportIncompatibleMethodOverride]
        B_large = x_large.shape[0]

        tmp_s = self.patch_embed[0](x_small)
        cls_s = self.cls_token[0].expand(x_small.shape[0], -1, -1)
        tmp_s = torch.cat((cls_s, tmp_s), dim=1)

        pos_embed_s = self.pos_embed[0]
        if tmp_s.shape[1] != pos_embed_s.shape[1]:
            # On sépare le CLS token du reste
            pos_embed_tok = pos_embed_s[:, 0:1]
            pos_embed_grid = pos_embed_s[:, 1:]

            # On calcule la taille de la grille actuelle
            gs_old = int(pos_embed_grid.shape[1] ** 0.5)
            gs_new = int((tmp_s.shape[1] - 1) ** 0.5)

            # Interpolation 2D pour adapter le Pos Embed à la nouvelle grille
            pos_embed_grid = pos_embed_grid.reshape(1, gs_old, gs_old, -1).permute(
                0, 3, 1, 2
            )
            pos_embed_grid = torch.nn.functional.interpolate(
                pos_embed_grid,
                size=(gs_new, gs_new),
                mode="bicubic",
                align_corners=False,
            )
            pos_embed_grid = pos_embed_grid.permute(0, 2, 3, 1).reshape(
                1, -1, pos_embed_s.shape[-1]
            )

            pos_embed_s = torch.cat((pos_embed_tok, pos_embed_grid), dim=1)
        # ------------------------------------

        tmp_s = tmp_s + pos_embed_s
        tmp_s = self.pos_drop(tmp_s)

        # Réduction : moyenne des N patchs pour retrouver le Batch Size B
        # [B*N, Tokens, Dim] -> [B, N, Tokens, Dim] -> [B, Tokens, Dim]
        tmp_s = tmp_s.view(B_large, num_patches, tmp_s.shape[1], tmp_s.shape[2]).mean(
            dim=1
        )

        # --- Branche Large (Global + Heatmap) ---
        tmp_l = self.patch_embed[1](x_large)
        if weights is not None:
            # Interpolation des poids pour coller à la grille de tokens (ex: 14x14 ou 16x16)
            if weights.shape[1] != tmp_l.shape[1]:
                grid_w = int(weights.shape[1] ** 0.5)
                w_2d = weights.view(B_large, 1, grid_w, grid_w)
                target_g = int(tmp_l.shape[1] ** 0.5)
                weights = torch.nn.functional.interpolate(
                    w_2d, size=(target_g, target_g), mode="nearest"
                ).view(B_large, -1, 1)
            tmp_l = tmp_l * weights

        cls_l = self.cls_token[1].expand(B_large, -1, -1)
        tmp_l = torch.cat((cls_l, tmp_l), dim=1)
        tmp_l = tmp_l + self.pos_embed[1]
        tmp_l = self.pos_drop(tmp_l)

        # --- Blocs de Fusion (Cross-Attention) ---
        xs = [tmp_s, tmp_l]
        for blk in self.blocks:
            xs = blk(xs)

        xs = [self.norm[i](x) for i, x in enumerate(xs)]
        return [x[:, 0] for x in xs]

    def forward(self, x_small, x_large, weights=None, alpha=None):  # pyright: ignore[reportIncompatibleMethodOverride]
        B, N, C, H, W = x_small.shape

        # 1. Mixage de la Heatmap avec Alpha
        if weights is not None and weights.numel() > 0:
            # Sécurité : On s'adapte au nombre de canaux réels du .pt
            num_channels = weights.shape[1]

            if alpha is None:
                # Défaut intelligent : Fond (dernier canal) à 0.1, le reste à 1.0
                curr_alpha = torch.ones(num_channels).to(weights.device)
                curr_alpha[-1] = 0.1
            else:
                # On s'assure que alpha a la bonne taille pour ce fichier .pt
                curr_alpha = alpha[:num_channels].to(weights.device)

            # Somme pondérée des canaux : (B, C, 64, 64) * (1, C, 1, 1) -> (B, 64, 64)
            weights = (weights * curr_alpha.view(1, -1, 1, 1)).sum(dim=1)
            weights = weights.view(B, -1, 1)
        else:
            weights = None

        # 2. Aplatissement des patchs pour l'encodeur
        x_small = x_small.view(B * N, C, H, W)

        # 3. Features et Logits
        xs = self.forward_features(x_small, x_large, weights, num_patches=N)
        ce_logits = [self.head[i](x) for i, x in enumerate(xs)]  # pyright: ignore[reportIndexIssue]

        return torch.mean(torch.stack(ce_logits, dim=0), dim=0)
