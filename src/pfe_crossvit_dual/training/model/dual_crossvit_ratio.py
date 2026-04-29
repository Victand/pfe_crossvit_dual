import torch
import torch.nn as nn

from CrossViT.models import crossvit
from pfe_crossvit_dual.training.model.recording_layers import MultiScaleBlock


class DualCrossVitRatio(crossvit.VisionTransformer):
    """
    Custom CrossVit class to accept two images in input : Original/detoured herb image.
    Inherits from the standard VisionTransformer (CrossViT implementation).
    Adds custom Blocks to record Attentions/cross-attention to perform Attention rollout.
    """

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=(8, 16),
        in_chans=3,
        num_classes=2,
        embed_dim=(192, 384),
        depth=([1, 3, 1], [1, 3, 1], [1, 3, 1]),
        num_heads=(6, 12),
        mlp_ratio=(2.0, 2.0, 4.0),
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        hybrid_backbone=None,
        norm_layer=nn.LayerNorm,
        multi_conv=False,
        **kwargs,
    ):
        super().__init__(
            img_size,
            patch_size,
            in_chans,
            num_classes,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            hybrid_backbone,
            norm_layer,
            multi_conv,
            **kwargs,
        )
        self.kwargs = kwargs

        self.patch_size = patch_size

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

    def forward_features(self, x_small, x_large, weights):  # pyright: ignore[reportIncompatibleMethodOverride]
        # [Branch 0 (Small), Branch 1 (Large)]
        xs_inputs = [x_small, x_large]
        B, C, H, W = x_small.shape
        xs = []

        for i in range(self.num_branches):
            current_img = xs_inputs[i]
            expected_size = self.img_size[i]

            # Interpolation de l'image si besoin
            if (
                current_img.shape[-2] != expected_size
                or current_img.shape[-1] != expected_size
            ):
                x_ = torch.nn.functional.interpolate(
                    current_img,
                    size=(expected_size, expected_size),
                    mode="bicubic",
                    align_corners=False,
                )
            else:
                x_ = current_img

            tmp = self.patch_embed[i](x_)

            # --- GESTION DES POIDS MULTI-CANAUX ---
            if weights is not None:
                # weights arrive ici en (B, 196, 1) après avoir été mixé dans le forward
                if weights.shape[1] != tmp.shape[1]:
                    B_w, N_w, C_w = weights.shape
                    grid_size_w = int(N_w**0.5)
                    w_2d = weights.view(B_w, C_w, grid_size_w, grid_size_w)
                    target_grid_size = int(tmp.shape[1] ** 0.5)

                    w_resized = torch.nn.functional.interpolate(
                        w_2d, size=(target_grid_size, target_grid_size), mode="nearest"
                    )
                    current_weights = w_resized.view(B_w, -1, C_w)
                else:
                    current_weights = weights

                tmp = tmp * current_weights
            # --------------------------------------

            cls_tokens = self.cls_token[i].expand(B, -1, -1)
            tmp = torch.cat((cls_tokens, tmp), dim=1)

            # Gestion du Pos Embed (ton code actuel est conservé)
            pos_embed = self.pos_embed[i]
            if tmp.shape[1] != pos_embed.shape[1]:
                pos_embed_cls = pos_embed[:, 0:1]
                pos_embed_patches = pos_embed[:, 1:].transpose(1, 2)
                pos_embed_patches = torch.nn.functional.interpolate(
                    pos_embed_patches,
                    size=tmp.shape[1] - 1,
                    mode="linear",
                    align_corners=False,
                )
                pos_embed = torch.cat(
                    (pos_embed_cls, pos_embed_patches.transpose(1, 2)), dim=1
                )

            tmp = tmp + pos_embed
            tmp = self.pos_drop(tmp)
            xs.append(tmp)

        for blk in self.blocks:
            xs = blk(xs)

        xs = [self.norm[i](x) for i, x in enumerate(xs)]
        out = [x[:, 0] for x in xs]
        return out

    def forward(self, x_small, x_large, weights=None, alpha=None):  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        alpha: Tenseur de poids [Fond, Fleur, Feuille, Tige]
        Ex: torch.tensor([0.1, 1.0, 1.0, 5.0])
        """
        # Si on a des poids multi-canaux (B, 4, 14, 14), on les mixe
        if weights is not None and weights.numel() > 0:
            if weights.dim() == 4:  # Format (B, 4, 14, 14)
                if alpha is None:
                    # Stratégie par défaut si on oublie alpha
                    alpha = torch.tensor([0.1, 1.0, 1.0, 1.0]).to(weights.device)

                # Mixage des canaux : somme pondérée
                # On multiplie chaque canal (Fleur, Tige...) par son importance alpha
                weights = (weights * alpha.view(1, -1, 1, 1)).sum(dim=1)
                # On aplatit pour le ViT : (B, 196, 1)
                weights = weights.view(weights.shape[0], -1, 1)
        else:
            weights = None

        xs = self.forward_features(x_small, x_large, weights)
        ce_logits = [self.head[i](x) for i, x in enumerate(xs)]  # pyright: ignore[reportIndexIssue]

        if "getIndividualLogits" in self.kwargs:
            return ce_logits

        return torch.mean(torch.stack(ce_logits, dim=0), dim=0)
