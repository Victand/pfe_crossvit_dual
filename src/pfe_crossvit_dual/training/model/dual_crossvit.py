import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.abspath(".."))
from CrossViT.models import crossvit
from pfe_crossvit_dual.training.model.recording_layers import (
    RecordingBlock,
    CrossAttentionBlock,
)


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim,
        patches,
        depth,
        num_heads,
        mlp_ratio,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=[0.0],
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        num_branches = len(dim)
        self.num_branches = num_branches
        # different branch could have different embedding size, the first one is the base
        self.blocks = nn.ModuleList()
        for d in range(num_branches):
            tmp = []
            for i in range(depth[d]):
                # NOTE Changed drop with proj_drop (was deprecated) - youenn B
                # NOTE Changed timm Block with our custom RecordingBlock to record Attention and perform Attention Rollout
                tmp.append(
                    RecordingBlock(
                        dim=dim[d],
                        num_heads=num_heads[d],
                        mlp_ratio=mlp_ratio[d],
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        proj_drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[i],
                        norm_layer=norm_layer,
                    )
                )
            if len(tmp) != 0:
                self.blocks.append(nn.Sequential(*tmp))

        '''if len(self.blocks) == 0:
            self.blocks = None'''

        self.projs = nn.ModuleList()
        for d in range(num_branches):
            if dim[d] == dim[(d + 1) % num_branches] and False:
                tmp = [nn.Identity()]
            else:
                tmp = [
                    norm_layer(dim[d]),
                    act_layer(),
                    nn.Linear(dim[d], dim[(d + 1) % num_branches]),
                ]
            self.projs.append(nn.Sequential(*tmp))

        self.fusion = nn.ModuleList()
        for d in range(num_branches):
            d_ = (d + 1) % num_branches
            nh = num_heads[d_]
            if depth[-1] == 0:  # backward capability:
                self.fusion.append(
                    CrossAttentionBlock(
                        dim=dim[d_],
                        num_heads=nh,
                        mlp_ratio=mlp_ratio[d],
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[-1],
                        norm_layer=norm_layer,
                        has_mlp=False,
                    )
                )
            else:
                tmp = []
                for _ in range(depth[-1]):
                    tmp.append(
                        CrossAttentionBlock(
                            dim=dim[d_],
                            num_heads=nh,
                            mlp_ratio=mlp_ratio[d],
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            drop=drop,
                            attn_drop=attn_drop,
                            drop_path=drop_path[-1],
                            norm_layer=norm_layer,
                            has_mlp=False,
                        )
                    )
                self.fusion.append(nn.Sequential(*tmp))

        self.revert_projs = nn.ModuleList()
        for d in range(num_branches):
            if dim[(d + 1) % num_branches] == dim[d] and False:
                tmp = [nn.Identity()]
            else:
                tmp = [
                    norm_layer(dim[(d + 1) % num_branches]),
                    act_layer(),
                    nn.Linear(dim[(d + 1) % num_branches], dim[d]),
                ]
            self.revert_projs.append(nn.Sequential(*tmp))

    def forward(self, x):
        outs_b = [block(x_) for x_, block in zip(x, self.blocks)]
        # only take the cls token out
        proj_cls_token = [proj(x[:, 0:1]) for x, proj in zip(outs_b, self.projs)]
        # cross attention
        outs = []
        for i in range(self.num_branches):
            tmp = torch.cat(
                (proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]),
                dim=1,
            )
            tmp = self.fusion[i](tmp)
            reverted_proj_cls_token = self.revert_projs[i](tmp[:, 0:1, ...])
            tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
            outs.append(tmp)
        return outs


class DualCrossVit(crossvit.VisionTransformer):
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
                num_patches,
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

    def forward_features(self, x_small, x_large, weights): # pyright: ignore[reportIncompatibleMethodOverride]
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

    def forward(self, x_small, x_large, weights=None, alpha=None): # pyright: ignore[reportIncompatibleMethodOverride]
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
        ce_logits = [self.head[i](x) for i, x in enumerate(xs)] # pyright: ignore[reportIndexIssue]

        if "getIndividualLogits" in self.kwargs:
            return ce_logits

        return torch.mean(torch.stack(ce_logits, dim=0), dim=0)
