import torch
import torch.nn as nn

from timm.layers.drop import DropPath
from timm.layers.mlp import Mlp
import sys
import os

sys.path.append(os.path.abspath(".."))

from CrossViT.models import crossvit


class RecordingAttention(nn.Module):
    """
    Custom Atention module which stores the attention heads to perform attention rollout.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # We store the the attention heads in a variable
        self.last_attn_map = None

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Storing
        # After the prediction, detaching to cpu to preserve GPU VRAM, before Dropout to capture model's reel intention
        # NOTE Attention : verifier si detach n'empeche pas le optimize.step() de prendre en compte notre loss custom pour l'iou
        self.last_attn_map = attn

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RecordingBlock(nn.Module):
    """
    Custom block which stores attention heads to perform attention rollout.
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        proj_drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = RecordingAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class RecordingCrossAttention(nn.Module):
    """
    custom Cross Atention module which stores the attention heads to perform attention rollout.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # We store the the attention heads in a variable
        self.last_attn_map = None

    def forward(self, x):
        B, N, C = x.shape
        # Attention : CrossViT projette x différemment pour q, k, v
        q = (
            self.wq(x[:, 0:1, ...])
            .reshape(B, 1, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.wk(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.wv(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Storing
        self.last_attn_map = attn.detach().cpu()

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):
    """
    Custom block which stores cross attention to perform attention rollout.
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        has_mlp=True,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # custom Cross Atention module which stores the attention heads
        self.attn = RecordingCrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )

    def forward(self, x):
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


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
        drop_path=0.0,
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

        if len(self.blocks) == 0:
            self.blocks = None

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

    def forward_features(self, x_small, x_large, weights, num_patches):
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

    def forward(self, x_small, x_large, weights=None, alpha=None):
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
        ce_logits = [self.head[i](x) for i, x in enumerate(xs)]

        return torch.mean(torch.stack(ce_logits, dim=0), dim=0)
