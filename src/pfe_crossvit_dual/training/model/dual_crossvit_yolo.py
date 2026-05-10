"""
DualCrossVitYolo — intégration KAN modulaire
============================================
Paramètre `kan_mode` à l'initialisation :

    "none"          → architecture originale (baseline Linear heads)
    "head"          → Option 1 : KANLinear remplace le head (recommandé, faible overfitting)
    "bottleneck"    → Option 2 : couche KAN résiduelle entre norm et head
    "ffn"           → Option 3 : FFN du dernier MultiScaleBlock remplacé par KAN (risqué)
    "head+bottleneck" → combinaison options 1 + 2

Dépendances :
    pip install efficient-kan
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from CrossViT.models import crossvit
from pfe_crossvit_dual.training.model.recording_layers import MultiScaleBlock

from pfe_crossvit_dual.training.model.kan_linear import KANLinear



# ---------------------------------------------------------------------------
# Type alias pour les modes KAN
# ---------------------------------------------------------------------------
KanMode = Literal["none", "head", "bottleneck", "head+bottleneck", "ffn"]


# ---------------------------------------------------------------------------
# Modules KAN auxiliaires
# ---------------------------------------------------------------------------

class KANBottleneck(nn.Module):
    """
    Option 2 — Couche KAN résiduelle insérée après le norm, avant le head.

    Architecture : LayerNorm(x + proj(KAN(x)))
    La connexion résiduelle + la petite dimension cachée limitent l'overfitting.
    """

    def __init__(self, dim: int, hidden_dim: int = 64, grid_size: int = 3):
        super().__init__()
        assert _KAN_AVAILABLE, "efficient-kan non installé : pip install efficient-kan"
        self.kan = KANLinear(dim, hidden_dim, grid_size=grid_size)
        self.proj = nn.Linear(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.proj(self.kan(x)))


def _replace_ffn_with_kan(block: nn.Module, embed_dim: tuple, grid_size: int = 3) -> None:
    """
    Option 3 — Remplace les MLP internes d'un MultiScaleBlock par des KANLinear.

    Parcourt récursivement le bloc à la recherche de sous-modules nn.Sequential
    contenant des nn.Linear consécutifs (pattern FFN standard) et les remplace.
    Compatible avec l'implémentation CrossViT standard (deux Linear séparés par
    une activation dans chaque branche du bloc).
    """
    assert _KAN_AVAILABLE, "efficient-kan non installé : pip install efficient-kan"

    for name, module in list(block.named_modules()):
        # Cible les sous-modules qui sont eux-mêmes des Sequential ou des Mlp
        # CrossViT utilise une classe Mlp avec fc1/act/drop/fc2
        if hasattr(module, "fc1") and hasattr(module, "fc2"):
            in_features = module.fc1.in_features
            hidden_features = module.fc1.out_features
            out_features = module.fc2.out_features
            kan_ffn = nn.Sequential(
                KANLinear(in_features, hidden_features, grid_size=grid_size),
                nn.GELU(),
                nn.Dropout(p=0.0),
                KANLinear(hidden_features, out_features, grid_size=grid_size),
            )
            # Remplace le module in-place via setattr sur le parent
            parent = block
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], kan_ffn)


# ---------------------------------------------------------------------------
# Modèle principal
# ---------------------------------------------------------------------------

class DualCrossVitYolo(crossvit.VisionTransformer):
    """
    CrossVit dual-stream avec intégration KAN sélectionnable.

    Paramètres
    ----------
    kan_mode : str
        Variante KAN à activer (voir module docstring).
    kan_grid_size : int
        Taille de la grille des splines KAN. Valeurs typiques : 3–8.
        Plus grand = plus expressif mais plus de paramètres.
    kan_bottleneck_dim : int
        Dimension cachée du KANBottleneck (option 2 / "bottleneck").
    kan_ffn_last_only : bool
        Si True (option 3), ne remplace le FFN que dans le *dernier* MultiScaleBlock.
        Recommandé pour limiter l'overfitting sur petit dataset.
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
        # ---- paramètres KAN ----
        kan_mode: KanMode = "none",
        kan_grid_size: int = 5,
        kan_bottleneck_dim: int = 64,
        kan_ffn_last_only: bool = True,
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
        self.kan_mode = kan_mode

        num_patches = crossvit._compute_num_patches(img_size, patch_size)
        self.patch_size = patch_size
        self.default_weights = [torch.ones((num_patches[i], 1)) for i in range(len(self.img_size))]

        # ----------------------------------------------------------------
        # Reconstruction des MultiScaleBlocks (identique à l'original)
        # ----------------------------------------------------------------
        total_depth = sum([sum(x[-2:]) for x in depth])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
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

        # ----------------------------------------------------------------
        # Option 3 : remplacement FFN dans MultiScaleBlock(s)
        # ----------------------------------------------------------------
        if "ffn" in kan_mode:
            _validate_kan()
            if kan_ffn_last_only:
                _replace_ffn_with_kan(self.blocks[-1], embed_dim, grid_size=kan_grid_size)
            else:
                for blk in self.blocks:
                    _replace_ffn_with_kan(blk, embed_dim, grid_size=kan_grid_size)

        # ----------------------------------------------------------------
        # Option 2 : bottleneck KAN entre norm et head
        # ----------------------------------------------------------------
        if "bottleneck" in kan_mode:
            _validate_kan()
            self.kan_bottlenecks = nn.ModuleList([
                KANBottleneck(
                    dim=embed_dim[i],
                    hidden_dim=kan_bottleneck_dim,
                    grid_size=kan_grid_size,
                )
                for i in range(len(embed_dim))
            ])
        else:
            self.kan_bottlenecks = None

        # ----------------------------------------------------------------
        # Option 1 : remplacement du head par KANLinear
        # ----------------------------------------------------------------
        if "head" in kan_mode:
            _validate_kan()
            # Récupère num_classes depuis le head existant (déjà construit par super())
            num_classes_list = [h.out_features for h in self.head]
            self.head = nn.ModuleList([
                KANLinear(
                    embed_dim[i],
                    num_classes_list[i],
                    grid_size=kan_grid_size,
                )
                for i in range(len(embed_dim))
            ])

    # ------------------------------------------------------------------
    # Freeze / unfreeze (identique à l'original)
    # ------------------------------------------------------------------

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_stage(self, stage: int):
        if stage >= 0:
            for head in self.head:
                for p in head.parameters():
                    p.requires_grad = True
        if stage >= 1:
            for n in self.norm:
                for p in n.parameters():
                    p.requires_grad = True
            if self.kan_bottlenecks is not None:
                for bn in self.kan_bottlenecks:
                    for p in bn.parameters():
                        p.requires_grad = True
        if stage >= 2:
            for blk in self.blocks:
                for p in blk.parameters():
                    p.requires_grad = True
        if stage >= 3:
            for p in self.cls_token:
                p.requires_grad = True
            for p in self.pos_embed:
                p.requires_grad = True
        if stage >= 4:
            for pe in self.patch_embed:
                for p in pe.parameters():
                    p.requires_grad = True

    def set_unfreeze_stage(self, stage: int):
        self.freeze_all()
        self.unfreeze_stage(stage)

    # ------------------------------------------------------------------
    # Forward features (identique à l'original)
    # ------------------------------------------------------------------

    def forward_features(self, x_small, x_large, weights, num_patches):  # pyright: ignore
        B_large = x_large.shape[0]

        tmp_s = self.patch_embed[0](x_small)
        cls_s = self.cls_token[0].expand(x_small.shape[0], -1, -1)
        tmp_s = torch.cat((cls_s, tmp_s), dim=1)

        pos_embed_s = self.pos_embed[0]
        if tmp_s.shape[1] != pos_embed_s.shape[1]:
            pos_embed_tok = pos_embed_s[:, 0:1]
            pos_embed_grid = pos_embed_s[:, 1:]
            gs_old = int(pos_embed_grid.shape[1] ** 0.5)
            gs_new = int((tmp_s.shape[1] - 1) ** 0.5)
            pos_embed_grid = pos_embed_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
            pos_embed_grid = torch.nn.functional.interpolate(
                pos_embed_grid, size=(gs_new, gs_new), mode="bicubic", align_corners=False,
            )
            pos_embed_grid = pos_embed_grid.permute(0, 2, 3, 1).reshape(1, -1, pos_embed_s.shape[-1])
            pos_embed_s = torch.cat((pos_embed_tok, pos_embed_grid), dim=1)

        tmp_s = tmp_s + pos_embed_s
        tmp_s = self.pos_drop(tmp_s)
        tmp_s = tmp_s.view(B_large, num_patches, tmp_s.shape[1], tmp_s.shape[2]).mean(dim=1)

        tmp_l = self.patch_embed[1](x_large)
        if weights is not None:
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

        xs = [tmp_s, tmp_l]
        for blk in self.blocks:
            xs = blk(xs)

        xs = [self.norm[i](x) for i, x in enumerate(xs)]
        return [x[:, 0] for x in xs]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x_small, x_large, weights, alpha):  # pyright: ignore
        B, N, C, H, W = x_small.shape

        if weights is not None:
            if weights.shape[1] == 1:
                weights = weights.view(B, -1, 1)
            else:
                weights = (weights * alpha.view(1, -1, 1, 1)).sum(dim=1)
                weights = weights.view(B, -1, 1)

        x_small = x_small.view(B * N, C, H, W)

        # Features
        xs = self.forward_features(x_small, x_large, weights, num_patches=N)

        # Option 2 : bottleneck KAN après norm
        if self.kan_bottlenecks is not None:
            xs = [self.kan_bottlenecks[i](x) for i, x in enumerate(xs)]

        # Head (Linear ou KANLinear selon kan_mode)
        ce_logits = [self.head[i](x) for i, x in enumerate(xs)]  # pyright: ignore

        return torch.mean(torch.stack(ce_logits, dim=0), dim=0)

    # ------------------------------------------------------------------
    # Utilitaire : résumé du mode KAN actif
    # ------------------------------------------------------------------

    def kan_summary(self) -> str:
        mode = self.kan_mode
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lines = [
            f"KAN mode      : {mode!r}",
            f"Total params  : {total:,}",
            f"Trainable     : {trainable:,}",
        ]
        if self.kan_bottlenecks is not None:
            bn_params = sum(p.numel() for bn in self.kan_bottlenecks for p in bn.parameters())
            lines.append(f"  Bottleneck  : {bn_params:,} params")
        if "head" in mode:
            head_params = sum(p.numel() for h in self.head for p in h.parameters())
            lines.append(f"  KAN heads   : {head_params:,} params")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

def _validate_kan():
    if not _KAN_AVAILABLE:
        raise ImportError(
            "Le paquet efficient-kan est requis pour les modes KAN.\n"
            "Installez-le avec : pip install efficient-kan"
        )


# ---------------------------------------------------------------------------
# Factory : crée facilement une variante pour les expériences
# ---------------------------------------------------------------------------

def build_model(
    kan_mode: KanMode = "none",
    num_classes: int = 10,
    kan_grid_size: int = 5,
    kan_bottleneck_dim: int = 64,
    kan_ffn_last_only: bool = True,
    **model_kwargs,
) -> DualCrossVitYolo:
    """
    Instancie une variante DualCrossVitYolo.

    Exemple
    -------
    >>> baseline   = build_model("none",       num_classes=10)
    >>> kan_head   = build_model("head",       num_classes=10, kan_grid_size=5)
    >>> kan_bn     = build_model("bottleneck", num_classes=10, kan_bottleneck_dim=64)
    >>> kan_full   = build_model("head+bottleneck", num_classes=10)
    >>> kan_ffn    = build_model("ffn",        num_classes=10, kan_ffn_last_only=True)
    """
    return DualCrossVitYolo(
        num_classes=num_classes,
        kan_mode=kan_mode,
        kan_grid_size=kan_grid_size,
        kan_bottleneck_dim=kan_bottleneck_dim,
        kan_ffn_last_only=kan_ffn_last_only,
        **model_kwargs,
    )


# ---------------------------------------------------------------------------
# Script de comparaison rapide (python dual_crossvit_kan.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    MODES: list[KanMode] = ["none", "head", "bottleneck", "head+bottleneck", "ffn"]
    NUM_CLASSES = 10

    print("=" * 60)
    print("Comparaison des variantes KAN — DualCrossVitYolo")
    print("=" * 60)

    for mode in MODES:
        try:
            model = build_model(mode, num_classes=NUM_CLASSES, kan_ffn_last_only=True)
            print(f"\n{model.kan_summary()}")
            print("-" * 40)
        except ImportError as e:
            print(f"\n[{mode}] Skipped : {e}")