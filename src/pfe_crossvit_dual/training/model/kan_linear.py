"""
kan_linear.py — Implémentation autonome de KANLinear (B-splines)
=================================================================
Source : https://github.com/Blealtan/efficient-kan (MIT License)
Embarquée directement pour éviter les problèmes d'installation PyPI.

Usage :
    from pfe_crossvit_dual.training.model.kan_linear import KANLinear
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    """
    Couche linéaire KAN basée sur des B-splines.

    Paramètres
    ----------
    in_features : int
    out_features : int
    grid_size : int
        Nombre de noeuds de la grille B-spline. Plus grand = plus expressif.
        Valeurs typiques : 3 (petit dataset) à 8 (grand dataset).
    spline_order : int
        Ordre des B-splines (3 = cubique, standard).
    scale_noise : float
        Bruit d'initialisation sur les poids spline.
    scale_base : float
        Échelle de la partie linéaire résiduelle (SiLU).
    scale_spline : float
        Échelle des poids spline à l'initialisation.
    enable_standalone_scale_spline : bool
        Active un paramètre de scale appris par activation.
    base_activation : nn.Module
        Activation de la branche linéaire résiduelle.
    grid_eps : float
        Mélange entre grille uniforme et grille adaptée aux données
        lors du update (0 = tout adaptatif, 1 = tout uniforme).
    grid_range : tuple
        Plage initiale de la grille.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation: type = nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: tuple = (-1, 1),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]
        ).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5
            ) * self.scale_noise / self.grid_size
            self.spline_weight.data.copy_(
                self.scale_spline
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcule les bases B-splines pour l'entrée x.
        x : (batch, in_features)
        retourne : (batch, in_features, grid_size + spline_order)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = self.grid  # (in_features, grid_size + 2*spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]
                + (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * bases[:, :, 1:]
            )
        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calcule les coefficients B-splines interpolant les points (x, y).
        x : (batch, in_features)
        y : (batch, in_features, out_features)
        retourne : (out_features, in_features, grid_size + spline_order)
        """
        A = self.b_splines(x).transpose(0, 1)  # (in, batch, grid+order)
        B = y.transpose(0, 1)                   # (in, batch, out)
        solution = torch.linalg.lstsq(A, B).solution  # (in, grid+order, out)
        result = solution.permute(2, 0, 1)             # (out, in, grid+order)
        assert result.size() == (self.out_features, self.in_features, self.grid_size + self.spline_order)
        return result.contiguous()

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_out = F.linear(self.base_activation(x), self.base_weight)
        spline_out = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        out = base_out + spline_out
        return out.reshape(*original_shape[:-1], self.out_features)

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01):
        """
        Met à jour la grille B-spline en fonction de la distribution de x.
        À appeler optionnellement après quelques epochs pour adapter les noeuds.
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x).permute(1, 0, 2)  # (in, batch, grid+order)
        splines = splines.reshape(self.in_features, batch, -1)
        orig_coeff = self.scaled_spline_weight.permute(1, 2, 0)  # (in, grid+order, out)
        orig_coeff = orig_coeff.reshape(self.in_features, -1, self.out_features)
        unreduced_spline_out = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_out = unreduced_spline_out.permute(1, 0, 2)  # (batch, in, out)

        x_sorted = torch.sort(x, dim=0).values
        grid_adaptive = x_sorted[
            torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(self.grid_size + 1, dtype=x.dtype, device=x.device).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.cat(
            [
                grid[:1] - uniform_step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:] + uniform_step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_out))

    def regularization_loss(self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0) -> torch.Tensor:
        """
        Perte de régularisation L1 sur les activations spline.
        Utile pour limiter l'overfitting sur petit dataset.

        Exemple d'utilisation dans la boucle d'entraînement :
            loss = criterion(logits, targets)
            for module in model.modules():
                if isinstance(module, KANLinear):
                    loss += 1e-4 * module.regularization_loss()
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )