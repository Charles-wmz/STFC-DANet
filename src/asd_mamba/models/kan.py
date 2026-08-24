import torch
from torch import nn
from torch.nn import functional as F


class SplineKANLayer(nn.Module):
    """Lightweight KAN-style layer using learnable radial spline bases.

    This keeps the KAN idea of learnable univariate nonlinear functions per
    input-output pair while avoiding external dependencies.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 8,
        grid_min: float = -3.0,
        grid_max: float = 3.0,
        base_activation: nn.Module = nn.SiLU(),
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.base_activation = base_activation

        grid = torch.linspace(grid_min, grid_max, grid_size)
        self.register_buffer("grid", grid)
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.log_width = nn.Parameter(torch.zeros(in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=5**0.5)
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(self.base_activation(x), self.base_weight, self.bias)
        width = F.softplus(self.log_width).view(1, self.in_features, 1) + 1e-3
        basis = torch.exp(-((x.unsqueeze(-1) - self.grid.view(1, 1, -1)) / width).pow(2))
        basis = basis / (basis.sum(dim=-1, keepdim=True) + 1e-6)
        spline = torch.einsum("big,oig->bo", basis, self.spline_weight)
        return base + spline


class KANBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layer = SplineKANLayer(
            in_features=in_features,
            out_features=out_features,
            grid_size=grid_size,
        )
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        x = self.norm(x)
        x = F.gelu(x)
        return self.dropout(x)

