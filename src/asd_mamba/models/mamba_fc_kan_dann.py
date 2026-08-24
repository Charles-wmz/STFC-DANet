import torch
from torch import nn
from torch.autograd import Function

from asd_mamba.models.kan import KANBlock
from asd_mamba.models.mamba import MambaEncoderLayer, RMSNorm


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_value: float) -> torch.Tensor:
        ctx.lambda_value = lambda_value
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_value * grad_output, None


class GradientReversal(nn.Module):
    def forward(self, x: torch.Tensor, lambda_value: float) -> torch.Tensor:
        return GradientReversalFunction.apply(x, lambda_value)


class MambaFcKanDann(nn.Module):
    """Mamba time-series branch + FC-KAN branch with DANN domain adversarial head."""

    def __init__(
        self,
        num_rois: int = 200,
        fc_dim: int = 19900,
        num_classes: int = 2,
        num_domains: int = 19,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        branch_dim: int = 64,
        fused_dim: int = 128,
        kan_grid_size: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.time_proj = nn.Linear(num_rois, d_model)
        self.time_layers = nn.ModuleList(
            [
                MambaEncoderLayer(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.time_norm = RMSNorm(d_model)
        self.time_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, branch_dim),
            nn.GELU(),
        )
        self.fc_reduce = nn.Sequential(
            nn.Linear(fc_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fc_kan = nn.Sequential(
            KANBlock(
                in_features=512,
                out_features=256,
                grid_size=kan_grid_size,
                dropout=dropout,
            ),
            KANBlock(
                in_features=256,
                out_features=128,
                grid_size=kan_grid_size,
                dropout=dropout,
            ),
            nn.Linear(128, branch_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(branch_dim * 2, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.label_classifier = nn.Sequential(
            KANBlock(
                in_features=fused_dim,
                out_features=fused_dim // 2,
                grid_size=kan_grid_size,
                dropout=dropout,
            ),
            nn.Linear(fused_dim // 2, num_classes),
        )
        self.domain_classifier = nn.Sequential(
            KANBlock(
                in_features=fused_dim,
                out_features=fused_dim // 2,
                grid_size=kan_grid_size,
                dropout=dropout,
            ),
            nn.Linear(fused_dim // 2, num_domains),
        )
        self.grl = GradientReversal()

    def extract_features(self, x_time: torch.Tensor, x_fc: torch.Tensor) -> torch.Tensor:
        x = self.time_proj(x_time)
        for layer in self.time_layers:
            x = layer(x)
        x = self.time_norm(x)
        time_feature = self.time_head(x.mean(dim=1))
        fc_feature = self.fc_kan(self.fc_reduce(x_fc))
        return self.fusion(torch.cat([time_feature, fc_feature], dim=-1))

    def forward(self, x_time: torch.Tensor, x_fc: torch.Tensor, lambda_value: float = 0.0):
        features = self.extract_features(x_time, x_fc)
        label_logits = self.label_classifier(features)
        reversed_features = self.grl(features, lambda_value)
        domain_logits = self.domain_classifier(reversed_features)
        return label_logits, domain_logits, features

