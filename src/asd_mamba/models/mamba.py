import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed * self.weight


class MambaBlock(nn.Module):
    """Self-contained Mamba-style selective state-space block for short fMRI sequences."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.d_conv = d_conv
        self.dt_rank = dt_rank or math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).repeat(self.d_inner, 1).float()))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xz = self.in_proj(x)
        x_branch, z_branch = xz.chunk(2, dim=-1)

        x_branch = self.conv1d(x_branch.transpose(1, 2))[:, :, : x.shape[1]].transpose(1, 2)
        x_branch = F.silu(x_branch)

        projected = self.x_proj(x_branch)
        dt, b_param, c_param = torch.split(
            projected,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        dt = F.softplus(self.dt_proj(dt))
        y = self.selective_scan(x_branch, dt, b_param, c_param)
        y = y * F.silu(z_branch)
        return self.dropout(self.out_proj(y))

    def selective_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        b_param: torch.Tensor,
        c_param: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        A = -torch.exp(self.A_log.float()).to(x.dtype)
        D = self.D.to(x.dtype)
        state = x.new_zeros(batch_size, self.d_inner, self.d_state)
        outputs = []

        for step in range(seq_len):
            dt_step = dt[:, step]
            x_step = x[:, step]
            b_step = b_param[:, step]
            c_step = c_param[:, step]

            delta_a = torch.exp(dt_step.unsqueeze(-1) * A.unsqueeze(0))
            delta_b_x = dt_step.unsqueeze(-1) * b_step.unsqueeze(1) * x_step.unsqueeze(-1)
            state = delta_a * state + delta_b_x
            y_step = (state * c_step.unsqueeze(1)).sum(dim=-1) + D * x_step
            outputs.append(y_step)

        return torch.stack(outputs, dim=1)


class MambaEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.block = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(self.norm(x))


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

