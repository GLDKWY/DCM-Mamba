"""Cross-Modal Interaction Mamba (CMIM), matching paper Eqs. (3)–(7)."""
import torch
import torch.nn as nn
from mamba_ssm import Mamba


class CMIM(nn.Module):
    """
    Inputs:
      f_rgb, f_tir: [B, L, C]
    Returns:
      f_rgb', f_tir', f_cmim  (Eqs. 5–7)
    """

    def __init__(self, dim, hidden_mult=2, drop=0.0, num_heads=4, mamba_kwargs=None, **kwargs):
        super().__init__()
        del hidden_mult, num_heads, kwargs
        mamba_kwargs = {} if mamba_kwargs is None else mamba_kwargs

        # W_c, W_d: learnable alignment projections
        self.W_c = nn.Linear(dim, dim)
        self.W_d = nn.Linear(dim, dim)
        # phi_sum / phi_diff
        self.phi_sum = nn.GELU()
        self.phi_diff = nn.GELU()

        # Parameter-shared Mamba for consensus / discrepancy streams
        self.mamba = Mamba(d_model=dim, **mamba_kwargs)

        # G(x) = tanh(omega x + b)
        self.gate = nn.Linear(dim, dim)

        # Channel fusion operator P on [f'_rgb, f'_tir]
        self.P = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Dropout(drop),
        )

    def _G(self, x):
        return torch.tanh(self.gate(x))

    def forward(self, f_rgb, f_tir):
        # Eq. (3)
        C = self.phi_sum(self.W_c(f_rgb + f_tir))
        D = self.phi_diff(self.W_d(f_rgb - f_tir))

        # Eq. (4): shared Mamba on both streams
        cd = torch.cat([C, D], dim=0)
        cd_hat = self.mamba(cd)
        C_hat, D_hat = cd_hat.chunk(2, dim=0)

        # Eqs. (5)–(6)
        G_C = self._G(C_hat)
        G_D = self._G(D_hat)
        f_rgb_p = f_rgb + G_D * f_tir + G_C * f_rgb
        f_tir_p = f_tir + G_D * f_rgb + G_C * f_tir

        # Eq. (7)
        f_cmim = self.P(torch.cat([f_rgb_p, f_tir_p], dim=-1))
        return f_rgb_p, f_tir_p, f_cmim


# Keep the old name so existing imports continue to work.
DFM = CMIM
