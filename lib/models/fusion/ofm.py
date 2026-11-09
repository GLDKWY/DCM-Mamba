"""Selective Multi-scale Fusion Mamba (SMFM), matching paper Eqs. (8)–(13)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class _Conv1d(nn.Module):
    """Conv1d on the token axis. Input (B, L, C) -> (B, L, C)."""

    def __init__(self, dim, k=3):
        super().__init__()
        pad = (k - 1) // 2
        self.conv = nn.Conv1d(dim, dim, kernel_size=k, padding=pad)

    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class _ScaleAwareGating(nn.Module):
    """Compute per-layer importance scores e_i and softmax weights omega_i (Eqs. 8–9)."""

    def __init__(self, dim, hidden=256, drop=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats):
        scores = []
        for f in feats:
            g = self.norm(f).mean(dim=1)  # [B, C]
            scores.append(self.mlp(g))    # [B, 1]
        e = torch.cat(scores, dim=-1)     # [B, N]
        omega = torch.softmax(e, dim=-1)
        return omega


class SMFM(nn.Module):
    """
    Input:  list of f_cmim from selected encoder layers, each [B, L, C]
    Output: f_out [B, L, C]
    """

    def __init__(self, dim, num_layers=1, drop=0.0, mamba_kwargs=None, **kwargs):
        super().__init__()
        del kwargs
        mamba_kwargs = {} if mamba_kwargs is None else mamba_kwargs

        self.scale_gate = _ScaleAwareGating(dim, hidden=max(64, 4 * dim), drop=drop)

        # Three parallel scans: Forward / Backward / Dynamic, each Conv1d + SSM
        self.fwd_conv = _Conv1d(dim)
        self.bwd_conv = _Conv1d(dim)
        self.dyn_conv = _Conv1d(dim)
        self.ssm_forward = Mamba(d_model=dim, **mamba_kwargs)
        self.ssm_backward = Mamba(d_model=dim, **mamba_kwargs)
        self.ssm_dynamic = Mamba(d_model=dim, **mamba_kwargs)

        # Attention gating: q from mean||std of f_agg, k from mean of each branch
        self.W_q = nn.Linear(dim * 2, dim)
        self.W_k = nn.Linear(dim, dim)
        self.scale = math.sqrt(dim)

        self.norm = nn.LayerNorm(dim)

    def forward(self, feats):
        # Eqs. (8)–(9)
        omega = self.scale_gate(feats).unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1]
        stack = torch.stack(feats, dim=1)  # [B, N, L, C]
        f_agg = (omega * stack).sum(dim=1)  # [B, L, C]

        # Forward scan
        F1 = self.ssm_forward(self.fwd_conv(f_agg))

        # Backward scan
        rev = torch.flip(f_agg, dims=[1])
        F2 = torch.flip(self.ssm_backward(self.bwd_conv(rev)), dims=[1])

        # Dynamic scan: modulate topology by local token intensities, then Conv1d + SSM
        intensity = torch.sigmoid(f_agg.norm(dim=-1, keepdim=True))
        F3 = self.ssm_dynamic(self.dyn_conv(f_agg * intensity))

        # Eqs. (10)–(12)
        q = self.W_q(torch.cat([f_agg.mean(dim=1), f_agg.std(dim=1, unbiased=False)], dim=-1))
        keys = torch.stack(
            [
                self.W_k(F1.mean(dim=1)),
                self.W_k(F2.mean(dim=1)),
                self.W_k(F3.mean(dim=1)),
            ],
            dim=1,
        )  # [B, 3, C]
        alpha = torch.softmax((q.unsqueeze(1) * keys).sum(-1) / self.scale, dim=-1)  # [B, 3]

        fused = (
            alpha[:, 0, None, None] * F1
            + alpha[:, 1, None, None] * F2
            + alpha[:, 2, None, None] * F3
        )

        # Eq. (13)
        f_out = self.norm(fused) * F.silu(f_agg)
        return f_out


# Keep the old name so existing imports continue to work.
OFM = SMFM
