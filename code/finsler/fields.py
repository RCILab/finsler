"""
Wind fields and the three metric baselines used throughout the experiments.

    finsler_field(W)      Randers metric of Zermelo navigation in wind W (asymmetric)
    riemannian_part(W)    its Riemannian part a(x) only (anisotropic but symmetric): cannot see the sign of W
    euclidean_field()     flat metric

All three are `RandersField`s, so exp/log/geodesic code is shared.
"""
from __future__ import annotations

import numpy as np

from .randers import RandersField, zermelo_to_randers


def channel_wind(beta=0.7, width=0.8, direction=(-1.0, 0.0)):
    """W(x) = beta * exp(-x2^2/width^2) * direction : a head-wind channel along the x1 axis (curl != 0)."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)

    def W(x):
        x = np.asarray(x, float)
        s = beta * np.exp(-(x[..., 1] ** 2) / width**2)
        return s[..., None] * d

    return W


def vortex_wind(beta=0.8, r0=0.8, center=(0.0, 0.0)):
    """Rankine vortex: |W| = beta r/r0 inside r0, beta r0/r outside; counter-clockwise."""
    c = np.asarray(center, float)

    def W(x):
        x = np.asarray(x, float) - c
        r = np.linalg.norm(x, axis=-1)
        mag = np.where(r < r0, beta * r / r0, beta * r0 / np.maximum(r, 1e-9))
        tang = np.stack([-x[..., 1], x[..., 0]], -1) / np.maximum(r, 1e-9)[..., None]
        return mag[..., None] * tang

    return W


def finsler_field(W) -> RandersField:
    return RandersField(W)


def riemannian_part(W) -> RandersField:
    """The Riemannian metric a(x) of the Randers metric with wind W, as a zero-wind RandersField."""

    def h(x):
        x = np.asarray(x, float)
        eye = np.broadcast_to(np.eye(2), x.shape + (2,))
        A, _, _ = zermelo_to_randers(eye, W(x))
        return A

    return RandersField(lambda x: np.zeros_like(np.asarray(x, float)), h)


def euclidean_field() -> RandersField:
    return RandersField(lambda x: np.zeros_like(np.asarray(x, float)))
