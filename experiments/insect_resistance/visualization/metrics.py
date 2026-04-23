"""Shared metric helpers for unified scoring and plotting."""

import numpy as np


def robust_minmax(values, inverse=False, q_low=5.0, q_high=95.0, fill_value=0.5):
    """Quantile-clipped min-max normalization for stable cross-plot scoring."""
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, fill_value, dtype=float)
    valid = np.isfinite(arr)
    if valid.sum() < 2:
        return out

    lo = np.nanpercentile(arr[valid], q_low)
    hi = np.nanpercentile(arr[valid], q_high)
    if hi - lo <= 1e-12:
        return out

    scaled = (arr[valid] - lo) / (hi - lo)
    scaled = np.clip(scaled, 0.0, 1.0)
    out[valid] = 1.0 - scaled if inverse else scaled
    return out


def infer_gain_stabilizer(control_yields, quantile=10.0, floor=1.0):
    """Compute denominator stabilizer tau from control yields."""
    arr = np.asarray(control_yields, dtype=float)
    valid = np.isfinite(arr)
    if valid.sum() == 0:
        return floor
    tau = float(np.nanpercentile(arr[valid], quantile))
    return max(tau, floor)


def compute_gain_rate(control_yields, nocontrol_yields, tau):
    """Relative gain rate with stabilizer: (N - C) / (C + tau)."""
    c = np.asarray(control_yields, dtype=float)
    n = np.asarray(nocontrol_yields, dtype=float)
    return (n - c) / (c + float(tau))


def similarity_from_distance(distances, scale=None):
    """Convert non-negative distance to similarity in (0, 1] using exp mapping."""
    d = np.asarray(distances, dtype=float)
    out = np.full(d.shape, np.nan, dtype=float)
    valid = np.isfinite(d)
    if valid.sum() == 0:
        return out

    if scale is None:
        scale = float(np.nanmedian(d[valid]))
    scale = max(scale, 1e-10)
    out[valid] = np.exp(-d[valid] / scale)
    return out
