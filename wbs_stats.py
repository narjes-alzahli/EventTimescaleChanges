"""
Statistical helpers 
====================

BH-FDR utilities and simple two-tailed z-test p-values against permutation
null distributions. Behavior matches the earlier utilities; only naming and
docstrings are improved.

"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm

__all__ = ["bh_fdr", "two_tailed_p_value", "perform_bh_fdr"]


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """
    Benjamini–Hochberg FDR correction.

    Parameters
    ----------
    p : ndarray, shape (m,)
        Vector of p-values.

    Returns
    -------
    ndarray, shape (m,)
        BH-adjusted q-values aligned to the original order.
    """
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order.argsort()] = q
    return out


def two_tailed_p_value(true_value: float, nulls: np.ndarray) -> float:
    """
    Two-tailed z-test from a permutation null.

    Parameters
    ----------
    true_value : float
        Observed statistic.
    nulls : ndarray
        Null distribution values.

    Returns
    -------
    float
        Two-tailed p-value (NaN if nulls are empty/degenerate).
    """
    nulls = nulls[np.isfinite(nulls)]
    if nulls.size == 0:
        return float("nan")
    m = nulls.mean()
    s = nulls.std(ddof=1)
    if s == 0:
        return float("nan")
    z = (true_value - m) / s
    return 2 * norm.sf(abs(z))

def perform_bh_fdr(p_vals: dict[str, float], alpha: float = 0.05) -> list[str]:
    """
    Return keys that survive BH-FDR.

    Parameters
    ----------
    p_vals : dict
        Mapping key -> p-value.
    alpha : float
        FDR threshold.

    Returns
    -------
    list of str
        Keys with q < alpha.
    """
    keys = np.array(list(p_vals.keys()), dtype=object)
    ps = np.array(list(p_vals.values()), dtype=float)
    finite = np.isfinite(ps)
    if not finite.any():
        return []
    q = np.full_like(ps, np.nan, dtype=float)
    q[finite] = bh_fdr(ps[finite])
    keep = keys[finite][q[finite] < alpha]
    return list(map(str, keep))
