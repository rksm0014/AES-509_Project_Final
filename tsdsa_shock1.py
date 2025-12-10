#!/usr/bin/env python
# coding: utf-8

# In[2]:


from __future__ import annotations

"""
tsdsa_shock1
============

Module for analyzing energetic particle profiles around an
interplanetary shock using TSDSA-like upstream Mittag–Leffler
profiles and a downstream ETLD-like tempered profile.

Features (computational only)
-----------------------------
- Load ACE/EPAM CSV data.
- Convert time to distance from a shock (x in AU).
- Fit upstream (L, α, λ^-1) using a Mittag–Leffler model.
- Fit downstream (α, λ^-1, Lsd2) using a model with a precomputed integral.
- Return structured results for each energy channel.

Intended for P2–P5 channels of AC_H3_EPM_614092.csv,
but functions are generic.
"""

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.interpolate import interp1d
from mittag_leffler import ml as ml_ff


# ================================================================
# Data container
# ================================================================

@dataclass
class ChannelFit:
    """Hold upstream & downstream fit results for a single EPAM channel."""
    name: str
    energy_label: str

    x_up: np.ndarray
    y_up: np.ndarray
    y_up_fit: np.ndarray
    upstream_params: Tuple[float, float, float]
    upstream_errors: Tuple[float, float, float]

    x_down: np.ndarray
    y_down: np.ndarray
    y_down_fit: np.ndarray
    downstream_params: Tuple[float, float, float]
    downstream_errors: Tuple[float, float, float]


# ================================================================
# Physical Models
# ================================================================

def _integrand(x_prime: float, lambda_: float, alpha: float, Lsd2: float) -> float:
    """Integrand for the downstream tempered solution."""
    x_ratio = abs(x_prime) / Lsd2
    return np.exp(-lambda_ * abs(x_prime)) * ml_ff(-(x_ratio ** (alpha - 1)), alpha - 1)


def precompute_integrals(x_vals: np.ndarray, lambda_: float, alpha: float, Lsd2: float) -> np.ndarray:
    """Compute ∫₀ˣ f(x') dx' for the downstream model."""
    out = []
    for x in x_vals:
        val, _ = quad(_integrand, 0, abs(x), args=(lambda_, alpha, Lsd2))
        out.append(val)
    return np.array(out)


def make_downstream_model(x_grid: np.ndarray, lambda_g: float, alpha_g: float, Lsd2_g: float):
    """Return f0(x, α, λ^-1, Lsd2) with precomputed integrals."""
    precomp = precompute_integrals(x_grid, lambda_g, alpha_g, Lsd2_g)

    def f0(x, alpha, lambda_inv, Lsd2):
        x = np.asarray(x)
        xa = np.abs(x)

        lam = 1 / lambda_inv if lambda_inv != 0 else 0
        a = (1 / Lsd2)**(alpha - 1)
        para1 = 1 / (1 - (lam**(alpha - 1) / (lam**(alpha - 1) + a)))

        integ = interp1d(x_grid, precomp, kind="linear", fill_value="extrapolate")(xa)

        x_ratio = xa / Lsd2
        part2 = np.exp(-lam * xa) * ml_ff(-(x_ratio ** (alpha - 1)), alpha - 1)

        part1 = (1 - ((2 * (alpha - 1)) / alpha) * para1)
        part3 = ((2 * (alpha - 1)) / alpha) * para1 * (1 - lam * integ)

        return part1 * part2 + part3

    return f0


def mittag_leffler_upstream(x: np.ndarray, L: float, alpha: float, lambda_inv: float) -> np.ndarray:
    """Upstream Mittag–Leffler solution."""
    x = np.asarray(x)
    out = np.zeros_like(x)

    mask = x > 0
    if not np.any(mask):
        return out

    lam = 1 / lambda_inv if lambda_inv != 0 else 0
    xv = x[mask]

    x_ratio = xv / L
    ml_val = ml_ff(-(x_ratio ** (alpha - 1)), alpha - 1)

    out[mask] = np.exp(-lam * xv) * ml_val
    return out


# ================================================================
# Data I/O & Preprocessing
# ================================================================

def load_epam_csv(path: str, skiprows: int = 74, skipfooter: int = 3) -> pd.DataFrame:
    """Load ACE EPAM CSV."""
    df = pd.read_csv(path, skiprows=skiprows, skipfooter=skipfooter, engine="python")
    df.columns = ["time", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time")
    df = df.replace([-999.9, 0], np.nan)
    return df


def add_distance_from_shock(df: pd.DataFrame, shock_time: str, speed_km_s: float = 668.0) -> pd.DataFrame:
    """Add x (AU) distance from shock."""
    shock = pd.to_datetime(shock_time)
    if shock.tz is not None:
        shock = shock.tz_convert(None)

    idx = df.index
    if idx.tz is not None:
        idx = idx.tz_convert(None)

    dt = (shock - idx).total_seconds()
    df["time_elapsed"] = dt
    df["x"] = dt * speed_km_s * 6.68459e-9  # convert to AU
    return df


# ================================================================
# Fitting Helpers
# ================================================================

def _normalize(df: pd.DataFrame, channel: str, mask: np.ndarray) -> pd.DataFrame:
    """Interpolate + normalize a channel within a region."""
    out = df.copy()
    out[channel] = out[channel].interpolate()
    out[channel] = out[channel].where(out[channel] > 0, np.nan)

    sub = out.loc[mask, channel].dropna()
    if sub.empty:
        raise ValueError(f"No valid data in region for {channel}")

    out[channel + "_norm"] = out[channel] / sub.max()
    return out


def fit_channel(df: pd.DataFrame, channel: str, label: str,
                x_up_max: float = 0.05, x_down_min: float = -0.05) -> ChannelFit:

    # -------------------- Upstream --------------------
    mask_up = (df["x"] > 0) & (df["x"] <= x_up_max)
    df_up = _normalize(df, channel, mask_up)
    d_up = df_up.loc[mask_up].dropna(subset=[channel + "_norm"])

    x_up = d_up["x"].values
    y_up = d_up[channel + "_norm"].values

    p0_up = [5e-4, 1.2, 0.05]
    bounds_up = ([1e-4, 1.0, 0.01], [5e-3, 2.0, 0.1])

    popt_up, pcov_up = curve_fit(mittag_leffler_upstream, x_up, y_up,
                                 p0=p0_up, bounds=bounds_up, maxfev=20000)
    L_up, alpha_up, lam_inv_up = popt_up
    err_up = np.sqrt(np.diag(pcov_up))
    y_up_fit = mittag_leffler_upstream(x_up, *popt_up)

    # -------------------- Downstream --------------------
    mask_dn = (df["x"] < 0) & (df["x"] >= x_down_min)
    df_dn = _normalize(df, channel, mask_dn)
    d_dn = df_dn.loc[mask_dn].dropna(subset=[channel + "_norm"])

    x_dn = np.abs(d_dn["x"].values)
    y_dn = d_dn[channel + "_norm"].values

    x_grid = np.linspace(0, x_dn.max(), 100)
    f0 = make_downstream_model(x_grid, lambda_g=0.05, alpha_g=1.2, Lsd2_g=5e-4)

    p0_dn = [1.2, 0.05, 5e-4]
    bounds_dn = ([1.0, 0.01, 1e-4], [2.0, 0.1, 5e-3])

    popt_dn, pcov_dn = curve_fit(f0, x_dn, y_dn,
                                 p0=p0_dn, bounds=bounds_dn, maxfev=20000)
    alpha_dn, lam_inv_dn, Lsd2_dn = popt_dn
    err_dn = np.sqrt(np.diag(pcov_dn))
    y_dn_fit = f0(x_dn, *popt_dn)

    # -------------------- Return structured results --------------------
    return ChannelFit(
        name=channel,
        energy_label=label,
        x_up=x_up,
        y_up=y_up,
        y_up_fit=y_up_fit,
        upstream_params=(L_up, alpha_up, lam_inv_up),
        upstream_errors=tuple(err_up),
        x_down=x_dn,
        y_down=y_dn,
        y_down_fit=y_dn_fit,
        downstream_params=(alpha_dn, lam_inv_dn, Lsd2_dn),
        downstream_errors=tuple(err_dn)
    )


def fit_channels(df: pd.DataFrame,
                 channels: Sequence[str],
                 labels: Dict[str, str],
                 x_up_max: float = 0.05,
                 x_down_min: float = -0.05) -> Dict[str, ChannelFit]:
    """Return dict of fits for channels."""
    return {
        ch: fit_channel(df, ch, labels.get(ch, ""), x_up_max, x_down_min)
        for ch in channels
    }


# In[ ]:





# In[ ]:




