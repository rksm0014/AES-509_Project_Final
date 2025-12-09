#!/usr/bin/env python
# coding: utf-8

# In[10]:


"""
tsdsa_shock.py
================

Module for analyzing energetic particle profiles around an
interplanetary shock.

Features:
- Load ACE/EPAM CSV data.
- Convert time to distance from a shock.
- Fit upstream TSDSA (Mittag–Leffler–type) profiles.
- Fit downstream tempered exponential profiles.
- Plot profiles and energy-channel trends.

This is the module for your final project.
"""

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mittag_leffler import ml as ml_ff   # your Mittag–Leffler library


# -------------------------------------------------------------------
# Data container
# -------------------------------------------------------------------

@dataclass
class ChannelFit:
    """Holds upstream and downstream fit results for one energy channel."""
    name: str
    energy_label: str

    x_up: np.ndarray
    y_up: np.ndarray
    y_up_fit: np.ndarray
    upstream_params: Tuple[float, float, float]  # (L, alpha, lambda_inv)

    x_down: np.ndarray
    y_down: np.ndarray
    y_down_fit: np.ndarray
    downstream_params: Tuple[float, float]       # (lambda_inv, scale)


# -------------------------------------------------------------------
# Physical models
# -------------------------------------------------------------------

def tsdsa_upstream(x: np.ndarray,
                   L: float,
                   alpha: float,
                   lambda_inv: float) -> np.ndarray:
    """
    Mittag–Leffler–type TSDSA upstream profile:

        f(x) = exp(-λ x) * E_{α-1}(-(x/L)^{α-1})

    where λ = 1 / lambda_inv and E is the Mittag–Leffler function
    from the `mittag_leffler` package (ml_ff).
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)

    mask = x > 0
    if not np.any(mask):
        return out

    lam = 1.0 / lambda_inv if lambda_inv != 0 else 0.0
    xv = x[mask]

    x_L = np.abs(xv) / L
    arg = -(x_L ** (alpha - 1.0))

    # Your library signature: ml_ff(z, alpha_minus_1)
    ml_vals = ml_ff(arg, alpha - 1.0)

    if lam > 0:
        out[mask] = np.exp(-lam * xv) * ml_vals
    else:
        out[mask] = ml_vals

    return out


def downstream_tempered(x: np.ndarray,
                        lambda_inv: float,
                        scale: float) -> np.ndarray:
    """
    Simple downstream profile:

        f(x) = scale * exp(-λ |x|),   λ = 1 / lambda_inv
    """
    x = np.asarray(x, dtype=float)
    lam = 1.0 / lambda_inv if lambda_inv != 0 else 0.0
    return scale * np.exp(-lam * np.abs(x))


# -------------------------------------------------------------------
# I/O utilities
# -------------------------------------------------------------------

def load_epam_csv(path: str,
                  skiprows: int = 74,
                  skipfooter: int = 3) -> pd.DataFrame:
    """
    Load ACE/EPAM CSV file with columns:
        time, P2, P3, P4, P5, P6, P7, P8

    Returns a DataFrame indexed by time.
    """
    df = pd.read_csv(
        path,
        skiprows=skiprows,
        skipfooter=skipfooter,
        engine="python"
    )
    df.columns = ["time", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df = df.set_index("time")
    df = df.replace([-999.9, 0], np.nan)
    return df


def add_distance_from_shock(df: pd.DataFrame,
                            shock_time: str,
                            speed_km_s: float) -> pd.DataFrame:
    """
    Add 'time_elapsed' (s) and 'x' (AU) columns relative to the shock.

    x > 0  upstream (before shock)
    x < 0  downstream (after shock)
    """
    out = df.copy()

    # Parse the shock time
    shock_ts = pd.to_datetime(shock_time)

    # Make shock_ts timezone-naive if needed
    if shock_ts.tz is not None:
        shock_ts = shock_ts.tz_convert(None)

    # Make index timezone-naive if needed
    idx = out.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)

    # Time difference (TimedeltaIndex → seconds)
    dt = shock_ts - idx
    out["time_elapsed"] = dt.total_seconds()

    # Convert seconds to AU with solar wind speed
    conv = 6.68459e-9  # km/s → AU/s
    out["x"] = out["time_elapsed"] * speed_km_s * conv

    return out


# -------------------------------------------------------------------
# Fitting helpers
# -------------------------------------------------------------------

def _normalize_channel(df: pd.DataFrame, ch: str, mask) -> pd.DataFrame:
    """
    Interpolate and normalize flux for one channel in the selected region.
    """
    out = df.copy()
    out[ch] = out[ch].interpolate()
    out[ch] = out[ch].where(out[ch] > 0, np.nan)

    sub = out.loc[mask, ch].dropna()
    if sub.empty:
        raise ValueError(f"No valid data for channel {ch} in the chosen region.")

    max_val = sub.max()
    out[ch + "_norm"] = out[ch] / max_val
    return out


def fit_upstream(df: pd.DataFrame,
                 ch: str,
                 x_max: float = 0.05):
    """
    Fit the TSDSA upstream model for one channel.

    Returns: x_up, y_up, params, y_fit
             params = (L, alpha, lambda_inv)
    """
    mask = (df["x"] > 0) & (df["x"] <= x_max)
    df_norm = _normalize_channel(df, ch, mask)
    sub = df_norm.loc[mask].dropna(subset=[ch + "_norm"])

    x_up = sub["x"].values
    y_up = sub[ch + "_norm"].values

    p0 = [5e-4, 1.3, 0.05]  # L, alpha, lambda_inv
    bounds = ([1e-4, 1.0, 0.01], [5e-3, 2.0, 0.2])

    popt, _ = curve_fit(
        tsdsa_upstream,
        x_up,
        y_up,
        p0=p0,
        bounds=bounds,
        maxfev=10000
    )
    y_fit = tsdsa_upstream(x_up, *popt)
    return x_up, y_up, tuple(popt), y_fit


def fit_downstream(df: pd.DataFrame,
                   ch: str,
                   x_min: float = -0.05):
    """
    Fit the downstream tempered exponential model for one channel.

    Returns: x_down, y_down, params, y_fit
             params = (lambda_inv, scale)
    """
    mask = (df["x"] < 0) & (df["x"] >= x_min)
    df_norm = _normalize_channel(df, ch, mask)
    sub = df_norm.loc[mask].dropna(subset=[ch + "_norm"])

    x_down = np.abs(sub["x"].values)
    y_down = sub[ch + "_norm"].values

    p0 = [0.05, 1.0]  # lambda_inv, scale
    bounds = ([0.01, 0.1], [0.2, 10.0])

    popt, _ = curve_fit(
        downstream_tempered,
        x_down,
        y_down,
        p0=p0,
        bounds=bounds,
        maxfev=10000
    )
    y_fit = downstream_tempered(x_down, *popt)
    return x_down, y_down, tuple(popt), y_fit


def fit_all(df: pd.DataFrame,
            channels: Sequence[str],
            labels: Dict[str, str],
            x_max: float = 0.05,
            x_min: float = -0.05) -> Dict[str, ChannelFit]:
    """
    Fit upstream and downstream profiles for multiple channels.

    Returns a dict: { channel_name: ChannelFit }
    """
    results: Dict[str, ChannelFit] = {}

    for ch in channels:
        x_up, y_up, up_params, y_up_fit = fit_upstream(df, ch, x_max=x_max)
        x_dn, y_dn, dn_params, y_dn_fit = fit_downstream(df, ch, x_min=x_min)

        results[ch] = ChannelFit(
            name=ch,
            energy_label=labels.get(ch, ""),
            x_up=x_up,
            y_up=y_up,
            y_up_fit=y_up_fit,
            upstream_params=up_params,
            x_down=x_dn,
            y_down=y_dn,
            y_down_fit=y_dn_fit,
            downstream_params=dn_params
        )

    return results


# -------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------

def plot_profiles(results: Dict[str, ChannelFit],
                  save: str | None = None) -> None:
    """
    Plot upstream and downstream profiles for up to four channels (2×2 grid).
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()

    for ax, (ch, cf) in zip(axs, results.items()):
        # Upstream
        L, alpha, lam_inv = cf.upstream_params
        ax.semilogy(cf.x_up, cf.y_up, "b.", label="Upstream data")
        ax.semilogy(cf.x_up, cf.y_up_fit, "b--",
                    label=f"Up fit: α={alpha:.2f}, L={L:.4g} AU, λ⁻¹={lam_inv:.3f}")

        # Downstream (plotted at negative x for symmetry)
        lam_inv_dn, scale_dn = cf.downstream_params
        ax.semilogy(-cf.x_down, cf.y_down, "r.", label="Downstream data")
        ax.semilogy(-cf.x_down, cf.y_down_fit, "r--",
                    label=f"Down fit: λ⁻¹={lam_inv_dn:.3f}, scale={scale_dn:.2f}")

        ax.axvline(0, color="k")
        ax.set_xlabel("x (AU)")
        ax.set_ylabel("Normalized flux")
        ax.set_title(f"{ch} ({cf.energy_label})")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(fontsize=7, loc="best")

    plt.tight_layout()
    if save is not None:
        plt.savefig(save, dpi=300)
    plt.show()


def plot_energy_trends(results: Dict[str, ChannelFit],
                       bands: Dict[str, Tuple[float, float]],
                       save_prefix: str | None = None) -> None:
    """
    Plot upstream α, upstream L, and downstream λ⁻¹ vs energy channel.
    """
    channels = list(results.keys())
    x = np.arange(len(channels))

    energies = []
    alpha_up = []
    L_up = []
    lam_inv_dn = []

    for ch in channels:
        lo, hi = bands[ch]
        energies.append(np.sqrt(lo * hi))

        cf = results[ch]
        L, alpha, _ = cf.upstream_params
        lam_down, _ = cf.downstream_params

        alpha_up.append(alpha)
        L_up.append(L)
        lam_inv_dn.append(lam_down)

    labels = [f"{ch}\n({int(E)} keV)" for ch, E in zip(channels, energies)]

    # α
    plt.figure(figsize=(7, 5))
    plt.plot(x, alpha_up, "o-")
    plt.xticks(x, labels)
    plt.ylabel("α (upstream)")
    plt.xlabel("Energy channel")
    plt.title("Upstream α vs Energy")
    plt.grid(True, ls="--")
    if save_prefix:
        plt.savefig(save_prefix + "_alpha.png", dpi=300)
    plt.show()

    # L
    plt.figure(figsize=(7, 5))
    plt.plot(x, L_up, "o-")
    plt.xticks(x, labels)
    plt.ylabel("L (AU)")
    plt.xlabel("Energy channel")
    plt.title("Upstream L vs Energy")
    plt.grid(True, ls="--")
    if save_prefix:
        plt.savefig(save_prefix + "_L.png", dpi=300)
    plt.show()

    # λ⁻¹ downstream
    plt.figure(figsize=(7, 5))
    plt.plot(x, lam_inv_dn, "s--")
    plt.xticks(x, labels)
    plt.ylabel(r"$\lambda^{-1}$ (AU$^{-1}$)")
    plt.xlabel("Energy channel")
    plt.title(r"Downstream $\lambda^{-1}$ vs Energy")
    plt.grid(True, ls="--")
    if save_prefix:
        plt.savefig(save_prefix + "_lambda_inv.png", dpi=300)
    plt.show()


# In[11]:


import importlib
import tsdsa_shock
importlib.reload(tsdsa_shock)

print(tsdsa_shock.__file__)      # see which file it’s using
print("fit_all" in dir(tsdsa_shock))  # check if fit_all is there


# In[ ]:




