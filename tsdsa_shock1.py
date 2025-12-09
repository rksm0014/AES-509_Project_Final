from __future__ import annotations

"""
tsdsa_shock1
============

Module for analyzing energetic particle profiles around an
interplanetary shock using TSDSA-like upstream Mittag–Leffler
profiles and a downstream ETLD-like tempered profile.

Features
--------
- Load ACE/EPAM CSV data.
- Convert time to distance from a shock (x in AU).
- Fit upstream (L, α, λ^-1) using a Mittag–Leffler model.
- Fit downstream (α, λ^-1, Lsd2) using a model with a precomputed integral.
- Plot upstream/downstream profiles (2×2 panel).
- Plot energy-channel trends of parameters with error bars.

Intended for P2–P5 channels of AC_H3_EPM_614092.csv, but the
functions are generic.
"""

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.interpolate import interp1d
from mittag_leffler import ml as ml_ff


# -------------------------------------------------------------------
# Data container
# -------------------------------------------------------------------

@dataclass
class ChannelFit:
    """
    Hold upstream and downstream fit results (with uncertainties)
    for a single EPAM energy channel.
    """
    name: str
    energy_label: str

    # Upstream
    x_up: np.ndarray
    y_up: np.ndarray
    y_up_fit: np.ndarray
    upstream_params: Tuple[float, float, float]      # (L, α, λ^-1)
    upstream_errors: Tuple[float, float, float]      # (σ_L, σ_α, σ_λ^-1)

    # Downstream
    x_down: np.ndarray
    y_down: np.ndarray
    y_down_fit: np.ndarray
    downstream_params: Tuple[float, float, float]    # (α, λ^-1, Lsd2)
    downstream_errors: Tuple[float, float, float]    # (σ_α, σ_λ^-1, σ_Lsd2)


# -------------------------------------------------------------------
# Physical models
# -------------------------------------------------------------------

def _integrand(x_prime: float, lambda_: float, alpha: float, Lsd2: float) -> float:
    """
    Integrand for the downstream ETLD-like model.
    """
    x_L_ratio1 = abs(x_prime) / Lsd2
    return np.exp(-lambda_ * abs(x_prime)) * ml_ff(
        -x_L_ratio1 ** (alpha - 1.0),
        alpha - 1.0
    )


def precompute_integrals(x_vals: np.ndarray,
                         lambda_: float,
                         alpha: float,
                         Lsd2: float) -> np.ndarray:
    """
    Precompute ∫_0^{x'} integrand dx' for each x' in x_vals (>= 0).
    """
    integrals = []
    for x_val in x_vals:
        val, _ = quad(_integrand, 0.0, abs(x_val), args=(lambda_, alpha, Lsd2))
        integrals.append(val)
    return np.array(integrals)


def make_downstream_model(x_grid_for_int: np.ndarray,
                          lambda_guess: float,
                          alpha_guess: float,
                          Lsd2_guess: float):
    """
    Build a downstream ETLD-like model f_0(x, α, λ^-1, Lsd2) using
    precomputed integrals on x_grid_for_int with some initial guesses.

    The integral kernel is fixed by (lambda_guess, alpha_guess, Lsd2_guess);
    the fit parameters still update the Mittag–Leffler part.
    """
    precomputed_integrals = precompute_integrals(
        x_grid_for_int, lambda_guess, alpha_guess, Lsd2_guess
    )

    def f_0_fit(x: np.ndarray,
                alpha: float,
                lambda_inv: float,
                Lsd2: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x_abs = np.abs(x)

        lambda_ = 1.0 / lambda_inv if lambda_inv != 0 else 0.0

        # para1 term
        a = (1.0 / Lsd2) ** (alpha - 1.0)
        para1 = 1.0 / (1.0 - ((lambda_ ** (alpha - 1.0)) /
                              (lambda_ ** (alpha - 1.0) + a)))

        # interpolate the integral on |x| (kernel from initial guesses)
        integral_interp = interp1d(
            x_grid_for_int,
            precomputed_integrals,
            kind="linear",
            fill_value="extrapolate"
        )
        integral_values = integral_interp(x_abs)

        # Mittag-Leffler factor with fit α, Lsd2
        x_L_ratio = x_abs / Lsd2
        part2 = np.exp(-lambda_ * x_abs) * ml_ff(
            -x_L_ratio ** (alpha - 1.0),
            alpha - 1.0
        )

        part1 = (1.0 - (((2.0 * (alpha - 1.0)) / alpha) * para1))
        part3 = (((2.0 * (alpha - 1.0)) / alpha) * para1) * (1.0 - (lambda_ * integral_values))

        return part1 * part2 + part3

    return f_0_fit


def mittag_leffler_upstream(x: np.ndarray,
                            L: float,
                            alpha: float,
                            lambda_inv: float) -> np.ndarray:
    """
    Upstream Mittag–Leffler TSDSA-like profile:

       f(x) = exp(-λ x) * E_{α-1}(-(x/L)^{α-1})  for x > 0

    where λ = 1 / λ_inv and ml_ff(z, β) implements E_β(z).
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)

    mask = x > 0
    if not np.any(mask):
        return out

    lam = 1.0 / lambda_inv if lambda_inv != 0 else 0.0
    xv = x[mask]

    x_L = np.abs(xv) / L
    arg = -(x_L ** (alpha - 1.0))
    ml_vals = ml_ff(arg, alpha - 1.0)

    if lam > 0:
        out[mask] = np.exp(-lam * xv) * ml_vals
    else:
        out[mask] = ml_vals

    return out


# -------------------------------------------------------------------
# I/O and preprocessing
# -------------------------------------------------------------------

def load_epam_csv(path: str,
                  skiprows: int = 74,
                  skipfooter: int = 3) -> pd.DataFrame:
    """
    Load ACE/EPAM CSV file with columns:
        time, P2, P3, P4, P5, P6, P7, P8

    Returns a DataFrame indexed by time (naive, no timezone).
    """
    df = pd.read_csv(
        path,
        skiprows=skiprows,
        skipfooter=skipfooter,
        engine="python",
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

    # parse shock_time
    shock_ts = pd.to_datetime(shock_time)
    if shock_ts.tz is not None:
        shock_ts = shock_ts.tz_convert(None)

    idx = out.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)

    dt = shock_ts - idx
    out["time_elapsed"] = dt.total_seconds()

    conv = 6.68459e-9  # km/s -> AU/s
    out["x"] = out["time_elapsed"] * speed_km_s * conv

    return out


# -------------------------------------------------------------------
# Fitting helpers
# -------------------------------------------------------------------

def _normalize_in_region(df: pd.DataFrame,
                         channel: str,
                         mask: np.ndarray) -> pd.DataFrame:
    """
    Interpolate and normalize a channel inside the given region (mask).
    """
    out = df.copy()
    out[channel] = out[channel].interpolate()
    out[channel] = out[channel].where(out[channel] > 0, np.nan)

    sub = out.loc[mask, channel].dropna()
    if sub.empty:
        raise ValueError(f"No valid data for channel {channel} in the selected region.")

    max_val = sub.max()
    out[channel + "_norm"] = out[channel] / max_val
    return out


def fit_channel(df: pd.DataFrame,
                channel: str,
                energy_label: str,
                x_up_max: float = 0.05,
                x_down_min: float = -0.05) -> ChannelFit:
    """
    Fit one energy channel upstream and downstream.

    Upstream:  Mittag–Leffler model -> (L, α, λ^-1) + errors.
    Downstream: ETLD-like model     -> (α, λ^-1, Lsd2) + errors.
    """
    # -------- Upstream region --------
    mask_up = (df["x"] > 0) & (df["x"] <= x_up_max)
    df_up = _normalize_in_region(df, channel, mask_up)
    sub_up = df_up.loc[mask_up].dropna(subset=[channel + "_norm"])

    x_up = sub_up["x"].values
    y_up = sub_up[channel + "_norm"].values

    p0_up = [5e-4, 1.2, 0.05]  # L, α, λ^-1
    bounds_up = ([1e-4, 1.0, 0.01], [5e-3, 2.0, 0.1])

    popt_up, pcov_up = curve_fit(
        mittag_leffler_upstream,
        x_up,
        y_up,
        p0=p0_up,
        bounds=bounds_up,
        maxfev=20000,
    )
    L_up, alpha_up, lam_inv_up = popt_up
    err_up = np.sqrt(np.diag(pcov_up))  # (σ_L, σ_α, σ_λ^-1)
    y_up_fit = mittag_leffler_upstream(x_up, *popt_up)

    # -------- Downstream region --------
    mask_dn = (df["x"] < 0) & (df["x"] >= x_down_min)
    df_dn = _normalize_in_region(df, channel, mask_dn)
    sub_dn = df_dn.loc[mask_dn].dropna(subset=[channel + "_norm"])

    x_down = np.abs(sub_dn["x"].values)
    y_down = sub_dn[channel + "_norm"].values

    if len(x_down) == 0:
        raise ValueError(f"No downstream data for channel {channel}")

    # Build downstream model using precomputed integrals
    x_grid = np.linspace(0.0, x_down.max(), 100)
    lambda_guess = 0.05
    alpha_guess = 1.2
    Lsd2_guess = 5e-4
    f0_model = make_downstream_model(x_grid, lambda_guess, alpha_guess, Lsd2_guess)

    p0_dn = [1.2, 0.05, 5e-4]          # α, λ^-1, Lsd2
    bounds_dn = ([1.0, 0.01, 1e-4], [2.0, 0.1, 5e-3])

    popt_dn, pcov_dn = curve_fit(
        f0_model,
        x_down,
        y_down,
        p0=p0_dn,
        bounds=bounds_dn,
        maxfev=20000,
    )
    alpha_dn, lam_inv_dn, Lsd2_dn = popt_dn
    err_dn = np.sqrt(np.diag(pcov_dn))  # (σ_α, σ_λ^-1, σ_Lsd2)
    y_down_fit = f0_model(x_down, *popt_dn)

    return ChannelFit(
        name=channel,
        energy_label=energy_label,
        x_up=x_up,
        y_up=y_up,
        y_up_fit=y_up_fit,
        upstream_params=(L_up, alpha_up, lam_inv_up),
        upstream_errors=tuple(err_up),
        x_down=x_down,
        y_down=y_down,
        y_down_fit=y_down_fit,
        downstream_params=(alpha_dn, lam_inv_dn, Lsd2_dn),
        downstream_errors=tuple(err_dn),
    )


def fit_channels(df: pd.DataFrame,
                 channels: Sequence[str],
                 labels: Dict[str, str],
                 x_up_max: float = 0.05,
                 x_down_min: float = -0.05) -> Dict[str, ChannelFit]:
    """
    Fit multiple channels and return a dict:
        { "P2": ChannelFit, "P3": ChannelFit, ... }
    """
    results: Dict[str, ChannelFit] = {}
    for ch in channels:
        cf = fit_channel(df, ch, labels.get(ch, ""), x_up_max=x_up_max, x_down_min=x_down_min)
        results[ch] = cf
    return results


# -------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------

def plot_profiles(results: Dict[str, ChannelFit],
                  save: str | None = None) -> None:
    """
    Plot upstream and downstream profiles for up to four channels (2×2 grid).
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()

    for ax, (ch, cf) in zip(axs, results.items()):
        L_up, alpha_up, lam_inv_up = cf.upstream_params
        alpha_dn, lam_inv_dn, Lsd2_dn = cf.downstream_params

        # Upstream (x > 0)
        ax.semilogy(cf.x_up, cf.y_up, color="blue", linewidth=1, label="Upstream obs")
        ax.semilogy(
            cf.x_up,
            cf.y_up_fit,
            linestyle="--",
            color="orange",
            linewidth=2,
            label=(f"Up fit: α={alpha_up:.2f}, "
                   f"L={L_up:.4g} AU, "
                   f"λ⁻¹={lam_inv_up:.3f}")
        )

        # Downstream plotted at negative x for symmetry
        ax.semilogy(-cf.x_down, cf.y_down, color="red", linewidth=1, label="Downstream obs")
        ax.semilogy(
            -cf.x_down,
            cf.y_down_fit,
            linestyle="--",
            color="black",
            linewidth=2,
            label=(f"Down fit: α={alpha_dn:.2f}, "
                   f"L={Lsd2_dn:.4g} AU, "
                   f"λ⁻¹={lam_inv_dn:.3f}")
        )

        ax.axvline(0, color="k", linestyle="-", linewidth=1.5)
        ax.set_title(f"{ch} ({cf.energy_label})", fontsize=12)
        ax.set_xlabel("x (AU)")
        ax.set_ylabel("Normalized flux")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(fontsize=7, loc="best")

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300)
    plt.show()


def plot_energy_trends(results: Dict[str, ChannelFit],
                       bands: Dict[str, Tuple[float, float]],
                       save_prefix: str | None = None) -> None:
    """
    Plot energy-channel trends with error bars for:

      - Upstream α and Downstream α
      - Upstream L and Downstream L (Lsd2)
      - Upstream λ^{-1} and Downstream λ^{-1}
    """
    channels = list(results.keys())
    x = np.arange(len(channels))

    energies = []

    alpha_up = []
    alpha_dn = []
    alpha_up_err = []
    alpha_dn_err = []

    L_up = []
    L_dn = []
    L_up_err = []
    L_dn_err = []

    lam_inv_up = []
    lam_inv_dn = []
    lam_inv_up_err = []
    lam_inv_dn_err = []

    for ch in channels:
        lo, hi = bands[ch]
        energies.append(np.sqrt(lo * hi))

        cf = results[ch]

        # ----- upstream -----
        L, a_up, lam_u = cf.upstream_params
        σL, σa_up, σlam_u = cf.upstream_errors

        # ----- downstream -----
        a_dn, lam_d, Lsd2_dn = cf.downstream_params
        σa_dn, σlam_d, σLsd2 = cf.downstream_errors

        alpha_up.append(a_up)
        alpha_dn.append(a_dn)
        alpha_up_err.append(σa_up)
        alpha_dn_err.append(σa_dn)

        L_up.append(L)
        L_dn.append(Lsd2_dn)
        L_up_err.append(σL)
        L_dn_err.append(σLsd2)

        lam_inv_up.append(lam_u)
        lam_inv_dn.append(lam_d)
        lam_inv_up_err.append(σlam_u)
        lam_inv_dn_err.append(σlam_d)

    labels = [f"{ch}\n({int(E)} keV)" for ch, E in zip(channels, energies)]

    # =========================
    # 1) α_up & α_dn
    # =========================
    plt.figure(figsize=(7, 5))
    plt.errorbar(
        x, alpha_up, yerr=alpha_up_err,
        fmt="o-", capsize=3, label="Upstream α"
    )
    plt.errorbar(
        x, alpha_dn, yerr=alpha_dn_err,
        fmt="s--", capsize=3, label="Downstream α"
    )
    plt.xticks(x, labels)
    plt.xlabel("Energy Channel")
    plt.ylabel("α")
    plt.title("Upstream & Downstream α vs Energy")
    plt.grid(True, linestyle="--", lw=0.5)
    plt.legend()
    if save_prefix:
        plt.savefig(save_prefix + "_alpha_up_down.png", dpi=300)
    plt.show()

    # =========================
    # 2) L_up & L_dn
    # =========================
    plt.figure(figsize=(7, 5))
    plt.errorbar(
        x, L_up, yerr=L_up_err,
        fmt="o-", capsize=3, label="Upstream L"
    )
    plt.errorbar(
        x, L_dn, yerr=L_dn_err,
        fmt="s--", capsize=3, label="Downstream L"
    )
    plt.xticks(x, labels)
    plt.xlabel("Energy Channel")
    plt.ylabel("L (AU)")
    plt.title("Upstream & Downstream L vs Energy")
    plt.grid(True, linestyle="--", lw=0.5)
    plt.legend()
    if save_prefix:
        plt.savefig(save_prefix + "_L_up_down.png", dpi=300)
    plt.show()

    # =========================
    # 3) λ^{-1}_up & λ^{-1}_dn
    # =========================
    plt.figure(figsize=(7, 5))
    plt.errorbar(
        x, lam_inv_up, yerr=lam_inv_up_err,
        fmt="o-", capsize=3, label=r"Upstream $\lambda^{-1}$"
    )
    plt.errorbar(
        x, lam_inv_dn, yerr=lam_inv_dn_err,
        fmt="s--", capsize=3, label=r"Downstream $\lambda^{-1}$"
    )
    plt.xticks(x, labels)
    plt.xlabel("Energy Channel")
    plt.ylabel(r"$\lambda^{-1}$ (AU$^{-1}$)")
    plt.title(r"Upstream & Downstream $\lambda^{-1}$ vs Energy")
    plt.grid(True, linestyle="--", lw=0.5)
    plt.legend()
    if save_prefix:
        plt.savefig(save_prefix + "_lambda_inv_up_down.png", dpi=300)
    plt.show()
