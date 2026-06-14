"""
Habitable-zone target selection pipeline aligned with Project 10 (PLATO / PlatoSim).

PlatoSim/PLATOnium is the intended simulator (picsim → varsim → payload → platonium).
When those products are unavailable, this project uses **PSLS** (PLATO Solar-like
Light-curve Simulator, `pip install psls`): `sls.gen_up` produces solar-like
background variability (granulation, oscillations, photometry noise), onto which we
inject a Mandel–Agol-style geometric transit.

This module provides:
  - Physics: L*/L☉, incident flux S (Earth units), equilibrium temperature T_eq
  - Habitable-zone screening and transparent ranking score
  - Optional pure-synthetic box transits (`generate_synthetic_plato_like_lightcurve`)

Replace light-curve generation with real PlatoSim/PSLS exports when available
(time, flux columns) via `load_time_flux_csv`.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.constants import R_sun, au as AU

# ---------------------------------------------------------------------------
# Physical constants 
# ---------------------------------------------------------------------------

T_SUN_K = 5772.0
SIGMA_S = 0.3  # width for H_HZ(S) Gaussian around Earth-like irradiation; ~matches the
               # conservative HZ flux band (0.95–1.37 S⊕) so cold/hot planets are penalized.
               # Was 1.0 (too wide — barely discriminated flux; cold B3 tied the HZ planets).
               # NOTE: changing this requires recomputing rank_score_label + retraining rankers.
SIGMA_R = 0.5  # Earth radii scale for H_Rp
SIGMA_DET = 0.15  # maps SNR to [0,1]-like detection term



@dataclass
class Star:
    """Host star (solar units for radius; Kelvin for Teff)."""

    name: str
    radius_solar: float  # R*/R☉
    teff_k: float  # T*

    def luminosity_solar(self) -> float:
        """L*/L☉ = (R*/R☉)² (T*/T☉)⁴."""
        return (self.radius_solar**2) * (self.teff_k / T_SUN_K) ** 4  # Stefan-Boltzmann: L ∝ R² T⁴


@dataclass
class Planet:
    """Planet in injected truth table."""

    name: str
    radius_earth: float  # R_p / R⊕
    period_days: float
    semi_major_axis_au: float
    impact_parameter: float = 0.0  # 0–1, for optional duration tweak
    albedo: float = 0.3


# ---------------------------------------------------------------------------
#  Habitability-related quantities
# ---------------------------------------------------------------------------


def stellar_luminosity_solar(radius_solar: float, teff_k: float) -> float:
    """L*/L☉ = (R*/R☉)² (T*/T☉)⁴."""
    return (radius_solar**2) * (teff_k / T_SUN_K) ** 4


def incident_flux_earth_units(luminosity_solar: float, a_au: float) -> float:
    """S = (L*/L☉) / (a/AU)² — incident flux relative to Earth at 1 AU for the Sun."""
    if a_au <= 0:
        raise ValueError("semi-major axis must be positive")
    return luminosity_solar / (a_au**2)  # inverse-square law; S=1 when L=L☉, a=1 AU


def equilibrium_temperature_k(
    teff_k: float,
    r_star_m: float,
    a_m: float,
    albedo: float = 0.3,
) -> float:
    """
    T_eq = T* sqrt(R*/(2a)) (1-A)^(1/4) with R* and a in the same length unit.
    """
    if a_m <= 0 or r_star_m <= 0:
        raise ValueError("radius and semi-major axis must be positive")
    return float(teff_k * np.sqrt(r_star_m / (2.0 * a_m)) * ((1.0 - albedo) ** 0.25))  # energy balance: absorbed stellar flux = emitted blackbody; factor 2 assumes uniform redistribution


def equilibrium_temperature_from_au(
    teff_k: float,
    radius_solar: float,
    a_au: float,
    albedo: float = 0.3,
) -> float:
    """Convenience: R* in R☉, a in AU."""
    r_m = radius_solar * R_sun.to_value(u.m)
    a_m = a_au * AU.to_value(u.m)
    return equilibrium_temperature_k(teff_k, r_m, a_m, albedo=albedo)


# ---------------------------------------------------------------------------
#  Habitable zone (using flux of main star and equilibrium temperature to define)
# ---------------------------------------------------------------------------


def habitable_zone_flags(
    s_earth: float,
    teq_k: float,
    s_inner: float = 0.95,
    s_outer: float = 1.37,
    teq_inner_k: float = 180.0,
    teq_outer_k: float = 270.0,
) -> dict[str, Any]:
    """
    Approximate conservative HZ screen: stellar flux and T_eq bands.
    These are screening tools, not evidence of habitability.
    """
    in_flux_hz = s_inner <= s_earth <= s_outer  # conservative Kopparapu-like flux bounds (runaway GH inner, max GH outer)
    in_teq_band = teq_inner_k <= teq_k <= teq_outer_k  # secondary T_eq screen; both must pass for confident HZ
    near_hz = (0.7 <= s_earth <= 1.6) or (160 <= teq_k <= 300)  # optimistic extended bounds for "interesting" targets
    return {
        "in_habitable_zone_flux": bool(in_flux_hz),
        "in_habitable_zone_teq_band": bool(in_teq_band),
        "near_habitable_zone_loose": bool(near_hz),
        "s_earth": s_earth,
        "teq_k": teq_k,
    }


# ---------------------------------------------------------------------------
# Task 7 — Transparent ranking model 
# ---------------------------------------------------------------------------


def h_hz_score(s_earth: float, sigma_s: float = SIGMA_S) -> float:
    """H_HZ = exp(-(S-1)² / (2 σ_S²))."""
    return float(np.exp(-((s_earth - 1.0) ** 2) / (2.0 * sigma_s**2)))  

def h_rp_score(radius_earth: float, sigma_r: float = SIGMA_R) -> float:
    """H_Rp = exp(-(R_p - 1.2)² / (2 σ_R²)) — favors Earth/super-Earth sizes."""
    return float(np.exp(-((radius_earth - 1.2) ** 2) / (2.0 * sigma_r**2)))  


def h_host_score(teff_k: float) -> float:
    """Reward Sun-like hosts: Gaussian around 5800 K with width 800 K."""
    t0, w = 5800.0, 800.0  # solar analog center; w=800 K means M-dwarfs (~3500 K) score ~0
    return float(np.exp(-((teff_k - t0) ** 2) / (2.0 * w**2)))


def h_det_score(snr_proxy: float, sigma_det: float = SIGMA_DET) -> float:
    """Map detection strength to [0,1]; use BLS power or SNR-like proxy. Using exponential bcz higher snr --> better"""
    x = float(np.clip(snr_proxy, 0.0, 20.0))  # cap prevents outlier BLS spikes from dominating
    return float(1.0 - np.exp(-x / max(sigma_det * 10.0, 1e-6)))  # exponential saturation 


def h_stab_score(eccentricity: float | None) -> float:
    """Orbital stability, Gaussian around e = 0"""
    if eccentricity is None:
        return 0.5  # unknown ecc --> neutral score
    return float(np.exp(-((eccentricity) ** 2) / (2.0 * 0.1**2)))  


def habitability_rank_score(
    s_earth: float,
    radius_earth: float,
    teff_k: float,
    snr_proxy: float,
    eccentricity: float | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Score = H_HZ + 0.2 H_Rp + 0.15 H_host + 0.15 H_det + 0.1 H_stab
    (document typo tH_host / tH_det corrected to H_host / H_det).
    """
    w = weights or {
        "H_HZ": 1.0,   # dominant term — flux position in HZ drives ranking
        "H_Rp": 0.2,   # secondary: rocky size preference
        "H_host": 0.15, # tertiary: penalize non-solar hosts
        "H_det": 0.15,  # detection quality (BLS SNR proxy)
        "H_stab": 0.1,  # smallest weight: ecc often unknown, neutral fallback common
    }
    hhz = h_hz_score(s_earth)
    hrp = h_rp_score(radius_earth)
    hh = h_host_score(teff_k)
    hd = h_det_score(snr_proxy)
    hs = h_stab_score(eccentricity)
    total = (
        w["H_HZ"] * hhz
        + w["H_Rp"] * hrp
        + w["H_host"] * hh
        + w["H_det"] * hd
        + w["H_stab"] * hs
    )
    return {
        "H_HZ": hhz,
        "H_Rp": hrp,
        "H_host": hh,
        "H_det": hd,
        "H_stab": hs,
        "rank_score": float(total),
    }


# ---------------------------------------------------------------------------
# Light curves: PSLS 
# ---------------------------------------------------------------------------

LightCurveBackend = Literal["psls", "synthetic"]


def _transit_depth_from_radii(rp_re: float, r_star_solar: float) -> float:
    """Approximate (R_p/R_*)^2 with R_p in R_earth, R_star in R_sun."""
    r_earth_m = 6.371e6
    r_sun_m = R_sun.to_value(u.m)
    rp_m = rp_re * r_earth_m
    rs_m = r_star_solar * r_sun_m
    return float((rp_m / rs_m) ** 2)  # transit depth = (Rp/R*)²; geometric occultation fraction of stellar disk


def _transit_duration_days(period_days: float) -> float:
    """Rough ingress+egress duration scale for BLS-friendly grids."""
    duration_days = 0.05 * (period_days / 10.0) ** (1.0 / 3.0)   #T_transit \propto P^(1/3)
    return float(np.clip(duration_days, 0.08, 0.45))


def apply_box_transit_to_flux(
    flux: np.ndarray,
    time_days: np.ndarray,
    period_days: float,
    t0_days: float,
    rp_earth: float,
    r_star_solar: float,
) -> np.ndarray:
    """Multiply flux by (1 - depth) in-transit bins (relative flux units)."""
    depth = _transit_depth_from_radii(rp_earth, r_star_solar)
    dur = _transit_duration_days(period_days)
    phase = ((time_days - t0_days) / period_days) % 1.0  # fold all times into [0,1); t0 maps to phase=0
    half = 0.5 * dur / period_days  # half-duration in phase units
    in_transit = (phase < half) | (phase > 1.0 - half)  # OR handles transit straddling the 0/1 phase boundary
    out = np.array(flux, dtype=float)
    out[in_transit] *= 1.0 - depth
    return out


def generate_synthetic_plato_like_lightcurve(
    time_days: np.ndarray,
    period_days: float,
    t0_days: float,
    rp_earth: float,
    r_star_solar: float,
    noise_ppm: float,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple box transit + Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    flux = np.ones_like(time_days, dtype=float)
    flux = apply_box_transit_to_flux(
        flux, time_days, period_days, t0_days, rp_earth, r_star_solar
    )
    noise = noise_ppm * 1e-6
    flux += rng.normal(0.0, noise, size=flux.shape)
    return time_days, flux


def run_bls_recovery(
    time_days: np.ndarray,
    flux: np.ndarray,
    period_min: float,
    period_max: float,
    n_periods: int = 2000,
    n_durations: int = 20,
) -> dict[str, Any]:
    """
    BLS period search (Astropy BoxLeastSquares).
    """
    from astropy.timeseries import BoxLeastSquares

    y = np.asarray(flux, dtype=float)
    y = y - np.nanmean(y)   #center flux around 0 to 
    t = np.asarray(time_days, dtype=float)
    noise = float(np.std(y)) or 1.0
    # dy: per-point uncertainty for BLS normalisation; uniform noise estimate
    bls = BoxLeastSquares(t, y, dy=np.full_like(y, noise))
    
    periods = np.linspace(period_min, period_max, n_periods)
    span = max(time_days[-1] - time_days[0], period_max * 2)
    # Keep lower bound small: long-period systems still have ~hours-long transits
    d_lo = 0.02
    d_hi = min(0.45 * period_max, 0.4 * span / max(len(t) / 200.0, 1.0)) #maxium duration is 45% of an orbit
    d_hi = float(np.clip(d_hi, d_lo + 0.01, 0.5))
    durations = np.linspace(d_lo, d_hi, n_durations)

    best_depth_snr = -np.inf
    best_snr_grid: np.ndarray | None = None
    best_depth_frac = 0.0
    best_p = float(period_min)
    best_dur = float(durations[len(durations) // 2])
    best_t0 = float(time_days[0])
    for d in durations:  # BLS is 2D (period × duration); outer loop fixes duration, inner scans period grid
        res = bls.power(periods, d)   #score all periods for given duration d
        ds = np.asarray(res.depth_snr)
        j = int(np.argmax(ds))   # take the best period of this duration
        if ds[j] > best_depth_snr:
            best_depth_snr = float(ds[j])
            best_p = float(periods[j])
            best_dur = float(d)
            best_snr_grid = ds.copy()
            best_depth_frac = max(0.0, float(np.asarray(res.depth)[j]))
            best_t0 = float(np.asarray(res.transit_time)[j])

    peak = max(best_depth_snr, 0.0)
    return {
        "period_recovered_days": best_p,
        "transit_time_days": best_t0,
        "bls_peak_snr": peak,
        "depth_snr": peak,
        "depth_frac": best_depth_frac,
        "duration_recovered_days": best_dur,
        "period_grid_days": periods,
        "snr_grid": best_snr_grid if best_snr_grid is not None else np.zeros(len(periods)),
    }


#sometime, the bls may peak at a harmonic of the true period, therefore need to refine the period by checking the harmonics of the recovered period
def refine_bls_harmonics(
    time_days: np.ndarray,
    flux: np.ndarray,
    coarse: dict[str, Any],
    period_min: float,
    period_max: float,
    max_num: int = 3,
) -> dict[str, Any]:
    """
    Re-evaluate BLS at integer-period ratio aliases (P, 2P, P/2, 2P/3, ...).
    Reduces common false positives where the global grid peaks on a harmonic.
    """
    from astropy.timeseries import BoxLeastSquares

    y = np.asarray(flux, dtype=float) - np.nanmean(flux)
    t = np.asarray(time_days, dtype=float)
    noise = float(np.std(y)) or 1.0
    # dy: per-point uncertainty for BLS normalisation; uniform noise estimate
    bls = BoxLeastSquares(t, y, dy=np.full_like(y, noise))
    p0 = float(coarse["period_recovered_days"])
    d0 = float(coarse["duration_recovered_days"])

    candidates: set[float] = {p0}   #set up the candidate periods to check
    for num in range(1, max_num + 1):
        for den in range(1, max_num + 1):
            pc = p0 * num / den               
            if period_min <= pc <= period_max:
                candidates.add(float(pc))

    # Neighbors of rational aliases (true P rarely equals exactly n/m × wrong peak)
    polished: set[float] = set()
    for pc in candidates:
        for fac in np.linspace(0.98, 1.02, 11):
            p2 = pc * fac
            if period_min <= p2 <= period_max:
                polished.add(float(p2))
    candidates |= polished #find peak near alias and take that

    best_power = -np.inf
    best_p = p0
    best_dur = d0
    best_depth_snr = -np.inf
    best_t0 = float(coarse.get("transit_time_days", t[0]))

    span = max(time_days[-1] - time_days[0], period_max * 2)
    d_lo = 0.02
    d_hi = min(0.45 * period_max, 0.4 * span / max(len(t) / 200.0, 1.0))
    d_hi = float(np.clip(d_hi, d_lo + 0.01, 0.5))

    d_scan = np.linspace(d_lo, d_hi, 14)
    for pc in candidates:
        for dur_try in d_scan:
            dur_try = float(dur_try)
            res = bls.power(np.asarray([pc]), dur_try)
            pw = float(np.asarray(res.power)[0])
            ds = float(np.asarray(res.depth_snr)[0])
            # Among harmonics, depth_snr tracks in-transit coherence better than raw power.
            if ds > best_depth_snr or (np.isclose(ds, best_depth_snr) and pw > best_power):
                best_power = pw
                best_p = pc
                best_dur = dur_try
                best_depth_snr = ds
                best_t0 = float(np.asarray(res.transit_time)[0])

    out = dict(coarse)
    out["period_recovered_days"] = best_p
    out["duration_recovered_days"] = best_dur
    # Refit t0 at the chosen harmonic — coarse t0 was fit at the coarse period and
    # mis-centers the phase fold when the harmonic pick jumps (P/2, 2P, ...).
    out["transit_time_days"] = best_t0
    out["bls_peak_snr"] = max(best_depth_snr, 0.0)
    out["depth_snr"] = max(best_depth_snr, 0.0)
    return out


def run_bls_recovery_with_refinement(
    time_days: np.ndarray,
    flux: np.ndarray,
    period_min: float,
    period_max: float,
    n_periods: int = 2000,
    n_durations: int = 20,
    refine_harmonics: bool = True,
) -> dict[str, Any]:
    """BLS grid search + optional harmonic alias pick."""
    coarse = run_bls_recovery(
        time_days, flux, period_min, period_max, n_periods=n_periods, n_durations=n_durations 
    )  #run bls on the coarse grid to get the best period and duration
    if not refine_harmonics:
        return coarse 
    return refine_bls_harmonics(time_days, flux, coarse, period_min, period_max) #in case the best period is a harmonic of the true period, refine it by checking the harmonics of the recovered period


def phased_angles(time_days: np.ndarray, period_days: float, t0_days: float) -> np.ndarray:
    """Return phase in [0, 1) with mid-transit at phase 0."""
    return ((time_days - t0_days) / period_days + 0.5) % 1.0 - 0.5


def phase_fold_lightcurve(
    flux: np.ndarray,
    time_days: np.ndarray,
    period_days: float,
    t0_days: float = 0.0,
    n_bins: int = 201,
) -> np.ndarray:
    """
    Phase-fold and bin a light curve to n_bins points.

    Uses phased_angles() so mid-transit (at t0_days) maps to phase 0 → center bin.
    Empty bins filled with global median. Output normalized to zero-median, unit std.

    Returns:
        np.ndarray of shape (n_bins,): normalized phase-folded flux.
    """
    phase = phased_angles(time_days, period_days, t0_days)  # [-0.5, 0.5) 
    sort_idx = np.argsort(phase)
    phase_s = phase[sort_idx]
    flux_s = flux[sort_idx]

    bin_edges = np.linspace(-0.5, 0.5, n_bins + 1) #use -0.5 to 0.5 to center the bins around transit dip
    binned = np.empty(n_bins)
    for i in range(n_bins):
        mask = (phase_s >= bin_edges[i]) & (phase_s < bin_edges[i + 1])
        binned[i] = np.median(flux_s[mask]) if mask.any() else np.nan #take median of the flux values in the bin, or NaN if no points in bin

    # Fill NaN bins with global median
    global_med = np.nanmedian(binned)
    binned = np.where(np.isnan(binned), global_med, binned)

    # Normalize to zero-median, unit std
    med = np.median(binned)
    std = np.std(binned)
    if std > 1e-12:
        binned = (binned - med) / std  # z-score normalize: CNN expects zero-mean unit-variance input
    else:
        binned = binned - med  # flat lightcurve guard: skip division to avoid NaN/inf
    return binned




# load cache of CNN classifier, MLP ranker, and scaler
def load_cnn_models(
    cnn_path: str = "models/cnn_classifier.pt",
    mlp_path: str = "models/mlp_ranker.pt",
    scaler_path: str = "models/mlp_scaler.pkl",
) -> tuple:
    """Load saved CNN classifier, MLP ranker, and StandardScaler. Returns (cnn, mlp, scaler)."""
    from ml_pipeline import load_cnn_models as _load
    return _load(cnn_path=cnn_path, mlp_path=mlp_path, scaler_path=scaler_path)


if __name__ == "__main__":
    inj, res = analyze_systems()
    print("=== Injected parameters ===")
    print(inj.to_string(index=False))
    print("\n=== Analysis ===")
    print(res.sort_values("rank_score", ascending=False).to_string(index=False))
