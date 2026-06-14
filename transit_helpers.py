"""
Provides:
  - load_systems_from_yaml()  : build SYSTEMS dict by reading sim_systems/*.yaml
  - load_psls_dat()           : read & downsample a PSLS .dat file
  - detrend_flux()            : running-median detrend
  - transit_depth_ppm()       : (Rp/Rs)^2 × 1e6
  - verdict()                 : Promising / Marginal / Non-promising label
"""

from __future__ import annotations

import glob
import os

import numpy as np
import yaml
from scipy.ndimage import median_filter

# ── Constants ──────────────────────────────────────────────────────────────────
RJ_TO_RE   = 11.209          # Jupiter radii → Earth radii
BLS_STRIDE = 72              # 25 s × 72 = 30 min cadence


def _rstar_from_teff(teff: float) -> float:
    """Approximate main-sequence R_star [R_sun] from T_eff using simple lookup."""
    if teff >= 6000:
        return 1.30   # F-dwarf
    if teff >= 5200:
        return 1.00   # G-dwarf
    if teff >= 4500:
        return 0.72   # K-dwarf
    return 0.60       # late K / early M


def load_systems_from_yaml(sim_dir: str) -> dict:
    """
    Build the SYSTEMS ground-truth dict by parsing every *.yaml in sim_dir.

    Each entry contains:
      p_inj, a_au, rp_rj, rp_re, teff, rstar, mag,
      star_type, science_case

    Fields are read directly from YAML keys:
      Transit.OrbitalPeriod      → p_inj
      Transit.PlanetSemiMajorAxis → a_au
      Transit.PlanetRadius        → rp_rj   (PSLS units = R_Jupiter)
      Star.Teff                   → teff
      Star.Mag                    → mag
      (rstar inferred from Teff)
    """
    yaml_paths = sorted(glob.glob(os.path.join(sim_dir, '*.yaml')))
    if not yaml_paths:
        raise FileNotFoundError(f'No YAML files found in {sim_dir}')

    systems = {}
    for path in yaml_paths:
        cfg_name = os.path.splitext(os.path.basename(path))[0]
        with open(path) as fh:
            cfg = yaml.safe_load(fh)

        star    = cfg.get('Star', {})
        transit = cfg.get('Transit', {})

        teff   = float(star.get('Teff', 5772.0))
        mag    = float(star.get('Mag', 10.0))
        rstar  = _rstar_from_teff(teff)

        p_inj  = float(transit.get('OrbitalPeriod', 0.0))
        a_au   = float(transit.get('PlanetSemiMajorAxis', 0.0))
        rp_rj  = float(transit.get('PlanetRadius', 0.0))

        meta = cfg.get('Metadata', {})
        star_type    = meta.get('star_type', 'Unknown')
        science_case = meta.get('science_case', 'Unknown')

        systems[cfg_name] = {
            'p_inj':        p_inj,
            'a_au':         a_au,
            'rp_rj':        rp_rj,
            'rp_re':        rp_rj * RJ_TO_RE,
            'teff':         teff,
            'rstar':        rstar,
            'mag':          mag,
            'star_type':    star_type,
            'science_case': science_case,
        }

    return systems


# ── Light-curve helpers ────────────────────────────────────────────────────────

def load_psls_dat(path: str, stride: int = BLS_STRIDE):
    """
    Read a PSLS .dat file, keep flag==0 rows, downsample by stride.
    Returns (time_days, flux_relative).
    """
    data = np.genfromtxt(path, comments='#')
    mask = data[:, 2] == 0           # col 2 is PSLS quality flag; 0 = good cadence
    data_down = data[mask][::stride]  # downsample: keep every 72nd cadence → 30 min effective sampling
    time_days = data_down[:, 0] / 86400.0  # PSLS time in seconds → days
    flux      = 1.0 + data_down[:, 1] * 1e-6  # PSLS flux in ppm offset → relative flux centered on 1.0
    return time_days, flux


def detrend_flux(
    time_days: np.ndarray,
    flux: np.ndarray,
    window_days: float = 1.0,
) -> np.ndarray:
    """
    Remove stellar variability with a running median filter.
    window_days should be > transit duration (~0.2 d) but << orbital period.
    Divides flux by the smoothed baseline to preserve transit dips.
    """
    dt = float(np.median(np.diff(time_days)))  # median cadence in days; robust to gaps
    kernel = max(int(round(window_days / dt)) | 1, 3) #enforce odd kernel size for median_filter
    if kernel % 2 == 0:
        kernel += 1
    baseline = median_filter(flux, size=kernel, mode='reflect')  # reflect pads edges to avoid boundary artifacts
    return flux / baseline  # division preserves transit dip depth relative to the smoothed stellar baseline


def transit_depth_ppm(rp_re: float, rstar_rsun: float) -> float:
    """(Rp/Rs)^2 × 1e6  [ppm]."""
    R_EARTH_M = 6.371e6
    R_SUN_M   = 6.957e8
    return float(((rp_re * R_EARTH_M) / (rstar_rsun * R_SUN_M)) ** 2 * 1e6)  # (Rp/Rs)² × 1e6: fractional stellar disk blocked, expressed in ppm


def verdict(detected: bool, in_hz: bool, near_hz: bool = False) -> str:
    """Return 'Promising', 'Marginal', or 'Non-promising'."""
    if detected and (in_hz or near_hz):  # AND: transit signal required; HZ location upgrades to Promising
        return 'Promising'
    if detected:
        return 'Marginal'  # transit found but outside HZ — scientifically interesting, not HZ candidate
    return 'Non-promising'


def plot_bls_diagnostic(
    config: str,
    time_days: np.ndarray,
    flux_det: np.ndarray,
    rec: dict,
    truth: dict | None = None,
) -> None:
    """
    3-panel diagnostic plot for a BLS detection.

    Panels:
      1. Detrended light curve with predicted transit times.
      2. BLS depth_snr periodogram (log period axis).
      3. Phase-folded light curve with binned median and box model.

    Parameters
    ----------
    config     : system name used as figure title
    time_days  : time array (days)
    flux_det   : detrended relative flux (around 1.0)
    rec        : dict returned by run_bls_recovery_with_refinement
    truth      : optional ground-truth dict with 'p_inj' key
    """
    import matplotlib.pyplot as plt

    p_rec = float(rec["period_recovered_days"])
    dur_rec = float(rec["duration_recovered_days"])
    period_grid = np.asarray(rec["period_grid_days"])
    snr_grid = np.asarray(rec["snr_grid"])
    depth_frac = float(rec.get("depth_frac", 0.0))
    bls_peak_snr = float(rec.get("bls_peak_snr", 0.0))

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(
        f"{config}  |  P_rec = {p_rec:.2f} d  |  BLS SNR = {bls_peak_snr:.1f}",
        fontsize=13,
    )
    

    #  Panel 1: detrended light curve 
    ax = axes[0]
    ax.plot(time_days, flux_det, "k.", ms=1, alpha=0.4, rasterized=True)
    t0 = float(rec.get("transit_time_days", time_days[0] + p_rec / 2.0))
    for tt in np.arange(t0, time_days[-1], p_rec):
        ax.axvline(tt, color="steelblue", alpha=0.35, lw=0.8) # vertical lines at predicted transit times
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Relative flux")
    ax.set_title("Detrended light curve")

    #  Panel 2: BLS periodogram 
    ax = axes[1]
    ax.semilogx(period_grid, snr_grid, "k-", lw=0.8)
    ax.axvline(p_rec, color="crimson", lw=1.5, label=f"P_rec = {p_rec:.2f} d")
    if truth is not None and "p_inj" in truth:
        ax.axvline(
            truth["p_inj"],
            color="forestgreen",
            lw=1.5,
            ls="--",
            label=f"P_inj = {truth['p_inj']:.2f} d",
        )
    ax.axhline(5.0, color="orange", lw=1.0, ls=":", label="SNR = 5 threshold")
    ax.set_xlabel("Period [days]")
    ax.set_ylabel("BLS depth SNR")
    ax.set_title("BLS periodogram")
    ax.legend(fontsize=8)

    #  Panel 3: phase-folded light curve — ZOOMED on the transit
    ax = axes[2]
    t0_fold = float(rec.get("transit_time_days", time_days[0]))
    phase = ((time_days - t0_fold) / p_rec) % 1.0
    phase = np.where(phase > 0.5, phase - 1.0, phase)  # center transit at phase 0

    # A transit is usually MUCH narrower than 1/20 of the period, so a 20-bin fold over the
    # full phase washes the dip out (each bin is mostly out-of-transit) and the box model
    # looks deeper than the data. Instead zoom to a few transit-widths around phase 0 and
    # bin finely WITHIN that window so the median resolves the true depth.
    half_phase = (dur_rec / (2.0 * p_rec)) if dur_rec > 0 else 0.01
    zoom = float(min(max(8.0 * half_phase, 0.02), 0.5))  # half-window in phase units
    n_zbins = 41
    zbins = np.linspace(-zoom, zoom, n_zbins + 1)
    zcent = 0.5 * (zbins[:-1] + zbins[1:])
    zmed = np.array([
        np.nanmedian(flux_det[(phase >= zbins[i]) & (phase < zbins[i + 1])])
        if ((phase >= zbins[i]) & (phase < zbins[i + 1])).any() else np.nan
        for i in range(n_zbins)
    ])

    ax.plot(phase, flux_det, "k.", ms=2, alpha=0.25, rasterized=True)
    ax.plot(zcent, zmed, "o-", color="steelblue", ms=3, lw=1.2, label=f"{n_zbins}-bin median")

    #draw the transit box
    if depth_frac > 0:
        box_ph = np.array([-zoom, -half_phase, -half_phase, half_phase, half_phase, zoom])
        box_fl = np.array([1.0, 1.0, 1.0 - depth_frac, 1.0 - depth_frac, 1.0, 1.0])
        ax.plot(box_ph, box_fl, "r-", lw=1.5, label="Box model")

    # Zoom axes; y-limits from the in-window scatter so the dip is clearly visible.
    ax.set_xlim(-zoom, zoom)
    in_win = np.abs(phase) <= zoom
    yv = flux_det[in_win]
    if np.isfinite(yv).any():
        lo = min(float(np.nanpercentile(yv, 2)), 1.0 - depth_frac * 1.5)
        hi = float(np.nanpercentile(yv, 98))
        pad = 0.15 * (hi - lo + 1e-9)
        ax.set_ylim(lo - pad, hi + pad)

    ax.set_xlabel("Phase (zoomed on transit)")
    ax.set_ylabel("Relative flux")
    ax.set_title(f"Phase-folded  (P = {p_rec:.2f} d,  dur = {dur_rec*24:.1f} h)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()
