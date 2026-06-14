"""
run_psls_heatmap.py — detectability heatmap from REAL PSLS curves.

Builds a 10x10 (period x Rp) grid on a fixed G-dwarf (Teff 5750), simulates each cell
with PSLS, recovers it with BLS, scores transit_prob with the trained RF classifier
(Model A), and saves the probability map to sim_systems/cache/heatmap_prob_map.npz.

The notebook cell-022 loads this npz (same keys/grids) and plots it.

Run from the project root:
    python3 sim_systems/run_psls_heatmap.py            # full 10x10 = 100 sims
    python3 sim_systems/run_psls_heatmap.py --smoke    # 2x2 = 4 sims (validation)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import psls_runner as pr  # noqa: E402
from transit_helpers import load_psls_dat, detrend_flux  # noqa: E402
from habitable_zone_pipeline import (  # noqa: E402
    run_bls_recovery_with_refinement,
    stellar_luminosity_solar,
    equilibrium_temperature_from_au,
)
from ml_pipeline import load_models, predict_transit_prob, FEATURE_COLS_A  # noqa: E402

# Fixed G-dwarf host + 270 d baseline (matches the sim_systems test set).
TEFF_FIXED = 5750.0
QUARTER = [90.0, 90.0, 90.0]
P_MIN, P_MAX, BLS_NPER = 1.0, 130.0, 2000
STAR_ID_BASE = 2_000_000

CACHE_DIR = ROOT / "sim_systems" / "cache"
RAW_DIR = CACHE_DIR / "psls_heatmap"
NPZ_PATH = CACHE_DIR / "heatmap_prob_map.npz"


def build_grids(smoke: bool) -> tuple[np.ndarray, np.ndarray]:
    n = 2 if smoke else 10
    periods_grid = np.linspace(5.0, 125.0, n)   # within 0.48*270 d so BLS stays valid
    rp_grid = np.linspace(0.4, 4.0, n)
    return periods_grid, rp_grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2x2 grid for validation")
    args = ap.parse_args()

    periods_grid, rp_grid = build_grids(args.smoke)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Fixed star: snap to a physical grid model for Mass/Radius.
    from generate_training_data import _logg_from_teff
    teff, logg, mstar, rstar = pr.match_grid_model(TEFF_FIXED, _logg_from_teff(TEFF_FIXED))
    print(f"Host: Teff={teff:.0f} logg={logg:.3f} M={mstar:.3f} R={rstar:.3f}")

    clf, _ = load_models()  # models/rf_classifier.pkl

    # Resume: keep already-computed cells (NaN = not yet done).
    prob_map = np.full((len(rp_grid), len(periods_grid)), np.nan)
    if NPZ_PATH.exists():
        prev = np.load(NPZ_PATH)
        if (np.array_equal(prev["periods_grid"], periods_grid)
                and np.array_equal(prev["rp_grid"], rp_grid)):
            prob_map = prev["prob_map"]

    for i, rp in enumerate(rp_grid):
        for j, period in enumerate(periods_grid):
            if np.isfinite(prob_map[i, j]):
                continue
            a_au = float((period / 365.25) ** (2.0 / 3.0) * mstar ** (1.0 / 3.0))
            star_id = STAR_ID_BASE + i * len(periods_grid) + j
            yaml_path = RAW_DIR / f"{star_id:010d}.yaml"
            pr.write_psls_yaml(
                yaml_path, teff=teff, logg=logg, period_d=period, rp_re=float(rp),
                a_au=a_au, star_id=star_id, master_seed=star_id,
                noise_mult=1.0, transit_enable=True, quarter_duration=QUARTER,
            )
            dat = pr.run_psls(yaml_path, RAW_DIR)
            time, flux_raw = load_psls_dat(str(dat))
            flux = detrend_flux(time, flux_raw, window_days=1.0)  # match notebook pipeline
            baseline = float(time[-1] - time[0])
            rec = run_bls_recovery_with_refinement(time, flux, P_MIN, P_MAX, n_periods=BLS_NPER)

            p_rec = float(rec["period_recovered_days"])
            n_tr = max(1, int(baseline / p_rec)) if p_rec > 0 else 0
            lum = stellar_luminosity_solar(rstar, teff)
            s_earth = lum / (a_au ** 2)
            teq = equilibrium_temperature_from_au(teff, rstar, a_au)
            feat = {
                "depth_snr":          rec["depth_snr"],
                "period_recovered_d": p_rec,
                "n_transits":         n_tr,
                "depth_ppm":          rec["depth_frac"] * 1e6,
                "duration_d":         rec["duration_recovered_days"],
                "S_earth":            s_earth,
                "Teq_K":              teq,
                "Rp_Rearth":          float(rp),
                "Teff_K":             teff,
                "Rstar_Rsun":         rstar,
            }
            row = np.array([[feat[c] for c in FEATURE_COLS_A]])
            prob_map[i, j] = float(predict_transit_prob(clf, row)[0])
            np.savez(NPZ_PATH, prob_map=prob_map, periods_grid=periods_grid, rp_grid=rp_grid)
        print(f"  row {i + 1}/{len(rp_grid)} done (Rp={rp:.2f} Re)", flush=True)

    print(f"Saved heatmap -> {NPZ_PATH}  (prob_map {prob_map.shape}, "
          f"range [{np.nanmin(prob_map):.3f}, {np.nanmax(prob_map):.3f}])")


if __name__ == "__main__":
    main()
