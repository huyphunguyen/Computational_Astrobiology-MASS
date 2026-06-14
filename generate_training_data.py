"""
Stratified PSLS training data generator.

Tiers (real PLATO simulations via grid_plato.hdf5):
  easy        P=5-30d,    Rp=2.0-4.0 Re, noise_mult=1.0      → n_easy curves
  medium      P=30-150d,   Rp=1.5-3.0 Re, noise_mult=1.0-2.0  → n_medium curves
  hard        P=150-270d,  Rp=0.8-2.0 Re, noise_mult=2.0-5.0  → n_hard curves
  no_transit  no planet injected                               → n_no_transit curves
  cold        S_earth-controlled cold-regime coverage          → n_cold curves

Output CSV columns: has_transit, tier, + FEATURE_COLS.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from habitable_zone_pipeline import (
    run_bls_recovery_with_refinement,
    stellar_luminosity_solar,
    equilibrium_temperature_from_au,
    habitability_rank_score,
    phase_fold_lightcurve,
)
import psls_runner as pr
from transit_helpers import load_psls_dat, detrend_flux

# All feature columns written to the CSV. ml_pipeline.FEATURE_COLS_A/_B select
# subsets; period_error_pct and noise_mult are kept as metadata only (leakage —
# truth-derived / simulator-only), never fed to models.
FEATURE_COLS = [
    "depth_snr", "period_recovered_d", "period_error_pct",
    "n_transits", "depth_ppm", "duration_d",
    "S_earth", "Teq_K", "Rp_Rearth", "Teff_K", "Rstar_Rsun", "noise_mult",
]

_P_MIN     = 1.0               # BLS search lower bound
_BLS_NPER  = 2000              # BLS period grid points
_BASE_NOISE_PPM = 80.0  # base photometric noise at noise_mult=1

# Star-type parameter ranges [Teff_min, Teff_max, Rstar_min, Rstar_max]
_STAR_TYPES = {
    "K": (4500, 5200, 0.6, 0.9),
    "G": (5200, 6000, 0.9, 1.1),
    "F": (6000, 6500, 1.1, 1.3),
}

# ---------------------------------------------------------------------------
# PSLS path — real PLATO simulations via the stellar grid (grid_plato.hdf5)
# ---------------------------------------------------------------------------

# 270-day baseline (3 PLATO quarters) to MATCH the 12 sim_systems test set.
_PSLS_QUARTER = [90.0, 90.0, 90.0]
_PSLS_BASELINE = 270.0
_PSLS_P_MAX = 0.48 * _PSLS_BASELINE  # ~130 d; BLS upper bound for a 270 d window
_PSLS_RAW_DIR = "training_data/psls_raw"
_STAR_ID_BASE = 1_000_000  # per-sample StarID = base + global index
_N_BINS = 201              # CNN phase-folded length (TransitCNN input)
_DETREND_WIN_D = 1.0       # running-median window — MUST match notebook cell-006/011


def _logg_from_teff(teff: float) -> float:
    """Rough main-sequence log g vs Teff (K~4.55 → F~4.15); grid match snaps to a real model."""
    frac = (teff - 4500.0) / (6500.0 - 4500.0)
    return float(np.clip(4.55 - 0.40 * frac, 4.0, 4.6))


def _sample_tier_params(tier: str, rng: np.random.Generator) -> tuple[float, float, float]:
    """Return (period_d, rp_re, noise_mult) for a tier — shared by tabular + CNN."""
    if tier == "easy":
        return float(rng.uniform(5.0, 30.0)), float(rng.uniform(2.0, 4.0)), 1.0
    if tier == "medium":
        return float(rng.uniform(30.0, 150.0)), float(rng.uniform(1.5, 3.0)), float(rng.uniform(1.0, 2.0))
    if tier == "hard":
        return float(rng.uniform(150.0, 270.0)), float(rng.uniform(0.8, 2.0)), float(rng.uniform(2.0, 5.0))
    return float(rng.uniform(5.0, 270.0)), 0.0, float(rng.uniform(1.0, 3.0))  # no_transit


def _generate_one_sample_psls(
    tier: str,
    rng: np.random.Generator,
    global_idx: int,
    raw_dir: str = _PSLS_RAW_DIR,
    n_bins: int = _N_BINS,
) -> tuple[dict, np.ndarray]:
    """
    Run ONE PSLS curve and derive BOTH the tabular feature row and the phase-folded
    CNN array from the same flux. Returns (row_dict, folded_array).
    """
    bond_albedo = float(rng.uniform(0.1, 0.4))
    has_transit = int(tier != "no_transit")

    # --- Star: sample Teff (K/G/F), snap to a physical grid model for Mass/Radius ---
    stype = rng.choice(["K", "G", "F"])
    tmin, tmax, _, _ = _STAR_TYPES[stype]
    teff_nom = float(rng.uniform(tmin, tmax))
    teff, logg, mstar, rstar = pr.match_grid_model(teff_nom, _logg_from_teff(teff_nom))

    if tier == "cold":
        # S_earth-CONTROLLED: the period-based tiers never produce S_earth < ~0.67, so the
        # rankers extrapolate (over-score cold planets). Here we pick a low target flux and
        # invert S = L/a^2 -> a -> period, guaranteeing cold-regime coverage. Many of these
        # are long-period (few/no transits in 270 d) — fine: the RANKER uses physics features
        # only, so an undetectable cold planet is still a valid ranker training row.
        rp = float(rng.uniform(0.8, 2.0))
        noise_mult = float(rng.uniform(1.0, 2.0))
        s_target = float(rng.uniform(0.15, 0.9))
        lum = stellar_luminosity_solar(rstar, teff)
        a_au = float(np.sqrt(lum / s_target))
        period = float(365.25 * np.sqrt(a_au ** 3 / mstar))
    else:
        period, rp, noise_mult = _sample_tier_params(tier, rng)
        a_au = float((period / 365.25) ** (2.0 / 3.0) * mstar ** (1.0 / 3.0))

    # --- Write config + run PSLS (idempotent; skips if .dat exists) ---
    star_id = _STAR_ID_BASE + global_idx
    yaml_path = Path(raw_dir) / f"{star_id:010d}.yaml"
    pr.write_psls_yaml(
        yaml_path,
        teff=teff, logg=logg, period_d=period, rp_re=rp, a_au=a_au,
        star_id=star_id, master_seed=int(rng.integers(0, 2**31)),
        noise_mult=noise_mult, transit_enable=bool(has_transit),
        quarter_duration=_PSLS_QUARTER,
    )
    dat = pr.run_psls(yaml_path, raw_dir)
    time, flux_raw = load_psls_dat(str(dat))
    # Detrend (running median) to match the notebook test pipeline (cell-006/011): PSLS
    # curves carry granulation/activity trends that must be removed before BLS + folding.
    flux = detrend_flux(time, flux_raw, window_days=_DETREND_WIN_D)
    baseline = float(time[-1] - time[0])

    # --- BLS recovery on the detrended curve ---
    rec = run_bls_recovery_with_refinement(time, flux, _P_MIN, _PSLS_P_MAX, n_periods=_BLS_NPER)
    p_rec = float(rec["period_recovered_days"])
    p_err = abs(p_rec - period) / period * 100.0 if has_transit else 0.0
    depth_ppm = float(rec["depth_frac"]) * 1e6
    n_transits = max(1, int(baseline / p_rec)) if p_rec > 0 else 0

    # --- Physical features (actual grid Rstar) ---
    lum = stellar_luminosity_solar(rstar, teff)
    s_earth = lum / (a_au ** 2) if a_au > 0 else 0.0
    teq = equilibrium_temperature_from_au(teff, rstar, a_au, albedo=bond_albedo)
    rp_for_score = max(rp, 0.5)
    rank_label = float(habitability_rank_score(s_earth, rp_for_score, teff, snr_proxy=0.0)["rank_score"])

    # --- Phase-folded CNN array from the SAME flux ---
    if has_transit:
        p_fold = p_rec if p_rec > 0 else period
        t0_fold = float(rec["transit_time_days"])
    else:
        p_fold = float(rng.uniform(5.0, _PSLS_P_MAX))
        t0_fold = 0.0
    folded = phase_fold_lightcurve(flux, time, period_days=p_fold, t0_days=t0_fold, n_bins=n_bins)

    row = {
        "has_transit":        has_transit,
        "tier":               tier,
        "depth_snr":          float(rec["depth_snr"]),
        "bls_peak_snr":       float(rec["bls_peak_snr"]),
        "period_recovered_d": p_rec,
        "period_error_pct":   p_err,
        "n_transits":         n_transits,
        "depth_ppm":          depth_ppm,
        "duration_d":         float(rec["duration_recovered_days"]),
        "S_earth":            s_earth,
        "Teq_K":              teq,
        "Rp_Rearth":          rp,
        "Teff_K":             teff,
        "Rstar_Rsun":         rstar,
        "noise_mult":         noise_mult,
        "rank_score_label":   rank_label,
        "period_d":           period,
        "a_AU":               a_au,
        "bond_albedo":        bond_albedo,
        "star_id":            star_id,
    }
    return row, folded.astype(np.float32)


def generate_training_dataset_psls(
    n_easy: int = 80,
    n_medium: int = 100,
    n_hard: int = 70,
    n_no_transit: int = 100,
    n_cold: int = 0,
    seed: int = 42,
    n_bins: int = _N_BINS,
    raw_dir: str = _PSLS_RAW_DIR,
    csv_path: str = "training_data/training_labels.csv",
    folded_path: str = "training_data/cnn_phase_folded.npy",
    labels_path: str = "training_data/cnn_labels.npy",
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Generate the stratified dataset with PSLS. One curve per sample feeds both the
    tabular row (CSV) and the phase-folded CNN array (npy), kept index-aligned.
    Writes training_labels.csv, cnn_phase_folded.npy (X), cnn_labels.npy (y=has_transit).

    Resumable: per-sample RNG is seeded by (seed + global_idx); rows already present
    in csv_path (matched by star_id) are skipped, and PSLS .dat files are reused.
    """
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    # NOTE: cold appended LAST so the existing 350 samples keep their global_idx (and thus
    # star_id + cached .dat); cold takes the new indices 350.. -> no collision, resumable.
    tasks: list[tuple[str, int]] = []
    for tier, count in [("easy", n_easy), ("medium", n_medium), ("hard", n_hard),
                        ("no_transit", n_no_transit), ("cold", n_cold)]:
        tasks += [(tier, i) for i in range(count)]

    # Resume: load any already-computed rows/folded.
    done_ids: set[int] = set()
    rows: list[dict] = []
    folded: list[np.ndarray] = []
    if Path(csv_path).exists() and Path(folded_path).exists():
        prev = pd.read_csv(csv_path)
        prev_folded = np.load(folded_path)
        if "star_id" in prev.columns and len(prev) == len(prev_folded):
            rows = prev.to_dict("records")
            folded = list(prev_folded)
            done_ids = set(int(s) for s in prev["star_id"])
            print(f"[resume] {len(done_ids)} samples already done")

    for global_idx, (tier, _) in enumerate(tasks):
        star_id = _STAR_ID_BASE + global_idx
        if star_id in done_ids:
            continue
        rng = np.random.default_rng(seed + global_idx)
        try:
            row, fold = _generate_one_sample_psls(tier, rng, global_idx, raw_dir=raw_dir, n_bins=n_bins)
        except Exception as exc:  # noqa: BLE001 — log and continue so one bad sim doesn't kill the run
            print(f"  [skip] idx={global_idx} tier={tier}: {exc}")
            continue
        rows.append(row)
        folded.append(fold)
        # Checkpoint each sample so a crash loses at most one curve.
        df_ck = pd.DataFrame(rows)
        df_ck.to_csv(csv_path, index=False)
        np.save(folded_path, np.array(folded, dtype=np.float32))
        np.save(labels_path, df_ck["has_transit"].to_numpy(dtype=np.int32))
        if (global_idx + 1) % 10 == 0:
            print(f"  {global_idx + 1}/{len(tasks)} done (tier={tier})", flush=True)

    df = pd.DataFrame(rows)
    X = np.array(folded, dtype=np.float32)
    return df, X


if __name__ == "__main__":
    import argparse, pathlib
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="training_data/training_labels.csv")
    parser.add_argument("--folded-out", default="training_data/cnn_phase_folded.npy")
    parser.add_argument("--labels-out", default="training_data/cnn_labels.npy")
    parser.add_argument("--n-easy",       type=int, default=80)
    parser.add_argument("--n-medium",     type=int, default=100)
    parser.add_argument("--n-hard",       type=int, default=70)
    parser.add_argument("--n-no-transit", type=int, default=100)
    parser.add_argument("--n-cold",       type=int, default=0,
                        help="S_earth-controlled cold-regime curves (fills the cold flux gap)")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print("Generating PSLS training dataset (one curve -> tabular + CNN)...")
    df, X = generate_training_dataset_psls(
        n_easy=args.n_easy, n_medium=args.n_medium, n_hard=args.n_hard,
        n_no_transit=args.n_no_transit, n_cold=args.n_cold, seed=args.seed,
        csv_path=args.out, folded_path=args.folded_out, labels_path=args.labels_out,
    )
    print(f"Saved {len(df)} rows to {args.out}; folded {X.shape} to {args.folded_out}")
    print(f"Transit: {df['has_transit'].sum()} | No-transit: {(df['has_transit']==0).sum()}")
    print(df.groupby("tier")["depth_snr"].describe()[["count", "mean", "50%"]])
