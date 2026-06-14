"""
psls_runner.py — drive PSLS (psls-1.9) to generate realistic PLATO light curves
using the stellar grid (grid_plato.hdf5).

Pipeline per curve:
  1. match_grid_model(teff, logg)        -> physical (teff, logg, mass, radius) PSLS will use
  2. write_psls_yaml(...)                -> a self-consistent config (grid model, real noise)
  3. run_psls(yaml, out_dir)             -> runs psls-1.9/psls.py, returns the .dat path
  4. (caller) transit_helpers.load_psls_dat(dat) -> (time_days, flux)

Key constraints (see psls-1.9/psls.py):
  - ModelType 'grid' + ModelName 'grid_plato' -> search_model_hdf5(teff, logg) chi2 match.
  - Oscillations.SurfaceEffects MUST be 0, else cool stars (K-dwarfs) fall outside the
    surface-effect polygon and PSLS sys.exit(1).
  - Transit.PlanetRadius is in JUPITER radii.
  - Systematics.Table / ModelDir must resolve from the working dir -> we use absolute paths.
  - Star params actually used are echoed to <out>/<StarID>.txt (authoritative).
"""

from __future__ import annotations


import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
GRID_PATH = PROJECT_ROOT / "grid_plato.hdf5"
PSLS_SCRIPT = PROJECT_ROOT / "psls-1.9" / "psls.py"
SYSTEMATICS_TABLE = PROJECT_ROOT / "psls-1.9" / "systematics" / "PLATO_systematics_BOL_V2.npy"
TEMPLATE_YAML = PROJECT_ROOT / "sim_systems" / "B2_G_hz.yaml"

# CGS solar constants used by the grid (mass in g, radius in cm) — matches psls sls.msun/rsun.
MSUN_G = 1.989e33
RSUN_CM = 6.957e10
# 1 Earth radius in Jupiter radii (6371 km / 71492 km).
RE_TO_RJUP = 6371.0 / 71492.0

# Base PLATO_SCALING noise-to-signal ratio from the template (NSR at noise_mult=1).
BASE_NSR = 73.0


def match_grid_model(
    teff: float,
    logg: float,
    grid_path: str | os.PathLike = GRID_PATH,
    dteff: float = 15.0,
    dlogg: float = 0.01,
) -> tuple[float, float, float, float]:
    """
    Replicate psls.search_model_hdf5: chi2-match (teff, logg) to the nearest main-sequence
    grid model so labels use the SAME star PSLS will pick at run time.

    Returns (teff_actual, logg_actual, mass_msun, radius_rsun).
    """
    best_chi2 = np.inf
    best = None
    with h5py.File(grid_path, "r") as f:
        for key in f.keys():
            if key == "license":
                continue
            g = f[key]["global"]
            t = np.asarray(g["teff"])
            lg = np.asarray(g["logg"])
            m = np.asarray(g["mass"])
            r = np.asarray(g["radius"])
            xc = np.asarray(g["Xc"])
            sel = xc > 1e-3  # ES='ms': main sequence
            chi2 = ((t - teff) / dteff) ** 2 + ((lg - logg) / dlogg) ** 2
            chi2 = np.where(sel, chi2, np.inf)
            j = int(np.argmin(chi2))
            if chi2[j] < best_chi2:
                best_chi2 = chi2[j]
                best = (float(t[j]), float(lg[j]), float(m[j] / MSUN_G), float(r[j] / RSUN_CM))
    if best is None:
        raise RuntimeError("no main-sequence grid model matched")
    return best


def write_psls_yaml(
    path: str | os.PathLike,
    *,
    teff: float,
    logg: float,
    period_d: float,
    rp_re: float,
    a_au: float,
    star_id: int,
    master_seed: int,
    noise_mult: float = 1.0,
    transit_enable: bool = True,
    quarter_duration: list[float] | None = None,
) -> Path:
    """
    Render a PSLS config from the B2 template, switched to the hdf5 grid model.
    Absolute paths are injected so the run works from any cwd.
    """
    cfg = yaml.safe_load(open(TEMPLATE_YAML))

    # --- Star: grid model, requested (Teff, logg) ---
    cfg["Star"]["ModelType"] = "grid"
    cfg["Star"]["ModelName"] = "grid_plato"
    cfg["Star"]["ModelDir"] = str(PROJECT_ROOT) + "/"
    cfg["Star"]["Teff"] = float(teff)
    cfg["Star"]["Logg"] = float(logg)
    cfg["Star"]["ID"] = int(star_id)

    # SurfaceEffects=0 is mandatory for cool stars (polygon crash otherwise).
    cfg["Oscillations"]["SurfaceEffects"] = 0

    # --- Instrument: absolute systematics table, scaled noise ---
    cfg["Instrument"]["Systematics"]["Table"] = str(SYSTEMATICS_TABLE)
    cfg["Instrument"]["RandomNoise"]["NSR"] = float(BASE_NSR * noise_mult)

    # --- Observation ---
    if quarter_duration is not None:
        cfg["Observation"]["QuarterDuration"] = list(quarter_duration)
    cfg["Observation"]["MasterSeed"] = int(master_seed)

    # --- Transit (PlanetRadius in Jupiter radii) ---
    cfg["Transit"]["Enable"] = 1 if transit_enable else 0
    cfg["Transit"]["PlanetRadius"] = float(rp_re * RE_TO_RJUP)
    cfg["Transit"]["OrbitalPeriod"] = float(period_d)
    cfg["Transit"]["PlanetSemiMajorAxis"] = float(a_au)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
    return out


def run_psls(yaml_path: str | os.PathLike, out_dir: str | os.PathLike) -> Path:
    """
    Run psls-1.9/psls.py on a config and return the produced .dat path.
    Idempotent: if the expected .dat already exists it is returned without rerunning.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(yaml_path))
    star_id = int(cfg["Star"]["ID"])
    dat = out_dir / f"{star_id:010d}.dat"
    if dat.exists():
        return dat

    env = dict(os.environ, MPLBACKEND="Agg")
    proc = subprocess.run(
        [sys.executable, str(PSLS_SCRIPT), "-o", str(out_dir), str(yaml_path)],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    if not dat.exists():
        raise RuntimeError(
            f"PSLS did not produce {dat}\nSTDOUT tail:\n{proc.stdout[-2000:]}\n"
            f"STDERR tail:\n{proc.stderr[-2000:]}"
        )
    return dat


def read_psls_star_params(dat_path: str | os.PathLike) -> dict[str, float]:
    """
    Read the authoritative star params PSLS actually used from the sidecar <StarID>.txt.
    Returns dict with teff, logg, mass, radius (solar units).
    """
    txt = Path(dat_path).with_suffix(".txt")
    params: dict[str, float] = {}
    for line in open(txt):
        if "mass =" in line and "radius =" in line:
            # e.g. " mass = 1.04373,  teff = 5747.12,  radius = 1.06505,  logg = 4.40155"
            for part in line.split(","):
                if "=" in part:
                    k, v = part.split("=")
                    try:
                        params[k.strip()] = float(v)
                    except ValueError:
                        pass
    return params
