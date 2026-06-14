"""
run_psls_training.py — generate the 350-curve PSLS training set (background entry point).

Thin wrapper over generate_training_data.generate_training_dataset_psls with the default
350-curve stratification (80/100/70/100). Resumable: rerun to continue after a crash —
existing PSLS .dat files and checkpointed CSV rows are reused.

Run from the project root (long: ~2.5-3 h, ~12 GB):
    python3 sim_systems/run_psls_training.py
    python3 sim_systems/run_psls_training.py --smoke   # 4 curves, validation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generate_training_data import generate_training_dataset_psls  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny 1/1/1/1/1 run for validation")
    ap.add_argument("--n-cold", type=int, default=100, help="S_earth-controlled cold-regime curves")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    counts = dict(n_easy=1, n_medium=1, n_hard=1, n_no_transit=1, n_cold=1) if args.smoke \
        else dict(n_easy=80, n_medium=100, n_hard=70, n_no_transit=100, n_cold=args.n_cold)

    df, X = generate_training_dataset_psls(seed=args.seed, **counts)
    print(f"\nDone: {len(df)} rows | folded {X.shape} | "
          f"transit={int(df['has_transit'].sum())} no-transit={int((df['has_transit'] == 0).sum())}")


if __name__ == "__main__":
    main()
