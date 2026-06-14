#!/bin/bash
# run_all.sh — Run all 8 PSLS simulations for HZ target selection study
# Usage: cd sim_systems && bash run_all.sh
# Outputs go to sim_systems/outputs/<config_name>/

set -e

export MPLBACKEND=Agg  # non-interactive backend: save PNGs without displaying windows

PSLS="python ../psls-1.9/psls.py"
OUTBASE="outputs"

configs=(
  "A1_K_hot"
  "A2_K_hz"
  "B1_G_hot"
  "B2_G_hz"
  "B3_G_cold"
  "C1_F_hot"
  "C2_F_hz"
  "C3_F_adversarial"
  "D1_K_hz_promising"
  "D2_K_hz_promising2"
  "D3_G_hot_control"
  "D4_G_cold_control"
)

for cfg in "${configs[@]}"; do
  echo "=========================================="
  echo "Running: $cfg"
  echo "=========================================="
  mkdir -p "$OUTBASE/$cfg"
  $PSLS --extended-plots -o "$OUTBASE/$cfg/" "${cfg}.yaml" < /dev/null
  echo "Done: $cfg -> $OUTBASE/$cfg/"
done

echo ""
echo "All 8 simulations complete."
echo "Outputs in: sim_systems/outputs/"
