# PLATO Habitable-Zone Transit Detection Pipeline

End-to-end pipeline for simulating, detecting, and ranking habitable-zone exoplanet candidates using PLATO-like photometry. Built for MASS Semester 2 Computational Astrobiology (Project 10).

## What this project does (non-specialist summary)

When a planet passes in front of its star, it blocks a tiny fraction of the star's light. By watching for these repeated dips in brightness, we can discover planets orbiting distant stars. This project simulates what the ESA PLATO space telescope would observe, then uses a combination of classical signal-processing and machine learning to:

1. **Find the dips** — an algorithm called Box Least Squares (BLS) scans the brightness record and identifies repeating dimming patterns that match a planetary transit.
2. **Score each candidate** — a physics-based formula rates how Earth-like each planet might be, considering how much starlight it receives, its estimated size, and the type of star it orbits.
3. **Cross-check with machine learning** — a neural network and a random forest independently assess whether the dip is a real planet signal and how habitable the planet might be, providing a second opinion on the rankings.

---

## Overview

```
PSLS simulation → BLS transit search → Transparent ranking → ML models → Hybrid ranking
```

1. **Simulate** PLATO-like light curves via PSLS (12 stellar systems)
2. **Detect** transits with Box Least Squares (BLS) + harmonic refinement
3. **Rank** candidates with a transparent physics-based Gaussian score
4. **Classify** transits with a 1D CNN and Random Forest
5. **Rank habitability** with an MLP and Random Forest regressor
6. **Compare** transparent vs ML-hybrid ranking schemes

---

## Repository Structure


```
.
├── habitable_zone_pipeline.py   # Physics core: L*, S, T_eq, HZ flags, transparent ranking model
├── transit_helpers.py           # LC loading, detrending, BLS diagnostics, verdict labels
├── ml_pipeline.py               # CNN, MLP, RF models — train/predict/evaluate/save/load
├── generate_training_data.py    # Synthetic LC generation for ML training (450 rows, 5 tiers)
├── retrain_cnn_mlp.py           # Standalone retraining script for CNN + MLP
├── psls_runner.py               # PSLS simulation runner
├── transit_search.ipynb         # Main analysis notebook (Tasks 1–8 + ML sections)
│
├── sim_systems/                 # 12 YAML system configs (star + planet + metadata)
│   ├── A1_K_hot.yaml            # K-dwarf, hot inner planet
│   ├── A2_K_hz.yaml             # K-dwarf, habitable zone
│   ├── B1_G_hot.yaml, B2_G_hz.yaml, B3_G_cold.yaml
│   ├── C1_F_hot.yaml, C2_F_hz.yaml, C3_F_adversarial.yaml
│   └── D1–D4_*.yaml             # Validation + control systems
│
├── models/                      # Trained model artefacts
│   ├── cnn_classifier.pt        # 1D CNN transit classifier
│   ├── mlp_ranker.pt            # MLP habitability ranker
│   ├── mlp_scaler.pkl           # StandardScaler for MLP input
│   ├── rf_classifier.pkl        # Random Forest transit classifier
│   └── rf_ranker.pkl            # Random Forest habitability ranker
│
└── training_data/
    ├── training_labels.csv      # 450-row tabular dataset (BLS features + labels)
    ├── cnn_phase_folded.npy     # Phase-folded flux arrays (N, 201) for CNN training
    └── cnn_labels.npy           # Binary transit labels for CNN training
```

---

## Physics

### Stellar luminosity
```
L*/L☉ = (R*/R☉)² (T*/T☉)⁴        [Stefan-Boltzmann]
```

### Incident flux
```
S = (L*/L☉) / (a/AU)²             [inverse-square law; S=1 = Earth at 1 AU]
```

### Equilibrium temperature
```
T_eq = T* √(R*/(2a)) (1-A)^(1/4)  [energy balance, uniform redistribution]
```

### Transparent ranking score
```
Score = 1.00 · H_HZ + 0.20 · H_Rp + 0.15 · H_host + 0.15 · H_det + 0.10 · H_stab
```

| Term | Formula | Peak at |
|---|---|---|
| H_HZ | exp(-(S-1)² / 2σ_S²), σ_S=1.0 | S = 1 (Earth-like flux) |
| H_Rp | exp(-(Rp-1.2)² / 2σ_R²), σ_R=0.5 | 1.2 R⊕ (rocky super-Earth) |
| H_host | exp(-(T*-5800)² / 2·800²) | 5800 K (solar analog) |
| H_det | 1 - exp(-SNR / 1.5) | saturates at high BLS SNR |
| H_stab | exp(-e² / 2·0.1²) | e = 0 (circular orbit) |

---

## ML Models

### Model A — Transit Classifier

| | CNN | Random Forest |
|---|---|---|
| Input | Phase-folded flux (201 bins) | BLS features + stellar params (7 cols) |
| Output | transit_prob ∈ [0,1] | transit_prob ∈ [0,1] |
| Label | has_transit (injected truth) | has_transit |
| Loss | BCE | — |
| Why | Detects local dip pattern via conv kernels | Tabular BLS features |

### Model B — Habitability Ranker

| | MLP | Random Forest |
|---|---|---|
| Input | Physics features (8 cols, StandardScaler normalized) | Physics features (5 cols) |
| Output | predicted rank_score | predicted rank_score |
| Label | rank_score_label (Gaussian formula, snr_proxy=0) | rank_score_label |
| Loss | MSE | — |
| Why | Learns nonlinear physics combinations | Tabular physics features |

`snr_proxy=0` in label: strips detection quality so ranker learns pure orbital/physical habitability.

---

## Simulation Systems (12)

| ID | Star | Science case |
|---|---|---|
| A1 | K-dwarf | Hot inner — non-HZ control |
| A2 | K-dwarf | Habitable zone |
| B1 | G-dwarf | Hot inner |
| B2 | G-dwarf | Habitable zone |
| B3 | G-dwarf | Cold outer |
| C1 | F-dwarf | Hot inner |
| C2 | F-dwarf | Habitable zone |
| C3 | F-dwarf | Adversarial (long period, small planet) |
| D1 | K-dwarf | HZ promising (validation) |
| D2 | K-dwarf | HZ promising 2 (validation) |
| D3 | G-dwarf | Hot control |
| D4 | G-dwarf | Cold control |

---

## Training Data

450 synthetic light curves across 5 difficulty tiers (350 transit + 100 no-transit):

| Tier | Count | Period | Rp | Noise |
|---|---|---|---|---|
| easy | 80 | 5–30 d | 2.0–4.0 R⊕ | 1× |
| medium | 100 | 30–150 d | 1.5–3.0 R⊕ | 1–2× |
| hard | 70 | 150–270 d | 0.8–2.0 R⊕ | 2–5× |
| cold | 100 | S⊕-controlled (0.15–0.9 S⊕) | 0.8–2.0 R⊕ | 1–2× |
| no_transit | 100 | — | — | 1–3× |

The `cold` tier inverts S = L/a² to pick long-period, cold-regime orbits (often undetectable in 270 d) so the ranker sees genuine low-insolation physics instead of extrapolating from the period-based tiers.

Base noise: 80 ppm. Cadence: 25 s × 72 downsample = 30 min effective.

---

## Ranking Experiments

Three schemes compared in `transit_search.ipynb`:

| Scheme | H_det source |
|---|---|
| Transparent | BLS depth_snr → Gaussian formula |
| CNN hybrid (D2) | CNN transit_prob replaces H_det slot (w=0.15) |
| RF combined (ML-3) | 0.6 × RF_rank + 0.4 × RF_transit_prob |

---

## Dependencies

```
numpy, pandas, astropy, scipy
scikit-learn, torch
matplotlib, seaborn
psls, pyyaml
```

Install PSLS: `pip install psls`

---

## How to Run

1. Open `transit_search.ipynb` in Jupyter
2. Select **Kernel → Restart & Run All**

The notebook runs top-to-bottom with no manual steps. Pre-trained models are included in `models/` so retraining is not required.

**Optional — regenerate training data and retrain models:**

```bash
python generate_training_data.py --n-cold 100   # regenerate 450-row training dataset
python retrain_cnn_mlp.py          # retrain CNN + MLP from scratch
```
