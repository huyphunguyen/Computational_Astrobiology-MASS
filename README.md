# PLATO Habitable-Zone Transit Detection Pipeline

End-to-end pipeline for simulating, detecting, and ranking habitable-zone exoplanet candidates using PLATO-like photometry. Built for MASS Semester 2 Computational Astrobiology (Project 10).

## What this project does 

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

## Pipeline Workflow

### Training Phase (`generate_training_data.py` → `retrain_cnn_mlp.py`)

450 PSLS light curves are generated across 5 difficulty tiers (easy / medium / hard / cold / no-transit). Each curve is detrended with a running median to remove stellar granulation, then fed into BLS to recover the transit period, depth, and SNR. Two parallel feature paths emerge: the phase-folded flux array (201 bins) trains the CNN and MLP via gradient descent (BCE and MSE loss respectively), while the tabular BLS + physics features train the Random Forest classifier and ranker via tree splits. All four models are saved to `models/`.

```
PSLS (450 curves, 5 tiers)
  │
  ├─► Detrend (running median, 1 d window)
  │
  └─► BLS recovery (period / depth / SNR / duration)
        │
        ├─► Phase-fold at recovered period → 201-bin array
        │     └─► CNN classifier  (input: 201-bin flux)
        │         MLP ranker      (input: 8 physics cols)
        │
        └─► Tabular features + physics labels
              └─► RF classifier  (input: 7 BLS+stellar cols)
                  RF ranker      (input: 5 physics cols)
```

### Inference Phase (`transit_search.ipynb`, 12 sim systems)
Each of the 12 YAML-defined stellar systems is simulated through PSLS to produce a realistic PLATO-like light curve. After detrending and BLS search, three independent ranking schemes are applied to every candidate. The transparent scheme uses only a Gaussian physics formula — no ML. The CNN-hybrid replaces the detection quality term (H_det) with the CNN's transit probability. The RF-combined also does the same with the RF classifier's transit probability.

```
sim_systems/ (12 YAMLs)
  │
  └─► PSLS → raw .dat light curve
        │
        └─► Detrend → BLS search
              │
              ├─► Physics score
              │     H_HZ · 1.00 + H_Rp · 0.20 + H_host · 0.15
              │     + H_det · 0.15 + H_stab · 0.10
              │           │
              │           ▼
              │     Transparent ranking
              │
              ├─► CNN transit_prob --> replaces H_det slot
              │         │
              │         ▼
              │     CNN-hybrid ranking
              │
              └─► RF transit_prob --> replaces H_det slot
                        │
                        ▼
                  RF-combined ranking
```

### Scheme Comparison
The three ranked lists are compared directly using a bump chart (rank position per system across schemes) and Spearman / Kendall correlation coefficients. This reveals where ML and physics agree, where they diverge, and whether swapping in a data-driven detector changes which planets are prioritised for follow-up.

```
12 systems → [Transparent | CNN-hybrid | RF-combined]
                    │
                    └─► Bump chart + Spearman/Kendall rank correlation
```

---

## File Guide

### `habitable_zone_pipeline.py` — Physics core
Central module. Everything that touches stars, orbits, or scores lives here.

| Function | Does |
|---|---|
| `stellar_luminosity_solar` | L★ from R★, T★ via Stefan-Boltzmann |
| `incident_flux_earth_units` | S = L★ / a² (S=1 → Earth-like) |
| `equilibrium_temperature_from_au` | T_eq from orbital distance + albedo |
| `habitable_zone_flags` | Returns `in_hz`, `near_hz`, `too_hot`, `too_cold` booleans |
| `habitability_rank_score` | Full Gaussian score → dict of H_HZ, H_Rp, H_host, H_det, H_stab, rank_score |
| `run_bls_recovery_with_refinement` | BLS period search + harmonic grid refinement → period/depth/SNR/duration |
| `phase_fold_lightcurve` | Fold + bin flux to N bins centred on transit |
| `load_cnn_models` | Load CNN + MLP + scaler from `models/` |

---

### `transit_helpers.py` — Light curve and diagnostics
Bridges raw PSLS output to the analysis pipeline.

| Function | Does |
|---|---|
| `load_systems_from_yaml` | Parse all `sim_systems/*.yaml` into a dict keyed by system ID |
| `load_psls_dat` | Read PSLS `.dat` file → `(time, flux)` arrays, applies stride downsampling |
| `detrend_flux` | Running-median detrend (removes granulation/activity trends) |
| `verdict` | Returns  label: `"HZ DETECTED"`, `"NON-HZ"`, `"MISSED"`, etc. |
| `plot_bls_diagnostic` | BLS power spectrum + phase-folded LC diagnostic plot |

---

### `ml_pipeline.py` — ML models
All model definitions, training, prediction, and persistence.

| Class / Function | Does |
|---|---|
| `TransitCNN` | 1D CNN: 3 conv layers + 2 FC layers, input 201-bin phase-folded flux |
| `HabitabilityMLP` | MLP: 3 hidden layers (128→64→32), input 8 physics features |
| `train_cnn_classifier` | Train CNN with BCE loss, Adam optimizer, early stopping |
| `train_mlp_ranker_nn` | Train MLP with MSE loss, StandardScaler normalization |
| `train_rf_classifier` | Train RF on 7 tabular BLS+stellar features |
| `train_rf_ranker` | Train RF on 5 physics features |
| `predict_transit_prob` / `predict_transit_prob_cnn` | Inference: transit probability ∈ [0,1] |
| `predict_rank_score` / `predict_rank_score_mlp` | Inference: habitability rank score |
| `save_models` / `load_models` | Persist/restore RF models (`.pkl`) + CNN/MLP (`.pt`) + scaler |

---

### `generate_training_data.py` — PSLS training data generator
Generates the 450-curve stratified dataset used to train all four models.

| Function | Does |
|---|---|
| `generate_training_dataset_psls` | Main entry point: runs PSLS per sample, extracts tabular + phase-folded features, writes CSV + npy. Resumable — skips already-computed `star_id`s |
| `_generate_one_sample_psls` | Single sample: write YAML → run PSLS → detrend → BLS → compute all features → phase-fold |
| `_sample_tier_params` | Sample (period, Rp, noise_mult) for a given difficulty tier |

Run directly:
```bash
python(or python3) generate_training_data.py --n-cold 100
```

---

### `retrain_cnn_mlp.py` — Retraining script
Standalone script to retrain CNN + MLP from existing `training_data/`. Does not regenerate data.

```bash
python(or python3) retrain_cnn_mlp.py
```
Saves updated weights to `models/cnn_classifier.pt`, `models/mlp_ranker.pt`, `models/mlp_scaler.pkl`.

---

### `psls_runner.py` — PSLS interface
Thin wrapper around the PSLS binary for PLATO light curve simulation.

| Function | Does |
|---|---|
| `match_grid_model` | Snap (Teff, logg) to nearest model in `grid_plato.hdf5` → returns exact (Teff, logg, Mstar, Rstar) |
| `write_psls_yaml` | Write PSLS config YAML for one stellar system |
| `run_psls` | Execute PSLS binary, return path to output `.dat` file (idempotent — skips if exists) |
| `read_psls_star_params` | Parse stellar parameters from `.dat` header |

---

### `transit_search.ipynb` — Main analysis notebook
Single notebook covering all tasks end-to-end. Run top-to-bottom with **Kernel → Restart & Run All**.

| Section | Tasks | Content |
|---|---|---|
| Setup | — | Imports, load systems, run PSLS simulations |
| Tasks 1–2 | A-B | Light curve plots, BLS detection on 12 systems |
| Task 3 | B-C | Physics scoring: S, T_eq, HZ flags, transparent ranking |
| Task 4–5 | C | ML training (CNN, MLP, RF), evaluation tables |
| Task 6 | C | CNN calibration, confusion matrix, Brier score |
| Task 7 | C-D | MLP regression diagnostics, RF feature importance |
| Task 8 | D | Rank-agreement comparison: bump chart + Spearman/Kendall |

Pre-trained models in `models/` load automatically — retraining not required.

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


---

## Ranking Experiments

Three schemes compared in `transit_search.ipynb`:

| Scheme | H_det source |
|---|---|
| Transparent | BLS depth_snr → Gaussian formula |
| CNN hybrid | CNN transit_prob replaces H_det slot (w=0.15) |
| RF combined | RF transit_prob replaces H_det slot (w =0.15) |

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
Additionally, the stellar model grid 'PlatoLightCurves/psls-1.9/grid_plato.hdf5' designed to capture oscillations of Sun-like stars was provided by directly contacting [mailto:reza.samadi@obspm.fr].

## Notebook checks

For checking notebooks, run this

```bash
pytest --nbmake --nbmake-timeout=60 transit_search.ipynb   # notebook runs top-to-bottom without errors
nbqa pylint transit_search.ipynb                            # static analysis on notebook cells
```

