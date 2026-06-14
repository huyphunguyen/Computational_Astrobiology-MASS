"""
retrain_cnn_mlp.py — retrain the CNN transit classifier + MLP habitability ranker
on the PSLS training data, and overwrite models/{cnn_classifier.pt, mlp_ranker.pt,
mlp_scaler.pkl}.

The notebook only LOADS these models (cell-024) — no cell trains them — so this script
is the retraining entry point after generate_training_data.py --source psls.

Run from the project root:
    python3 retrain_cnn_mlp.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_pipeline import (
    FEATURE_COLS_B_MLP,
    train_cnn_classifier,
    train_mlp_ranker_nn,
    save_cnn_models,
    predict_transit_prob_cnn,
    predict_rank_score_mlp,
)
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split



def main() -> None:
    # --- CNN: phase-folded curves -> has_transit (EXCLUDE cold tier, like the RF classifier) ---
    # Cold planets are mostly undetectable -> noise folds labelled has_transit=1 -> they teach
    # "noise = transit" and collapse no_transit recall. Detection trains on the 350 non-cold.
    labels_df = pd.read_csv("training_data/training_labels.csv")
    X_cnn = np.load("training_data/cnn_phase_folded.npy")
    y_cnn = np.load("training_data/cnn_labels.npy")
    if "tier" in labels_df.columns and len(labels_df) == len(X_cnn):
        keep = (labels_df["tier"] != "cold").to_numpy()
        X_cnn, y_cnn = X_cnn[keep], y_cnn[keep]
    print(f"CNN data: {X_cnn.shape}  transit={int(y_cnn.sum())} no-transit={int((y_cnn==0).sum())} (cold excluded)")
    Xc_tr, Xc_val, yc_tr, yc_val = train_test_split(
        X_cnn, y_cnn, test_size=0.2, random_state=42, stratify=y_cnn
    )
    cnn = train_cnn_classifier(Xc_tr, yc_tr)
    auc_cnn = roc_auc_score(yc_val, predict_transit_prob_cnn(cnn, Xc_val))
    print(f"  CNN val AUC={auc_cnn:.3f}")

    # --- MLP: physics features -> rank_score_label ---
    df = pd.read_csv("training_data/training_labels.csv")
    X_mlp = df[FEATURE_COLS_B_MLP].values.astype(np.float32)
    y_mlp = df["rank_score_label"].values.astype(np.float32)
    Xm_tr, Xm_val, ym_tr, ym_val = train_test_split(X_mlp, y_mlp, test_size=0.2, random_state=42)
    # Peaky sigma_S=0.3 target needs longer training + a wider net (see HabitabilityMLP).
    mlp, scaler = train_mlp_ranker_nn(Xm_tr, ym_tr, epochs=1000, patience=50)
    pred = predict_rank_score_mlp(mlp, scaler.transform(Xm_val))
    print(f"  MLP val MAE={mean_absolute_error(ym_val, pred):.4f}  R2={r2_score(ym_val, pred):.4f}")

    save_cnn_models(cnn, mlp, scaler)
    print("Saved CNN + MLP -> models/{cnn_classifier.pt, mlp_ranker.pt, mlp_scaler.pkl}")


if __name__ == "__main__":
    main()
