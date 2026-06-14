"""
ML pipeline: Random Forest transit classifier (Model A) and habitability ranker (Model B).

Model A — Classifier:
  Input:  FEATURE_COLS_A (10 BLS + physical features, observables only)
  Output: transit_prob ∈ [0, 1]
  Label:  has_transit (injected truth)

Model B — Ranker:
  Input:  FEATURE_COLS_B (5 physical features only)
  Output: predicted rank_score (continuous, physics-based)
  Label:  rank_score_label (Gaussian formula with snr_proxy=0)
"""
from __future__ import annotations

import pickle
from pathlib import Path


import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error, r2_score,
)
from sklearn.preprocessing import StandardScaler

# Feature column definitions — subset of generate_training_data.FEATURE_COLS.
# NOTE: bls_peak_snr excluded (raw periodogram power, not calibrated SNR).
# NOTE: period_error_pct excluded (requires injected truth period — unavailable blind).
# NOTE: noise_mult excluded (simulator knob, not observable on real data).
# NOTE: S_earth, Teq_K, Rp_Rearth excluded (derived from injected planet truth —
#       zero for all negatives, so they leak the label; Teff_K/Rstar_Rsun stay
#       because stellar params are known from spectroscopy independent of any transit).
FEATURE_COLS_A = [
    "depth_snr", "period_recovered_d",
    "n_transits", "depth_ppm", "duration_d",
    "Teff_K", "Rstar_Rsun",
]

FEATURE_COLS_B = [
    "S_earth", "Teq_K", "Rp_Rearth", "Teff_K", "Rstar_Rsun",
]

DEFAULT_CLF_PATH = "models/rf_classifier.pkl"
DEFAULT_REG_PATH = "models/rf_ranker.pkl"
DEFAULT_CNN_PATH = "models/cnn_classifier.pt"
DEFAULT_MLP_PATH = "models/mlp_ranker.pt"
DEFAULT_MLP_SCALER_PATH = "models/mlp_scaler.pkl"

FEATURE_COLS_B_MLP = [
    "S_earth", "Teq_K", "Rp_Rearth", "Teff_K", "Rstar_Rsun",
    "period_d", "a_AU", "bond_albedo",
]


class TransitCNN(nn.Module):
    """1D CNN transit classifier. Input: phase-folded flux [201 bins]. Output: transit_prob."""

    def __init__(self) -> None:
        super().__init__()
        # Feature extractor: two 1D conv blocks scan the 201-bin fold for the local
        # dip pattern of a transit. Each Conv1d learns kernels (length-5 windows) that
        # activate on transit-shaped flux drops; out-channels = number of such patterns.
        self.conv_block = nn.Sequential(
            # 1 input channel (flux) -> 16 learned patterns; pad=2 keeps length at 201.
            nn.Conv1d(1, 16, kernel_size=5, padding=2),  # (B, 16, 201)
            nn.ReLU(),                                   # keep positive responses (pattern matched), zero the rest
            nn.MaxPool1d(2),                             # halve length, keep strongest response -> (B, 16, 100)

            # combine the 16 patterns into 32 higher-level ones.
            nn.Conv1d(16, 32, kernel_size=5, padding=2), # (B, 32, 100)
            nn.ReLU(),
            nn.MaxPool1d(2),                             # (B, 32, 50)
        )
        
        # Classifier head: flatten conv features -> dense layers -> probability.
        self.classifier = nn.Sequential(
            nn.Flatten(),           # (B, 32, 50) -> (B, 1600)
            nn.Linear(1600, 64),    # learn combinations of the conv features
            nn.ReLU(),
            nn.Dropout(0.3),        # randomly zero 30% of units while training -> curb overfitting
            nn.Linear(64, 1),       # collapse to a single logit
            nn.Sigmoid(),           # logit -> transit probability in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 201) → transit_prob: (B,)"""
        # unsqueeze(1): (B, 201) -> (B, 1, 201), add the channel dim Conv1d expects.
        # conv_block extracts dip features; classifier maps them to a probability.
        # squeeze(1): (B, 1) -> (B,), one scalar prob per light curve.
        return self.classifier(self.conv_block(x.unsqueeze(1))).squeeze(1)


class HabitabilityMLP(nn.Module):
    """MLP habitability ranker. Input: n physics features (StandardScaler normalized). Output: rank_score."""

    def __init__(self, n_features: int = 8) -> None:
        super().__init__()
        # 128-wide hidden layers: the tightened H_HZ (sigma_S=0.3) makes rank_score a
        # peaky function of S_earth; the old 64/32 net under-fit it (R2~0.78, over-scored
        # cold planets). 128/128 recovers R2~0.88.
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_features) → rank_score: (B,)"""
        return self.net(x).squeeze(1)


def train_rf_classifier(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 200,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Fit RF classifier. X shape: (n_samples, len(FEATURE_COLS_A)).
    y: binary array, 1=transit present, 0=no transit.
    class_weight='balanced' compensates for tier imbalance.
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def predict_transit_prob(
    model: RandomForestClassifier,
    X: np.ndarray,
) -> np.ndarray:
    """Return transit probability ∈ [0,1] for each row in X."""
    return model.predict_proba(X)[:, 1]  # [:, 1] = positive class (transit present) probability


def evaluate_classifier(
    model: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    """Compute precision, recall, F1, AUC on (X, y)."""
    probs = predict_transit_prob(model, X)
    preds = (probs >= 0.5).astype(int)
    return {
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall":    float(recall_score(y, preds, zero_division=0)),
        "f1":        float(f1_score(y, preds, zero_division=0)),
        "auc":       float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else float("nan"),  # guard: AUC undefined if val split contains only one class
    }


def train_rf_ranker(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 8,
    random_state: int = 42,
) -> RandomForestRegressor:
    """
    Fit RF regressor. X shape: (n_samples, len(FEATURE_COLS_B)).
    y: rank_score_label (Gaussian formula, snr_proxy=0).
    """
    reg = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    reg.fit(X, y)
    return reg


def predict_rank_score(
    model: RandomForestRegressor,
    X: np.ndarray,
) -> np.ndarray:
    """Return predicted rank_score (continuous) for each row in X."""
    return model.predict(X)


def evaluate_ranker(
    model: RandomForestRegressor,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    """Compute MAE and R² on (X, y)."""
    preds = predict_rank_score(model, X)
    return {
        "mae": float(mean_absolute_error(y, preds)),
        "r2":  float(r2_score(y, preds)),
    }


def train_cnn_classifier(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.2,
    patience: int = 5,
    seed: int = 42,
) -> TransitCNN:
    """
    Train 1D CNN on phase-folded light curves.

    Args:
        X: (N, 201) float32 array of normalized phase-folded flux arrays
        y: (N,) int array of binary transit labels {0, 1}
        seed: RNG seed for the shuffled train/val split and torch init.

    Returns:
        Trained TransitCNN with best validation loss weights restored.
    """
    torch.manual_seed(seed)
    # Shuffle before split — input rows may be ordered by tier/label.
    order = np.random.default_rng(seed).permutation(len(X))
    X, y = X[order], y[order]
    n_val = int(len(X) * val_split)
    X_val, y_val = X[:n_val], y[:n_val]
    X_tr, y_tr = X[n_val:], y[n_val:]

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    model = TransitCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()  # binary cross-entropy: correct loss for sigmoid output + binary labels

    best_val_loss = float("inf")
    best_state: dict | None = None
    wait = 0

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr_t))
        for i in range(0, len(X_tr_t), batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_tr_t[idx]), y_tr_t[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}  # clone: snapshot weights, not a reference that mutates with training
            wait = 0
        else:
            wait += 1
            if wait >= patience:  # early stopping: restore best weights to avoid overfitting the last epochs
                break

    if best_state is not None:
        model.load_state_dict(best_state)  # rewind to best val-loss checkpoint
    model.eval()
    return model


def predict_transit_prob_cnn(model: TransitCNN, X: np.ndarray) -> np.ndarray:
    """Run CNN inference. X: (N, 201). Returns probs: (N,) in [0, 1]."""
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()


def evaluate_cnn(
    model: TransitCNN,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    """Evaluate CNN classifier. Returns precision, recall, f1, auc."""
    probs = predict_transit_prob_cnn(model, X)
    preds = (probs >= 0.5).astype(int)
    return {
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall":    float(recall_score(y, preds, zero_division=0)),
        "f1":        float(f1_score(y, preds, zero_division=0)),
        "auc":       float(roc_auc_score(y, probs)),
    }


def train_mlp_ranker_nn(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    val_split: float = 0.2,
    patience: int = 10,
    seed: int = 42,
) -> tuple[HabitabilityMLP, StandardScaler]:
    """
    Train MLP habitability ranker on physics features.

    Args:
        X: (N, 8) array with columns matching FEATURE_COLS_B_MLP
        y: (N,) array of rank_score_label targets
        seed: RNG seed for the shuffled train/val split and torch init.

    Returns:
        (model, scaler): trained MLP and fitted StandardScaler.
        Apply scaler.transform(X) before calling predict_rank_score_mlp().
    """
    torch.manual_seed(seed)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)  # MLP sensitive to feature scale; CNN skips this (phase-fold already normalized)

    # Shuffle before split — input rows may be ordered by tier/label.
    order = np.random.default_rng(seed).permutation(len(X_scaled))
    X_scaled, y = X_scaled[order], y[order]
    n_val = int(len(X_scaled) * val_split)
    X_val, y_val = X_scaled[:n_val], y[:n_val]
    X_tr, y_tr = X_scaled[n_val:], y[n_val:]

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    #model + optimizer
    model = HabitabilityMLP(n_features=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # regression target (rank_score), not binary — MSE penalizes large deviations

    best_val_loss = float("inf")
    best_state: dict | None = None
    wait = 0

    #training loop, 
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr_t)) #reshuffle training data each epoch to avoid bias from ordering
        for i in range(0, len(X_tr_t), batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_tr_t[idx]), y_tr_t[idx])
            loss.backward()             #compute gradients
            optimizer.step()            #update weights

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}  # clone: snapshot weights, not a reference that mutates with training
            wait = 0
        else:
            wait += 1
            if wait >= patience:  #no improvement for 'patience' epochs --> stop
                break

    if best_state is not None:
        model.load_state_dict(best_state)  # rewind to best val-loss checkpoint
    model.eval()
    return model, scaler


def predict_rank_score_mlp(model: HabitabilityMLP, X_scaled: np.ndarray) -> np.ndarray:
    """Run MLP inference. X_scaled: (N, 8) already StandardScaler-transformed. Returns (N,)."""
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X_scaled, dtype=torch.float32)).numpy()


def evaluate_mlp(
    model: HabitabilityMLP,
    X_scaled: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    """Evaluate MLP ranker. X_scaled already transformed. Returns mae, r2."""
    preds = predict_rank_score_mlp(model, X_scaled)
    return {
        "mae": float(mean_absolute_error(y, preds)),
        "r2":  float(r2_score(y, preds)),
    }


def save_cnn_models(
    cnn: TransitCNN,
    mlp: HabitabilityMLP,
    scaler: StandardScaler,
    cnn_path: str = DEFAULT_CNN_PATH,
    mlp_path: str = DEFAULT_MLP_PATH,
    scaler_path: str = DEFAULT_MLP_SCALER_PATH,
) -> None:
    """Save CNN, MLP, and StandardScaler to disk."""
    Path(cnn_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cnn, cnn_path)  # saves full model object (not just state_dict) so architecture loads without re-defining the class
    torch.save(mlp, mlp_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved CNN     → {cnn_path}")
    print(f"Saved MLP     → {mlp_path}")
    print(f"Saved scaler  → {scaler_path}")


def load_cnn_models(
    cnn_path: str = DEFAULT_CNN_PATH,
    mlp_path: str = DEFAULT_MLP_PATH,
    scaler_path: str = DEFAULT_MLP_SCALER_PATH,
) -> tuple[TransitCNN, HabitabilityMLP, StandardScaler]:
    """Load CNN, MLP, and StandardScaler from disk."""
    cnn = torch.load(cnn_path, weights_only=False)
    mlp = torch.load(mlp_path, weights_only=False)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return cnn, mlp, scaler


def save_models(
    clf: RandomForestClassifier,
    reg: RandomForestRegressor,
    clf_path: str = DEFAULT_CLF_PATH,
    reg_path: str = DEFAULT_REG_PATH,
) -> None:
    """Pickle both models to disk."""
    Path(clf_path).parent.mkdir(parents=True, exist_ok=True)
    with open(clf_path, "wb") as f:
        pickle.dump(clf, f)
    Path(reg_path).parent.mkdir(parents=True, exist_ok=True)
    with open(reg_path, "wb") as f:
        pickle.dump(reg, f)
    print(f"Saved classifier → {clf_path}")
    print(f"Saved ranker     → {reg_path}")


def load_models(
    clf_path: str = DEFAULT_CLF_PATH,
    reg_path: str = DEFAULT_REG_PATH,
) -> tuple[RandomForestClassifier, RandomForestRegressor]:
    """Load both models from disk."""
    with open(clf_path, "rb") as f:
        clf = pickle.load(f)
    with open(reg_path, "rb") as f:
        reg = pickle.load(f)
    return clf, reg


if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split

    print("=== Loading training data ===")
    df = pd.read_csv("training_data/training_labels.csv")
    print(f"  {len(df)} rows | transit={df['has_transit'].sum()} no-transit={(df['has_transit']==0).sum()}")

    # SPLIT: the classifier (detection) excludes the cold tier — those planets are mostly
    # undetectable (P >> baseline), so their folds/features are noise labelled has_transit=1,
    # which teaches "noise = transit" and wrecks no_transit recall. The ranker (physics) keeps
    # ALL rows, including cold, because it needs cold-flux coverage and ignores detectability.
    df_clf = df[df["tier"] != "cold"] if "tier" in df.columns else df
    print(f"  classifier on {len(df_clf)} rows (cold excluded); ranker on {len(df)} rows")

    X_a = df_clf[FEATURE_COLS_A].values
    y_a = df_clf["has_transit"].values
    X_b = df[FEATURE_COLS_B].values
    y_b = df["rank_score_label"].values

    # Train/val split — stratified by has_transit for classifier
    X_a_tr, X_a_val, y_a_tr, y_a_val = train_test_split(
        X_a, y_a, test_size=0.2, random_state=42, stratify=y_a
    )
    X_b_tr, X_b_val, y_b_tr, y_b_val = train_test_split(
        X_b, y_b, test_size=0.2, random_state=42
    )

    print("\n=== Training Model A (RF Classifier) ===")
    clf = train_rf_classifier(X_a_tr, y_a_tr)
    clf_val = evaluate_classifier(clf, X_a_val, y_a_val)
    print(f"  Val precision={clf_val['precision']:.3f}  recall={clf_val['recall']:.3f}  "
          f"F1={clf_val['f1']:.3f}  AUC={clf_val['auc']:.3f}")

    print("\n=== Training Model B (RF Ranker) ===")
    reg = train_rf_ranker(X_b_tr, y_b_tr)
    reg_val = evaluate_ranker(reg, X_b_val, y_b_val)
    print(f"  Val MAE={reg_val['mae']:.4f}  R²={reg_val['r2']:.4f}")

    print("\n=== Saving models ===")
    save_models(clf, reg)
    print("Done.")
