"""
Noise sensitivity analysis.
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


def noise_sensitivity_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    classifier,
    sigma_levels: Optional[List[float]] = None,
    n_seeds: int = 5,
    n_splits: int = 5,
) -> pd.DataFrame:
    if sigma_levels is None:
        sigma_levels = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    records = []

    for sigma in sigma_levels:
        seed_f1, seed_auc = [], []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            X_noised = X.copy()
            if sigma > 0:
                noise = rng.normal(0, sigma, size=(len(X), len(numeric_cols)))
                X_noised[numeric_cols] = X_noised[numeric_cols].values + noise

            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            fold_f1, fold_auc = [], []
            for train_idx, val_idx in cv.split(X_noised, y):
                X_tr, X_val = X_noised.iloc[train_idx], X_noised.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
                pipe = classifier._build_pipeline()
                pipe.fit(X_tr, y_tr)
                y_pred = pipe.predict(X_val)
                y_prob = pipe.predict_proba(X_val)[:, 1]
                fold_f1.append(f1_score(y_val, y_pred, average="macro"))
                fold_auc.append(roc_auc_score(y_val, y_prob))
            seed_f1.append(np.mean(fold_f1))
            seed_auc.append(np.mean(fold_auc))

        records.append({
            "sigma":    sigma,
            "f1_mean":  float(np.mean(seed_f1)),
            "f1_std":   float(np.std(seed_f1)),
            "auc_mean": float(np.mean(seed_auc)),
            "auc_std":  float(np.std(seed_auc)),
        })

    return pd.DataFrame(records)
