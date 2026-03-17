"""
Feature group ablation study.

Replicates Experiment 1 from the paper: systematically removes each
feature group and measures the impact on macro F1 and AUC-ROC.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from vehiclepm.features.engineering import build_feature_matrix, ALL_FEATURE_GROUPS


def run_ablation_study(
    df: pd.DataFrame,
    target_col: str,
    classifier,
    drop_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Run a feature group ablation study.

    For each group in [mechanical, driver, environmental, interactions],
    removes that group from the feature matrix and evaluates the classifier.
    Also evaluates the full model and internal-only (no context) baseline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame containing all feature columns.
    target_col : str
        Name of the binary target column.
    classifier : VehiclePMClassifier
        Unfitted classifier instance.
    drop_cols : list of str, optional
        Additional columns to exclude from df before feature building.

    Returns
    -------
    pd.DataFrame with columns:
        configuration, n_features, f1_mean, f1_std, auc_mean, f1_drop

    Example
    -------
    >>> from vehiclepm import VehiclePMClassifier, generate_synthetic_dataset
    >>> from vehiclepm.evaluation.ablation import run_ablation_study
    >>>
    >>> df = generate_synthetic_dataset()
    >>> clf = VehiclePMClassifier()
    >>> results = run_ablation_study(df, target_col="maintenance_needed", classifier=clf)
    >>> print(results)
    """
    y = df[target_col]
    base_df = df.drop(columns=[target_col] + (drop_cols or []), errors="ignore")

    configurations = [
        ("All Features (Full Model)", list(ALL_FEATURE_GROUPS.keys())),
        ("Without Internal Mechanical", ["driver", "environmental", "interactions"]),
        ("Without Driver Behaviour",    ["mechanical", "environmental", "interactions"]),
        ("Without Environmental (V2X)", ["mechanical", "driver", "interactions"]),
        ("Without Engineered Interactions", ["mechanical", "driver", "environmental"]),
        ("Internal Only (No Context)",  ["mechanical"]),
    ]

    records = []
    full_f1 = None

    for config_name, groups in configurations:
        X = build_feature_matrix(base_df, include_groups=groups)
        results = classifier.cross_validate(X, y)

        if config_name == "All Features (Full Model)":
            full_f1 = results["f1_mean"]

        records.append({
            "configuration": config_name,
            "n_features":    X.shape[1],
            "f1_mean":       round(results["f1_mean"], 3),
            "f1_std":        round(results["f1_std"], 3),
            "auc_mean":      round(results["auc_mean"], 3),
            "f1_drop":       None,  # filled below
        })

    result_df = pd.DataFrame(records)
    result_df["f1_drop"] = (result_df["f1_mean"] - full_f1).round(3)
    result_df.loc[result_df["configuration"] == "All Features (Full Model)", "f1_drop"] = 0.0

    return result_df
