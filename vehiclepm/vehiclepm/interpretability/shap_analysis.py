"""
SHAP-based feature importance analysis.

Replicates Experiment 5 from the paper: top-15 feature importance
bar chart and beeswarm plot using SHAP TreeExplainer.
"""

import numpy as np
import pandas as pd
from typing import Optional, List


def compute_shap_importance(
    model,
    X: pd.DataFrame,
    max_display: int = 15,
    plot: bool = True,
) -> pd.DataFrame:
    """
    Compute SHAP feature importance using TreeExplainer.

    Parameters
    ----------
    model : fitted tree-based estimator
        e.g. LGBMClassifier, XGBClassifier, RandomForestClassifier.
        Pass clf.get_base_model() from VehiclePMClassifier.
    X : pd.DataFrame
        Feature matrix (same columns used for training).
    max_display : int
        Number of top features to display. Default 15.
    plot : bool
        If True, generate summary bar plot and beeswarm plot.

    Returns
    -------
    pd.DataFrame with columns ['feature', 'mean_abs_shap']
        sorted by importance descending.

    Example
    -------
    >>> from vehiclepm import VehiclePMClassifier, generate_synthetic_dataset
    >>> from vehiclepm.features import build_feature_matrix
    >>> from vehiclepm.interpretability.shap_analysis import compute_shap_importance
    >>>
    >>> df = generate_synthetic_dataset()
    >>> X = build_feature_matrix(df.drop(columns=["risk_score","maintenance_needed"]))
    >>> y = df["maintenance_needed"]
    >>> clf = VehiclePMClassifier()
    >>> clf.fit(X, y)
    >>> importance_df = compute_shap_importance(clf.get_base_model(), X)
    """
    try:
        import shap
    except ImportError:
        raise ImportError(
            "SHAP is required for interpretability analysis. "
            "Install it with: pip install shap"
        )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # For binary classification, shap_values may be a list [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature":        X.columns.tolist(),
        "mean_abs_shap":  mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    if plot:
        try:
            import matplotlib.pyplot as plt

            # Bar plot — top N features
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))

            top = importance_df.head(max_display)
            axes[0].barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#2196F3")
            axes[0].set_xlabel("Mean |SHAP Value|")
            axes[0].set_title(f"Top {max_display} Features by Mean |SHAP|")
            axes[0].grid(axis="x", alpha=0.3)

            # Beeswarm via shap
            plt.sca(axes[1])
            shap.summary_plot(
                shap_values, X,
                max_display=max_display,
                show=False,
                plot_size=None,
            )
            axes[1].set_title("SHAP Beeswarm — Directional Feature Effects")

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("matplotlib not found — skipping plots. Install with: pip install matplotlib")

    return importance_df
