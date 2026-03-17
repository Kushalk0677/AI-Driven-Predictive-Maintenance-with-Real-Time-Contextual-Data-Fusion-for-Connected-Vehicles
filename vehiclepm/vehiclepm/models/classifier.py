"""
VehiclePMClassifier — unified interface for vehicle predictive maintenance.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    _IMBLEARN = True
except ImportError:
    _IMBLEARN = False

try:
    from lightgbm import LGBMClassifier
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False


_MODEL_REGISTRY = {
    "lightgbm": lambda: LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        class_weight="balanced", random_state=42, verbose=-1
    ) if _LGBM_AVAILABLE else None,
    "xgboost": lambda: XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=5,
        use_label_encoder=False, eval_metric="logloss", random_state=42
    ) if _XGB_AVAILABLE else None,
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight="balanced", random_state=42
    ),
    "logistic_regression": lambda: LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000, random_state=42
    ),
}


class VehiclePMClassifier:
    """
    Predictive maintenance classifier with built-in SMOTE and cross-validation.

    Parameters
    ----------
    model_type : str
        One of 'random_forest' (default, no extra deps),
        'lightgbm', 'xgboost', 'logistic_regression'.
    n_splits : int
        Number of stratified CV folds. Default 5.
    calibrate : bool
        Wrap final model with Platt scaling. Default False.
    random_state : int
        Random seed. Default 42.

    Example
    -------
    >>> from vehiclepm import VehiclePMClassifier, generate_synthetic_dataset
    >>> from vehiclepm.features import build_feature_matrix
    >>> df = generate_synthetic_dataset()
    >>> X = build_feature_matrix(df.drop(columns=["risk_score","maintenance_needed"]))
    >>> y = df["maintenance_needed"]
    >>> clf = VehiclePMClassifier()
    >>> results = clf.cross_validate(X, y)
    >>> print(f"F1: {results['f1_mean']:.3f}")
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        n_splits: int = 5,
        calibrate: bool = False,
        smote_sampling_strategy: Any = "auto",
        random_state: int = 42,
    ):
        if model_type not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Choose from: {list(_MODEL_REGISTRY.keys())}"
            )
        self.model_type = model_type
        self.n_splits = n_splits
        self.calibrate = calibrate
        self.smote_sampling_strategy = smote_sampling_strategy
        self.random_state = random_state
        self._model = None
        self._feature_names: Optional[List[str]] = None

    def _build_pipeline(self):
        base = _MODEL_REGISTRY[self.model_type]()
        if base is None:
            raise ImportError(
                f"'{self.model_type}' is not installed. "
                f"Run: pip install {self.model_type}"
            )
        if self.calibrate:
            base = CalibratedClassifierCV(base, method="sigmoid", cv=3)

        if _IMBLEARN:
            smote = SMOTE(
                sampling_strategy=self.smote_sampling_strategy,
                random_state=self.random_state,
            )
            return ImbPipeline([("smote", smote), ("classifier", base)])
        else:
            # fallback: no SMOTE, plain sklearn pipeline
            return Pipeline([("classifier", base)])

    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        self._feature_names = list(X.columns)
        cv = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        f1_scores, auc_scores = [], []
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            pipe = self._build_pipeline()
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_val)
            y_prob = pipe.predict_proba(X_val)[:, 1]
            f1_scores.append(f1_score(y_val, y_pred, average="macro"))
            auc_scores.append(roc_auc_score(y_val, y_prob))
        self._model = pipe
        return {
            "f1_mean":    float(np.mean(f1_scores)),
            "f1_std":     float(np.std(f1_scores)),
            "f1_scores":  f1_scores,
            "auc_mean":   float(np.mean(auc_scores)),
            "auc_std":    float(np.std(auc_scores)),
            "auc_scores": auc_scores,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "VehiclePMClassifier":
        self._feature_names = list(X.columns)
        self._model = self._build_pipeline()
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self._model.predict_proba(X)

    def get_base_model(self):
        self._check_fitted()
        return self._model.named_steps["classifier"]

    def _check_fitted(self):
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() or cross_validate() first.")

    def __repr__(self):
        return (
            f"VehiclePMClassifier(model_type='{self.model_type}', "
            f"n_splits={self.n_splits}, calibrate={self.calibrate})"
        )
