"""
Feature engineering for vehicle predictive maintenance.

Four feature groups as defined in the paper:
    Group A — Internal Mechanical (9 features)
    Group B — Driver Behaviour    (4 features)
    Group C — Environmental/V2X   (6 features)
    Group D — Engineered Interactions (4 features)
"""

import pandas as pd
import numpy as np
from typing import Optional, List


# ─── Column name constants ────────────────────────────────────────────────────

MECHANICAL_FEATURES = [
    "engine_temp",
    "fuel_level",
    "battery_health",
    "brake_thickness",
    "tire_tread",
    "oil_degradation",
    "mileage",
    "vehicle_age",
    "sensor_fault",
]

DRIVER_FEATURES = [
    "hard_braking_freq",
    "accel_variance",
    "idle_ratio",
    "driving_style",  # Aggressive / Smooth / Stop-and-Go
]

ENVIRONMENTAL_FEATURES = [
    "ambient_temp",
    "road_roughness",
    "monthly_precipitation",
    "traffic_density",
    "road_type",
    "weather_condition",
]

INTERACTION_FEATURES = [
    "engine_thermal_load",
    "brake_stress_idx",
    "traffic_road_impact",
    "engine_battery_ratio",
]

ALL_FEATURE_GROUPS = {
    "mechanical": MECHANICAL_FEATURES,
    "driver": DRIVER_FEATURES,
    "environmental": ENVIRONMENTAL_FEATURES,
    "interactions": INTERACTION_FEATURES,
}


# ─── Per-group validators ─────────────────────────────────────────────────────

def _check_columns(df: pd.DataFrame, required: List[str], group: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {group} columns: {missing}")


# ─── Group A: Internal Mechanical ────────────────────────────────────────────

def compute_mechanical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return Group A (Internal Mechanical) features.

    Expected input columns:
        engine_temp, fuel_level, battery_health, brake_thickness,
        tire_tread, oil_degradation, mileage, vehicle_age, sensor_fault

    Returns a DataFrame with only the mechanical feature columns.
    """
    _check_columns(df, MECHANICAL_FEATURES, "mechanical")
    return df[MECHANICAL_FEATURES].copy()


# ─── Group B: Driver Behaviour ────────────────────────────────────────────────

def compute_driver_features(df: pd.DataFrame, encode_style: bool = True) -> pd.DataFrame:
    """
    Return Group B (Driver Behaviour) features.

    Expected input columns:
        hard_braking_freq  — events per hour (7-day rolling window)
        accel_variance     — variance of acceleration
        idle_ratio         — fraction of time idling
        driving_style      — categorical: Aggressive / Smooth / Stop-and-Go

    Parameters
    ----------
    encode_style : bool
        If True, one-hot encode the driving_style column.
    """
    _check_columns(df, DRIVER_FEATURES, "driver")
    out = df[DRIVER_FEATURES].copy()
    if encode_style:
        style_dummies = pd.get_dummies(out["driving_style"], prefix="driving_style")
        out = pd.concat([out.drop(columns=["driving_style"]), style_dummies], axis=1)
    return out


# ─── Group C: Environmental / V2X ────────────────────────────────────────────

def compute_environmental_features(
    df: pd.DataFrame, encode_categoricals: bool = True
) -> pd.DataFrame:
    """
    Return Group C (Environmental / V2X) features.

    Expected input columns:
        ambient_temp           — degrees Celsius
        road_roughness         — IRI scale (m/km)
        monthly_precipitation  — mm
        traffic_density        — vehicles per km
        road_type              — categorical: Highway / Urban / Rural
        weather_condition      — categorical: Clear / Rain / Snow / Fog

    Parameters
    ----------
    encode_categoricals : bool
        If True, one-hot encode road_type and weather_condition.
    """
    _check_columns(df, ENVIRONMENTAL_FEATURES, "environmental")
    out = df[ENVIRONMENTAL_FEATURES].copy()
    if encode_categoricals:
        for col in ["road_type", "weather_condition"]:
            dummies = pd.get_dummies(out[col], prefix=col)
            out = pd.concat([out.drop(columns=[col]), dummies], axis=1)
    return out


# ─── Group D: Engineered Interactions ────────────────────────────────────────

def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Group D (Engineered Interaction) features.

    Requires columns from Groups A, B, and C to be present in df:
        engine_temp, traffic_density, ambient_temp  → engine_thermal_load
        hard_braking_freq, road_roughness, brake_thickness → brake_stress_idx
        traffic_density, road_roughness             → traffic_road_impact
        engine_temp, battery_health                 → engine_battery_ratio

    Returns a DataFrame with only the four interaction feature columns.
    """
    required = [
        "engine_temp", "traffic_density", "ambient_temp",
        "hard_braking_freq", "road_roughness", "brake_thickness",
        "battery_health",
    ]
    _check_columns(df, required, "interaction")

    out = pd.DataFrame(index=df.index)

    # Engine thermal load: combined thermal stress
    out["engine_thermal_load"] = (
        df["engine_temp"] * 0.5
        + df["traffic_density"] * 0.3
        + df["ambient_temp"] * 0.2
    )

    # Brake stress index: mechanical + environmental brake pressure
    out["brake_stress_idx"] = (
        df["hard_braking_freq"] * 0.4
        + df["road_roughness"] * 0.4
        + (1 / (df["brake_thickness"].clip(lower=0.1))) * 0.2
    )

    # Traffic-road impact: combined road stress
    out["traffic_road_impact"] = df["traffic_density"] * df["road_roughness"]

    # Engine-to-battery ratio: electrical load indicator
    out["engine_battery_ratio"] = df["engine_temp"] / df["battery_health"].clip(lower=0.1)

    return out


# ─── Full feature matrix ──────────────────────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame,
    include_groups: Optional[List[str]] = None,
    encode_categoricals: bool = True,
) -> pd.DataFrame:
    """
    Build the complete feature matrix by combining all (or selected) groups.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame containing all required columns.
    include_groups : list of str, optional
        Subset of groups to include. Options: 'mechanical', 'driver',
        'environmental', 'interactions'. Defaults to all four.
    encode_categoricals : bool
        Whether to one-hot encode categorical columns.

    Returns
    -------
    pd.DataFrame
        Combined feature matrix ready for model training.

    Example
    -------
    >>> X = build_feature_matrix(df)
    >>> X = build_feature_matrix(df, include_groups=["mechanical", "environmental"])
    """
    if include_groups is None:
        include_groups = list(ALL_FEATURE_GROUPS.keys())

    parts = []

    if "mechanical" in include_groups:
        parts.append(compute_mechanical_features(df))

    if "driver" in include_groups:
        parts.append(compute_driver_features(df, encode_style=encode_categoricals))

    if "environmental" in include_groups:
        parts.append(
            compute_environmental_features(df, encode_categoricals=encode_categoricals)
        )

    if "interactions" in include_groups:
        parts.append(compute_interaction_features(df))

    if not parts:
        raise ValueError("No feature groups selected. Choose from: mechanical, driver, environmental, interactions.")

    return pd.concat(parts, axis=1)
