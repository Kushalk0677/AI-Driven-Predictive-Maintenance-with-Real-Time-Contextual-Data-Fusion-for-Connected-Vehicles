"""
Physics-informed synthetic dataset generator.

Replicates the probabilistic labelling scheme from the paper:
    risk = mechanical_risk + driver_risk + environmental_risk + Gaussian(0, sigma)
    label = 1 if risk > threshold else 0

This additive design ensures each feature group contributes an independent,
measurable fraction of the risk score — enabling valid ablation studies.
"""

import numpy as np
import pandas as pd
from typing import Optional


def generate_synthetic_dataset(
    n_samples: int = 2000,
    noise_sigma: float = 1.0,
    failure_rate_target: float = 0.3,
    random_state: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate a physics-informed synthetic vehicle maintenance dataset.

    Each row represents one vehicle-month observation. Labels are assigned
    probabilistically via a continuous risk score, preventing deterministic
    label leakage.

    Parameters
    ----------
    n_samples : int
        Number of vehicle-month observations to generate. Default 2000.
    noise_sigma : float
        Standard deviation of Gaussian noise injected into risk score
        before thresholding. Higher values produce more label uncertainty.
        Default 1.0 (baseline from paper).
    failure_rate_target : float
        Approximate fraction of samples labelled as requiring maintenance.
        Controls the risk threshold. Default 0.3.
    random_state : int or None
        Random seed for reproducibility. Default 42.

    Returns
    -------
    pd.DataFrame
        DataFrame with all raw feature columns plus:
            - 'risk_score'  : continuous risk score before thresholding
            - 'maintenance_needed' : binary label (1 = maintenance required)

    Example
    -------
    >>> from vehiclepm.data.synthetic import generate_synthetic_dataset
    >>> df = generate_synthetic_dataset(n_samples=2000, noise_sigma=1.0)
    >>> df["maintenance_needed"].value_counts(normalize=True)
    """
    rng = np.random.default_rng(random_state)

    # ── Group A: Internal Mechanical ─────────────────────────────────────────
    engine_temp        = rng.uniform(70, 120, n_samples)       # °C
    fuel_level         = rng.uniform(0.05, 1.0, n_samples)     # fraction
    battery_health     = rng.uniform(0.5, 1.0, n_samples)      # fraction SoH
    brake_thickness    = rng.uniform(1, 12, n_samples)          # mm
    tire_tread         = rng.uniform(1.6, 8, n_samples)         # mm
    oil_degradation    = rng.uniform(0, 1, n_samples)           # 0=new, 1=degraded
    mileage            = rng.uniform(0, 200_000, n_samples)     # km
    vehicle_age        = rng.uniform(0, 15, n_samples)          # years
    sensor_fault       = rng.choice([0, 1], n_samples, p=[0.95, 0.05])

    # ── Group B: Driver Behaviour ─────────────────────────────────────────────
    hard_braking_freq  = rng.uniform(0, 10, n_samples)          # events/hour
    accel_variance     = rng.uniform(0, 5, n_samples)
    idle_ratio         = rng.uniform(0, 0.5, n_samples)         # fraction
    driving_style      = rng.choice(
        ["Aggressive", "Smooth", "Stop-and-Go"], n_samples, p=[0.25, 0.5, 0.25]
    )

    # ── Group C: Environmental / V2X ─────────────────────────────────────────
    ambient_temp           = rng.uniform(-10, 45, n_samples)    # °C
    road_roughness         = rng.uniform(0, 10, n_samples)      # IRI m/km
    monthly_precipitation  = rng.uniform(0, 300, n_samples)     # mm
    traffic_density        = rng.uniform(0, 100, n_samples)     # vehicles/km
    road_type              = rng.choice(["Highway", "Urban", "Rural"], n_samples)
    weather_condition      = rng.choice(
        ["Clear", "Rain", "Snow", "Fog"], n_samples, p=[0.6, 0.25, 0.1, 0.05]
    )

    # ── Risk Score Components ─────────────────────────────────────────────────

    # Mechanical risk — threshold violations
    mechanical_risk = (
        (brake_thickness < 3).astype(float) * 2.0
        + (tire_tread < 2.5).astype(float) * 1.5
        + (battery_health < 0.6).astype(float) * 1.5
        + oil_degradation * 1.0
        + (mileage / 200_000) * 0.5
        + sensor_fault * 1.5
    )

    # Driver behaviour risk
    driving_style_risk = np.where(driving_style == "Aggressive", 1.5,
                         np.where(driving_style == "Stop-and-Go", 0.8, 0.2))
    driver_risk = (
        (hard_braking_freq / 10) * 1.0
        + (accel_variance / 5) * 0.5
        + driving_style_risk
    )

    # Environmental risk
    weather_risk = np.where(weather_condition == "Snow", 1.5,
                   np.where(weather_condition == "Rain", 1.0,
                   np.where(weather_condition == "Fog", 0.5, 0.0)))
    environmental_risk = (
        (road_roughness / 10) * 1.0
        + weather_risk
        + (ambient_temp < 0).astype(float) * 0.5
        + (traffic_density / 100) * 0.5
    )

    # Total risk with noise
    risk_score = mechanical_risk + driver_risk + environmental_risk
    risk_score += rng.normal(0, noise_sigma, n_samples)

    # Threshold to hit approximate target failure rate
    threshold = np.quantile(risk_score, 1 - failure_rate_target)
    maintenance_needed = (risk_score > threshold).astype(int)

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame({
        # Group A
        "engine_temp":       engine_temp,
        "fuel_level":        fuel_level,
        "battery_health":    battery_health,
        "brake_thickness":   brake_thickness,
        "tire_tread":        tire_tread,
        "oil_degradation":   oil_degradation,
        "mileage":           mileage,
        "vehicle_age":       vehicle_age,
        "sensor_fault":      sensor_fault,
        # Group B
        "hard_braking_freq": hard_braking_freq,
        "accel_variance":    accel_variance,
        "idle_ratio":        idle_ratio,
        "driving_style":     driving_style,
        # Group C
        "ambient_temp":          ambient_temp,
        "road_roughness":        road_roughness,
        "monthly_precipitation": monthly_precipitation,
        "traffic_density":       traffic_density,
        "road_type":             road_type,
        "weather_condition":     weather_condition,
        # Labels
        "risk_score":          risk_score,
        "maintenance_needed":  maintenance_needed,
    })

    return df
