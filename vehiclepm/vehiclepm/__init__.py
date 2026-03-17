"""
vehiclepm — Predictive Maintenance for Connected Vehicles
==========================================================
Plug in your OBD-II dongle and get live maintenance predictions.

Quick start:
    >>> from vehiclepm import VehiclePMClassifier, generate_synthetic_dataset, LivePredictor
    >>> from vehiclepm.features import build_feature_matrix
    >>> from vehiclepm.obd import VehicleContext
    >>>
    >>> # Train
    >>> df = generate_synthetic_dataset()
    >>> X = build_feature_matrix(df.drop(columns=["risk_score","maintenance_needed"]))
    >>> y = df["maintenance_needed"]
    >>> clf = VehiclePMClassifier()
    >>> clf.fit(X, y)
    >>>
    >>> # Live prediction from OBD dongle
    >>> ctx = VehicleContext(mileage=75000, vehicle_age=5, weather_condition="Rain")
    >>> predictor = LivePredictor(model=clf, context=ctx)
    >>> predictor.run()
"""

__version__ = "0.2.0"
__author__ = "Kushal Khemani"
__email__ = "kushal.khemani@gmail.com"

from vehiclepm.models.classifier import VehiclePMClassifier
from vehiclepm.data.synthetic import generate_synthetic_dataset
from vehiclepm.evaluation.noise import noise_sensitivity_analysis
from vehiclepm.obd.live import LivePredictor
from vehiclepm.obd.adapter import VehicleContext

__all__ = [
    "VehiclePMClassifier",
    "generate_synthetic_dataset",
    "noise_sensitivity_analysis",
    "LivePredictor",
    "VehicleContext",
]
