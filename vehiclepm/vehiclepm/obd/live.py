"""
Live inference engine.

Connects OBD reader → feature adapter → trained model → alert system
into a single real-time prediction loop.

This is the main entry point for real-world deployment.
"""

import time
import logging
from typing import Optional, Callable
from dataclasses import dataclass

from vehiclepm.obd.reader import OBDReader
from vehiclepm.obd.adapter import OBDFeatureAdapter, VehicleContext
from vehiclepm.features.engineering import build_feature_matrix

logger = logging.getLogger(__name__)


@dataclass
class MaintenanceAlert:
    """A maintenance prediction result."""
    timestamp: float
    probability: float          # 0.0 – 1.0
    needs_maintenance: bool
    severity: str               # 'OK', 'WATCH', 'WARNING', 'CRITICAL'
    engine_temp: Optional[float]
    battery_health: Optional[float]
    sensor_fault: int
    dtc_count: int
    driving_style: str

    def __str__(self):
        return (
            f"[{self.severity}] Maintenance probability: {self.probability:.1%} | "
            f"Engine: {self.engine_temp:.1f}°C | "
            f"Battery: {self.battery_health:.0%} | "
            f"DTCs: {self.dtc_count} | "
            f"Style: {self.driving_style}"
        )


def _get_severity(prob: float) -> str:
    if prob < 0.3:
        return "OK"
    elif prob < 0.5:
        return "WATCH"
    elif prob < 0.75:
        return "WARNING"
    else:
        return "CRITICAL"


class LivePredictor:
    """
    Real-time maintenance predictor that reads from an OBD-II dongle
    and produces continuous maintenance probability scores.

    Parameters
    ----------
    model : fitted VehiclePMClassifier
        Pre-trained model. Call clf.fit(X_train, y_train) before passing here.
    context : VehicleContext or None
        Vehicle-specific context (mileage, tyres, road, weather).
        Update this whenever conditions change.
    port : str or None
        OBD serial port. None = auto-detect.
    interval : float
        Seconds between predictions. Default 5.0.
    alert_threshold : float
        Probability above which needs_maintenance is True. Default 0.5.
    on_alert : callable or None
        Optional callback called on every prediction with a MaintenanceAlert.
        If None, results are printed to stdout.

    Example — basic usage
    ----------------------
    >>> from vehiclepm import VehiclePMClassifier, generate_synthetic_dataset
    >>> from vehiclepm.features import build_feature_matrix
    >>> from vehiclepm.obd.live import LivePredictor
    >>> from vehiclepm.obd.adapter import VehicleContext
    >>>
    >>> # 1. Train the model
    >>> df = generate_synthetic_dataset(n_samples=2000)
    >>> X = build_feature_matrix(df.drop(columns=["risk_score","maintenance_needed"]))
    >>> y = df["maintenance_needed"]
    >>> clf = VehiclePMClassifier()
    >>> clf.fit(X, y)
    >>>
    >>> # 2. Set your vehicle context
    >>> ctx = VehicleContext(
    ...     mileage=82000,
    ...     vehicle_age=6,
    ...     brake_thickness=4.0,
    ...     tire_tread=3.5,
    ...     oil_degradation=0.6,
    ...     weather_condition="Rain",
    ...     road_roughness=5.0,
    ...     road_type="Urban",
    ... )
    >>>
    >>> # 3. Plug in OBD dongle and run
    >>> predictor = LivePredictor(model=clf, context=ctx)
    >>> predictor.run()   # runs until Ctrl+C

    Example — custom alert callback
    --------------------------------
    >>> def my_alert(alert):
    ...     if alert.severity in ("WARNING", "CRITICAL"):
    ...         send_push_notification(f"Service needed! {alert.probability:.0%}")
    >>>
    >>> predictor = LivePredictor(model=clf, context=ctx, on_alert=my_alert)
    >>> predictor.run()
    """

    def __init__(
        self,
        model,
        context: Optional[VehicleContext] = None,
        port: Optional[str] = None,
        interval: float = 5.0,
        alert_threshold: float = 0.5,
        on_alert: Optional[Callable[[MaintenanceAlert], None]] = None,
    ):
        self.model = model
        self.context = context or VehicleContext()
        self.port = port
        self.interval = interval
        self.alert_threshold = alert_threshold
        self.on_alert = on_alert or self._default_print

    def _align_columns(self, X):
        """Align live feature columns to match training columns exactly."""
        if not hasattr(self.model, '_feature_names') or self.model._feature_names is None:
            return X
        for col in self.model._feature_names:
            if col not in X.columns:
                X[col] = 0
        return X[self.model._feature_names]

    def run(self, max_readings: Optional[int] = None):
        """
        Start the live prediction loop.

        Parameters
        ----------
        max_readings : int or None
            Stop after this many readings. None = run forever (until Ctrl+C).
        """
        adapter = OBDFeatureAdapter(context=self.context)
        count = 0

        print("vehiclepm — Live Maintenance Predictor")
        print("Connecting to OBD-II dongle...")

        with OBDReader(port=self.port) as reader:
            print(f"Connected. Predicting every {self.interval}s. Press Ctrl+C to stop.\n")

            for obd_reading in reader.stream(interval=self.interval):
                try:
                    # Build feature row
                    raw_df = adapter.to_features(obd_reading)
                    X = build_feature_matrix(raw_df)
                    X = self._align_columns(X)

                    # Predict
                    prob = float(self.model.predict_proba(X)[0, 1])
                    severity = _get_severity(prob)

                    driver = adapter._compute_driver_features()

                    alert = MaintenanceAlert(
                        timestamp=obd_reading.timestamp,
                        probability=prob,
                        needs_maintenance=prob >= self.alert_threshold,
                        severity=severity,
                        engine_temp=obd_reading.engine_temp,
                        battery_health=obd_reading.battery_health,
                        sensor_fault=obd_reading.sensor_fault,
                        dtc_count=obd_reading.dtc_count,
                        driving_style=driver["driving_style"],
                    )

                    self.on_alert(alert)
                    count += 1

                    if max_readings and count >= max_readings:
                        break

                except Exception as e:
                    logger.error(f"Prediction error: {e}")
                    continue

    @staticmethod
    def _default_print(alert: MaintenanceAlert):
        """Default alert handler — prints to console with colour coding."""
        colours = {
            "OK":       "\033[92m",   # green
            "WATCH":    "\033[93m",   # yellow
            "WARNING":  "\033[33m",   # orange
            "CRITICAL": "\033[91m",   # red
        }
        reset = "\033[0m"
        colour = colours.get(alert.severity, "")
        print(f"{colour}{alert}{reset}")

    def predict_once(self, obd_reading) -> MaintenanceAlert:
        """
        Run a single prediction from an OBDReading without starting the loop.
        Useful for testing or embedding in your own application.

        Parameters
        ----------
        obd_reading : OBDReading

        Returns
        -------
        MaintenanceAlert
        """
        adapter = OBDFeatureAdapter(context=self.context)
        raw_df = adapter.to_features(obd_reading)
        X = build_feature_matrix(raw_df)
        X = self._align_columns(X)
        prob = float(self.model.predict_proba(X)[0, 1])
        driver = adapter._compute_driver_features()

        return MaintenanceAlert(
            timestamp=obd_reading.timestamp,
            probability=prob,
            needs_maintenance=prob >= self.alert_threshold,
            severity=_get_severity(prob),
            engine_temp=obd_reading.engine_temp,
            battery_health=obd_reading.battery_health,
            sensor_fault=obd_reading.sensor_fault,
            dtc_count=obd_reading.dtc_count,
            driving_style=driver["driving_style"],
        )
