"""
Adapter that converts raw OBDReading snapshots into the feature
matrix format expected by VehiclePMClassifier.

Since OBD-II provides only Group A (mechanical) and partial Group B
(driver behaviour) signals, this module:
    1. Maps OBD fields to Group A feature columns
    2. Computes rolling driver behaviour metrics from a reading history
    3. Accepts optional V2X / API context (weather, road, traffic)
    4. Fills missing values with safe defaults

The result is a single-row pd.DataFrame ready for model.predict().
"""

import time
import numpy as np
import pandas as pd
from collections import deque
from typing import Optional, Dict, Any, Deque
from dataclasses import dataclass

from vehiclepm.obd.reader import OBDReading


# ── Defaults for missing sensors ─────────────────────────────────────────────
# Conservative "unknown" values that do not falsely inflate risk score

_DEFAULTS = {
    # Group A
    "engine_temp":      90.0,   # normal operating temp
    "fuel_level":        0.5,
    "battery_health":    0.8,
    "brake_thickness":   6.0,   # mid-life (user must provide or estimate)
    "tire_tread":        4.0,   # mid-life
    "oil_degradation":   0.3,
    "mileage":           50_000,
    "vehicle_age":       5.0,
    "sensor_fault":      0,
    # Group B
    "hard_braking_freq": 1.0,
    "accel_variance":    1.0,
    "idle_ratio":        0.1,
    "driving_style":    "Smooth",
    # Group C
    "ambient_temp":      20.0,
    "road_roughness":     3.0,
    "monthly_precipitation": 50.0,
    "traffic_density":   20.0,
    "road_type":        "Urban",
    "weather_condition": "Clear",
}


@dataclass
class VehicleContext:
    """
    Optional external context to supplement OBD data.
    Provide as much as you have — missing fields use safe defaults.

    Parameters
    ----------
    mileage : float
        Total odometer reading in km.
    vehicle_age : float
        Vehicle age in years.
    brake_thickness : float
        Estimated brake pad thickness in mm.
    tire_tread : float
        Estimated tire tread depth in mm.
    oil_degradation : float
        Oil degradation index 0 (new) to 1 (degraded).
    ambient_temp : float
        Outside temperature in °C.
    road_roughness : float
        IRI road roughness index (m/km). 0=smooth, 10=very rough.
    traffic_density : float
        Vehicles per km.
    road_type : str
        'Highway', 'Urban', or 'Rural'.
    weather_condition : str
        'Clear', 'Rain', 'Snow', or 'Fog'.
    monthly_precipitation : float
        Cumulative monthly rainfall in mm.
    """
    mileage: float = 50_000
    vehicle_age: float = 5.0
    brake_thickness: float = 6.0
    tire_tread: float = 4.0
    oil_degradation: float = 0.3
    ambient_temp: float = 20.0
    road_roughness: float = 3.0
    traffic_density: float = 20.0
    road_type: str = "Urban"
    weather_condition: str = "Clear"
    monthly_precipitation: float = 50.0


class OBDFeatureAdapter:
    """
    Converts live OBDReading snapshots into a model-ready feature row.

    Maintains a rolling window of readings to compute driver behaviour
    metrics (hard braking frequency, acceleration variance, idle ratio).

    Parameters
    ----------
    window_size : int
        Number of readings to keep for rolling behaviour metrics. Default 60
        (1 minute at 1Hz sampling).
    context : VehicleContext or None
        External context (mileage, weather, road, etc.). If None, defaults
        are used for all non-OBD fields.

    Example
    -------
    >>> from vehiclepm.obd.reader import OBDReader
    >>> from vehiclepm.obd.adapter import OBDFeatureAdapter, VehicleContext
    >>> from vehiclepm import VehiclePMClassifier
    >>>
    >>> context = VehicleContext(
    ...     mileage=75000, vehicle_age=4, road_type="Urban",
    ...     weather_condition="Rain", road_roughness=5.0
    ... )
    >>> adapter = OBDFeatureAdapter(context=context)
    >>> clf = VehiclePMClassifier()
    >>> # clf.fit(X_train, y_train)  # pre-trained model
    >>>
    >>> with OBDReader() as reader:
    ...     for obd_reading in reader.stream(interval=1.0):
    ...         X = adapter.to_features(obd_reading)
    ...         prob = clf.predict_proba(X)[0, 1]
    ...         print(f"Maintenance probability: {prob:.1%}")
    """

    def __init__(
        self,
        window_size: int = 60,
        context: Optional[VehicleContext] = None,
        hard_brake_threshold: float = -3.0,   # m/s² deceleration threshold
    ):
        self.window_size = window_size
        self.context = context or VehicleContext()
        self.hard_brake_threshold = hard_brake_threshold

        self._history: Deque[OBDReading] = deque(maxlen=window_size)
        self._prev_speed: Optional[float] = None
        self._prev_time: Optional[float] = None
        self._hard_brake_events: int = 0
        self._speed_history: Deque[float] = deque(maxlen=window_size)

    def update(self, reading: OBDReading):
        """Add a new OBD reading to the rolling window."""
        self._history.append(reading)

        # Track hard braking events
        if self._prev_speed is not None and self._prev_time is not None:
            dt = reading.timestamp - self._prev_time
            if dt > 0 and reading.speed is not None and self._prev_speed is not None:
                # speed in km/h → m/s
                dv = (reading.speed - self._prev_speed) / 3.6
                accel = dv / dt  # m/s²
                if accel < self.hard_brake_threshold:
                    self._hard_brake_events += 1

        if reading.speed is not None:
            self._speed_history.append(reading.speed)
            self._prev_speed = reading.speed

        self._prev_time = reading.timestamp

    def _compute_driver_features(self) -> Dict[str, Any]:
        """Compute rolling driver behaviour metrics from history."""
        if len(self._history) < 2:
            return {
                "hard_braking_freq": _DEFAULTS["hard_braking_freq"],
                "accel_variance":    _DEFAULTS["accel_variance"],
                "idle_ratio":        _DEFAULTS["idle_ratio"],
                "driving_style":     _DEFAULTS["driving_style"],
            }

        # Hard braking frequency (events per hour)
        window_hours = len(self._history) / 3600.0
        hard_braking_freq = self._hard_brake_events / max(window_hours, 1 / 3600)

        # Acceleration variance from speed history
        speeds = np.array(list(self._speed_history))
        accel_variance = float(np.var(np.diff(speeds))) if len(speeds) > 1 else 0.0

        # Idle ratio (fraction of readings where speed < 2 km/h)
        idle_count = sum(1 for r in self._history if r.speed is not None and r.speed < 2)
        idle_ratio = idle_count / len(self._history)

        # Driving style heuristic
        if hard_braking_freq > 5 or accel_variance > 3:
            driving_style = "Aggressive"
        elif idle_ratio > 0.3:
            driving_style = "Stop-and-Go"
        else:
            driving_style = "Smooth"

        return {
            "hard_braking_freq": hard_braking_freq,
            "accel_variance":    accel_variance,
            "idle_ratio":        idle_ratio,
            "driving_style":     driving_style,
        }

    def to_features(self, reading: OBDReading) -> pd.DataFrame:
        """
        Convert an OBDReading into a single-row feature DataFrame.

        Automatically calls update() to maintain rolling driver metrics.

        Parameters
        ----------
        reading : OBDReading
            Latest sensor snapshot from OBDReader.

        Returns
        -------
        pd.DataFrame with shape (1, n_features)
            Ready to pass to VehiclePMClassifier.predict_proba().
        """
        self.update(reading)
        driver = self._compute_driver_features()
        ctx = self.context

        def get(val, default_key):
            return val if val is not None else _DEFAULTS[default_key]

        row = {
            # Group A — from OBD
            "engine_temp":      get(reading.engine_temp,     "engine_temp"),
            "fuel_level":       get(reading.fuel_level,      "fuel_level"),
            "battery_health":   get(reading.battery_health,  "battery_health"),
            "brake_thickness":  ctx.brake_thickness,
            "tire_tread":       ctx.tire_tread,
            "oil_degradation":  ctx.oil_degradation,
            "mileage":          ctx.mileage,
            "vehicle_age":      ctx.vehicle_age,
            "sensor_fault":     reading.sensor_fault,

            # Group B — rolling computed
            "hard_braking_freq": driver["hard_braking_freq"],
            "accel_variance":    driver["accel_variance"],
            "idle_ratio":        driver["idle_ratio"],
            "driving_style":     driver["driving_style"],

            # Group C — from context / V2X
            "ambient_temp":           ctx.ambient_temp,
            "road_roughness":         ctx.road_roughness,
            "monthly_precipitation":  ctx.monthly_precipitation,
            "traffic_density":        ctx.traffic_density,
            "road_type":              ctx.road_type,
            "weather_condition":      ctx.weather_condition,
        }

        return pd.DataFrame([row])
