"""
vehiclepm.obd.logger
====================
Data logging and replay module for live OBD-II drives.

Logs all OBD sensor readings + weather + GPS context to CSV.
Replays logged drives through the model offline.
"""

import os
import time
import csv
import json
import logging
import requests
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Weather fetcher
# ─────────────────────────────────────────────────────────────────────────────

def get_location() -> Optional[Dict[str, float]]:
    """
    Get approximate laptop location from IP address.
    Returns {'lat': float, 'lon': float, 'city': str} or None.
    """
    try:
        resp = requests.get("http://ip-api.com/json/", timeout=5)
        data = resp.json()
        if data.get("status") == "success":
            return {
                "lat":  data["lat"],
                "lon":  data["lon"],
                "city": data.get("city", "Unknown"),
            }
    except Exception as e:
        logger.warning(f"Could not get location: {e}")
    return None


def get_weather(lat: float, lon: float, api_key: str) -> Dict[str, Any]:
    """
    Fetch current weather from OpenWeatherMap.
    Returns dict with ambient_temp, weather_condition, precipitation, wind_speed.

    Get a free API key at: https://openweathermap.org/api
    Free tier: 1000 calls/day
    """
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        )
        resp = requests.get(url, timeout=5)
        data = resp.json()

        main_weather = data["weather"][0]["main"]  # Clear, Rain, Snow, Clouds, etc.
        condition_map = {
            "Clear":        "Clear",
            "Clouds":       "Clear",
            "Rain":         "Rain",
            "Drizzle":      "Rain",
            "Thunderstorm": "Rain",
            "Snow":         "Snow",
            "Mist":         "Fog",
            "Fog":          "Fog",
            "Haze":         "Fog",
        }
        weather_condition = condition_map.get(main_weather, "Clear")

        # Precipitation in last 1h (mm), default 0 if not raining
        precip = data.get("rain", {}).get("1h", 0.0)

        return {
            "ambient_temp":           round(data["main"]["temp"], 1),
            "weather_condition":      weather_condition,
            "weather_raw":            main_weather,
            "monthly_precipitation":  round(precip * 24 * 30, 1),
            "wind_speed":             round(data["wind"]["speed"], 1),
            "humidity":               round(data["main"]["humidity"], 1),
            "pressure_hpa":           round(data["main"]["pressure"], 1),
        }
    except Exception as e:
        logger.warning(f"Could not fetch weather: {e}")
        return {
            "ambient_temp":          25.0,
            "weather_condition":     "Clear",
            "weather_raw":           "Unknown (no API key)",
            "monthly_precipitation": 40.0,
            "wind_speed":            0.0,
            "humidity":              50.0,
            "pressure_hpa":          1013.0,
        }


def estimate_road_roughness(speed_history: list) -> float:
    """
    Estimate road roughness from speed variance.
    Rough roads cause more speed fluctuation even at constant throttle.

    Returns IRI-scale estimate (0=smooth, 10=very rough).
    """
    import numpy as np
    if len(speed_history) < 3:
        return 3.0  # default urban
    variance = float(np.var(np.diff(speed_history)))
    # Map variance to IRI scale empirically
    roughness = min(10.0, max(0.0, variance * 0.5))
    return round(roughness, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Log row dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DriveLogRow:
    """One row of a logged drive — all sensor + context + prediction data."""
    timestamp:              str
    elapsed_seconds:        float

    # OBD sensors
    engine_temp:            float
    rpm:                    float
    speed:                  float
    fuel_level:             float
    engine_load:            float
    throttle_pos:           float
    battery_voltage:        float
    battery_health:         float
    dtc_count:              int
    sensor_fault:           int

    # Driver behaviour (rolling)
    hard_braking_freq:      float
    accel_variance:         float
    idle_ratio:             float
    driving_style:          str

    # Context / Weather
    ambient_temp:           float
    weather_condition:      str
    weather_raw:            str
    monthly_precipitation:  float
    wind_speed:             float
    humidity:               float
    pressure_hpa:           float
    road_roughness:         float
    traffic_density:        float
    road_type:              str

    # Vehicle info
    mileage:                float
    vehicle_age:            float
    brake_thickness:        float
    tire_tread:             float
    oil_degradation:        float

    # Location
    latitude:               float
    longitude:              float
    city:                   str

    # Prediction
    maintenance_probability: float
    severity:               str
    needs_maintenance:      bool


# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────

class DriveLogger:
    """
    Logs a live OBD drive to CSV with weather and location context.

    Parameters
    ----------
    output_path : str
        Path to output CSV file.
        Default: logs/YYYY-MM-DD_HH-MM-SS_drive.csv
    weather_api_key : str or None
        OpenWeatherMap API key. Get free at openweathermap.org/api
        If None, uses default weather values.
    weather_interval : float
        Seconds between weather API calls. Default 60.

    Example
    -------
    >>> from vehiclepm.obd.logger import DriveLogger
    >>> logger = DriveLogger(weather_api_key="your_key_here")
    >>> logger.start()
    >>> logger.log_row(obd_reading, adapter, clf, training_columns)
    >>> logger.stop()
    """

    def __init__(
        self,
        output_path: Optional[str] = None,
        weather_api_key: Optional[str] = None,
        weather_interval: float = 60.0,
    ):
        if output_path is None:
            os.makedirs("logs", exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_path = f"logs/{ts}_drive.csv"

        self.output_path     = output_path
        self.api_key         = weather_api_key
        self.weather_interval = weather_interval

        self._start_time     = None
        self._csv_file       = None
        self._csv_writer     = None
        self._location       = None
        self._weather        = {}
        self._last_weather_t = 0
        self._speed_history  = []
        self._row_count      = 0

    def start(self):
        """Open CSV file and fetch initial location + weather."""
        self._start_time = time.time()

        # Get location once
        print("  📍 Getting location...")
        self._location = get_location()
        if self._location:
            print(f"     Location: {self._location['city']} "
                  f"({self._location['lat']:.2f}, {self._location['lon']:.2f})")
        else:
            print("     Could not get location — using defaults")
            self._location = {"lat": 0.0, "lon": 0.0, "city": "Unknown"}

        # Get initial weather
        if self.api_key:
            print("  🌤  Fetching weather...")
            self._weather = get_weather(
                self._location["lat"], self._location["lon"], self.api_key
            )
            print(f"     {self._weather['weather_raw']} | "
                  f"{self._weather['ambient_temp']}°C | "
                  f"Wind: {self._weather['wind_speed']} m/s")
        else:
            self._weather = {
                "ambient_temp": 25.0,
                "weather_condition": "Clear",
                "weather_raw": "Unknown (no API key)",
                "monthly_precipitation": 40.0,
                "wind_speed": 0.0,
                "humidity": 50.0,
                "pressure_hpa": 1013.0,
            }
            print("  ⚠️  No weather API key — using defaults")

        # Open CSV
        self._csv_file = open(self.output_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=[f.name for f in DriveLogRow.__dataclass_fields__.values()]
        )
        self._csv_writer.writeheader()
        print(f"\n  📝 Logging to: {self.output_path}")

    def log_row(
        self,
        obd_reading,
        adapter,
        clf,
        training_columns: list,
        context,
    ):
        """
        Log one reading to CSV.

        Parameters
        ----------
        obd_reading : OBDReading
        adapter : OBDFeatureAdapter
        clf : fitted VehiclePMClassifier
        training_columns : list of str
        context : VehicleContext
        """
        import pandas as pd
        from vehiclepm.features.engineering import build_feature_matrix
        from vehiclepm.obd.live import _get_severity

        # Refresh weather every N seconds
        now = time.time()
        if self.api_key and (now - self._last_weather_t) > self.weather_interval:
            self._weather = get_weather(
                self._location["lat"], self._location["lon"], self.api_key
            )
            self._last_weather_t = now

        # Speed history for road roughness
        if obd_reading.speed is not None:
            self._speed_history.append(obd_reading.speed)
            if len(self._speed_history) > 60:
                self._speed_history.pop(0)

        road_roughness = estimate_road_roughness(self._speed_history)

        # Features + prediction
        raw_df = adapter.to_features(obd_reading)
        X_live = build_feature_matrix(raw_df)
        for col in training_columns:
            if col not in X_live.columns:
                X_live[col] = 0
        X_live = X_live[training_columns]

        prob     = float(clf.predict_proba(X_live)[0, 1])
        severity = _get_severity(prob)
        driver   = adapter._compute_driver_features()

        row = DriveLogRow(
            timestamp              = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            elapsed_seconds        = round(now - self._start_time, 1),
            engine_temp            = obd_reading.engine_temp or 0.0,
            rpm                    = obd_reading.rpm or 0.0,
            speed                  = obd_reading.speed or 0.0,
            fuel_level             = obd_reading.fuel_level or 0.0,
            engine_load            = obd_reading.engine_load or 0.0,
            throttle_pos           = obd_reading.throttle_pos or 0.0,
            battery_voltage        = obd_reading.battery_voltage or 12.6,
            battery_health         = obd_reading.battery_health or 0.8,
            dtc_count              = obd_reading.dtc_count,
            sensor_fault           = obd_reading.sensor_fault,
            hard_braking_freq      = driver["hard_braking_freq"],
            accel_variance         = driver["accel_variance"],
            idle_ratio             = driver["idle_ratio"],
            driving_style          = driver["driving_style"],
            ambient_temp           = self._weather["ambient_temp"],
            weather_condition      = self._weather["weather_condition"],
            weather_raw            = self._weather.get("weather_raw", "Unknown"),
            monthly_precipitation  = self._weather["monthly_precipitation"],
            wind_speed             = self._weather.get("wind_speed", 0.0),
            humidity               = self._weather.get("humidity", 50.0),
            pressure_hpa           = self._weather.get("pressure_hpa", 1013.0),
            road_roughness         = road_roughness,
            traffic_density        = context.traffic_density,
            road_type              = context.road_type,
            mileage                = context.mileage,
            vehicle_age            = context.vehicle_age,
            brake_thickness        = context.brake_thickness,
            tire_tread             = context.tire_tread,
            oil_degradation        = context.oil_degradation,
            latitude               = self._location["lat"],
            longitude              = self._location["lon"],
            city                   = self._location["city"],
            maintenance_probability = round(prob, 4),
            severity               = severity,
            needs_maintenance      = prob >= 0.5,
        )

        self._csv_writer.writerow(asdict(row))
        self._csv_file.flush()
        self._row_count += 1
        return row

    def stop(self):
        """Close the CSV file."""
        if self._csv_file:
            self._csv_file.close()
        elapsed = time.time() - self._start_time
        print(f"\n  ✅ Drive logged: {self._row_count} readings in {elapsed:.0f}s")
        print(f"     Saved to: {self.output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Replay
# ─────────────────────────────────────────────────────────────────────────────

def replay_drive(
    csv_path: str,
    speed: float = 1.0,
    show_summary: bool = True,
):
    """
    Replay a logged drive CSV through the console.

    Parameters
    ----------
    csv_path : str
        Path to a drive log CSV produced by DriveLogger.
    speed : float
        Playback speed multiplier. 1.0 = real time, 10.0 = 10x faster.
        0 = instant (no delay).
    show_summary : bool
        Print summary statistics after replay.

    Example
    -------
    >>> from vehiclepm.obd.logger import replay_drive
    >>> replay_drive("logs/2026-03-18_drive.csv", speed=5.0)
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    total = len(df)

    colours = {
        "OK":       "\033[92m",
        "WATCH":    "\033[93m",
        "WARNING":  "\033[33m",
        "CRITICAL": "\033[91m",
    }
    icons = {"OK": "✅", "WATCH": "👀", "WARNING": "⚠️ ", "CRITICAL": "🚨"}
    reset = "\033[0m"

    print(f"\n{'='*65}")
    print(f"  🚗 Drive Replay: {os.path.basename(csv_path)}")
    print(f"  {total} readings | Speed: {speed}x")
    print(f"{'='*65}\n")
    print(f"  {'Time':<10} {'Sev':<10} {'Risk':<8} {'Engine':<10} "
          f"{'Speed':<10} {'Battery':<10} {'DTCs'}")
    print("  " + "-" * 63)

    prev_elapsed = 0.0

    for _, row in df.iterrows():
        severity = row["severity"]
        colour   = colours.get(severity, "")
        icon     = icons.get(severity, "")

        elapsed  = float(row["elapsed_seconds"])
        mins     = int(elapsed // 60)
        secs     = int(elapsed % 60)
        time_str = f"{mins:02d}:{secs:02d}"

        prob     = float(row["maintenance_probability"])
        eng_temp = float(row["engine_temp"])
        speed_v  = float(row["speed"])
        bat      = float(row["battery_health"])
        dtcs     = int(row["dtc_count"])

        print(
            f"  {time_str:<10} "
            f"{colour}{icon} {severity:<8}{reset} "
            f"{f'{prob:.1%}':<8} "
            f"{eng_temp:.1f}°C{'':>4} "
            f"{speed_v:.0f} km/h{'':>3} "
            f"{bat:.0%}{'':>6} "
            f"{dtcs}"
        )

        # Paced playback
        if speed > 0:
            delay = (elapsed - prev_elapsed) / speed
            if delay > 0:
                time.sleep(min(delay, 2.0))
        prev_elapsed = elapsed

    if show_summary:
        print(f"\n{'='*65}")
        print("  📊 Drive Summary")
        print(f"{'='*65}")
        print(f"  Duration:        {df['elapsed_seconds'].max():.0f}s "
              f"({df['elapsed_seconds'].max()/60:.1f} min)")
        print(f"  Avg speed:       {df['speed'].mean():.1f} km/h")
        print(f"  Max engine temp: {df['engine_temp'].max():.1f}°C")
        print(f"  Avg risk:        {df['maintenance_probability'].mean():.1%}")
        print(f"  Max risk:        {df['maintenance_probability'].max():.1%}")
        print(f"  Severity counts:")
        for sev, count in df["severity"].value_counts().items():
            icon = icons.get(sev, "")
            print(f"    {icon} {sev}: {count} readings ({count/total:.0%})")
        print(f"  Driving style:   {df['driving_style'].mode()[0]}")
        print(f"  Avg road rough:  {df['road_roughness'].mean():.1f} IRI")
        print(f"{'='*65}\n")
