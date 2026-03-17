"""
OBD-II real-time data reader.

Connects to a vehicle's OBD-II port via USB or Bluetooth dongle
and reads live sensor data using the python-obd library.

Supported dongles:
    - ELM327 USB
    - ELM327 Bluetooth
    - Any OBD-II dongle compatible with python-obd

Install dependency:
    pip install obd
"""

import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class OBDReading:
    """
    A single snapshot of vehicle sensor data from the OBD-II port.
    All values are None if the sensor is not supported by the vehicle.
    """
    timestamp: float = field(default_factory=time.time)

    # Engine
    engine_temp: Optional[float] = None        # °C  (PID: COOLANT_TEMP)
    rpm: Optional[float] = None                # RPM
    throttle_pos: Optional[float] = None       # % (0–100)
    intake_temp: Optional[float] = None        # °C
    maf: Optional[float] = None                # g/s (Mass Air Flow)
    fuel_level: Optional[float] = None         # % (0–100) → normalised to 0–1
    engine_load: Optional[float] = None        # % (0–100)

    # Speed & movement
    speed: Optional[float] = None              # km/h

    # Battery (12V system)
    battery_voltage: Optional[float] = None   # V — proxy for battery health

    # Fault codes
    dtc_count: int = 0                         # number of active DTCs

    # Derived / computed after reading
    battery_health: Optional[float] = None    # normalised 0–1 from voltage
    sensor_fault: int = 0                      # 1 if DTCs present


class OBDReader:
    """
    Live OBD-II reader that connects to a vehicle dongle and
    returns structured OBDReading snapshots.

    Parameters
    ----------
    port : str or None
        Serial port (e.g. '/dev/ttyUSB0', 'COM3').
        If None, python-obd will auto-scan for the dongle.
    baudrate : int
        Serial baud rate. Default 38400 (ELM327 standard).
    timeout : float
        Connection timeout in seconds. Default 30.
    fast : bool
        If True, use python-obd fast mode (skips slow PIDs). Default True.

    Example
    -------
    >>> from vehiclepm.obd.reader import OBDReader
    >>> reader = OBDReader()
    >>> reader.connect()
    >>> reading = reader.read()
    >>> print(reading.engine_temp)
    >>> reader.disconnect()

    Or use as a context manager:
    >>> with OBDReader() as reader:
    ...     reading = reader.read()
    """

    # OBD PID command names (python-obd uses these strings)
    _PID_MAP = {
        "engine_temp":    "COOLANT_TEMP",
        "rpm":            "RPM",
        "throttle_pos":   "THROTTLE_POS",
        "intake_temp":    "INTAKE_TEMP",
        "maf":            "MAF",
        "fuel_level":     "FUEL_LEVEL",
        "engine_load":    "ENGINE_LOAD",
        "speed":          "SPEED",
        "battery_voltage": "CONTROL_MODULE_VOLTAGE",
    }

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 38400,
        timeout: float = 30,
        fast: bool = True,
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.fast = fast
        self._connection = None

    def connect(self) -> bool:
        """
        Connect to the OBD-II dongle.

        Returns
        -------
        bool : True if connected successfully, False otherwise.
        """
        try:
            import obd
        except ImportError:
            raise ImportError(
                "python-obd is required for OBD-II support. "
                "Install it with: pip install obd"
            )

        kwargs = {
            "baudrate": self.baudrate,
            "timeout":  self.timeout,
            "fast":     self.fast,
        }
        if self.port:
            kwargs["portstr"] = self.port

        self._connection = obd.OBD(**kwargs)

        if self._connection.is_connected():
            logger.info(f"Connected to OBD-II dongle on {self._connection.port_name()}")
            return True
        else:
            logger.error("Failed to connect to OBD-II dongle. Check port and dongle.")
            return False

    def disconnect(self):
        """Close the OBD-II connection."""
        if self._connection:
            self._connection.close()
            logger.info("OBD-II connection closed.")

    def read(self) -> OBDReading:
        """
        Read a single snapshot of all supported sensor values.

        Returns
        -------
        OBDReading : dataclass with all available sensor values.
        """
        if not self._connection or not self._connection.is_connected():
            raise ConnectionError(
                "Not connected to OBD-II dongle. Call connect() first."
            )

        import obd

        reading = OBDReading()

        for field_name, pid in self._PID_MAP.items():
            try:
                cmd = obd.commands[pid]
                response = self._connection.query(cmd)
                if not response.is_null():
                    val = response.value.magnitude  # pint quantity → float
                    setattr(reading, field_name, float(val))
            except Exception as e:
                logger.debug(f"Could not read PID {pid}: {e}")

        # Read active DTCs
        try:
            dtc_response = self._connection.query(obd.commands.GET_DTC)
            if not dtc_response.is_null():
                reading.dtc_count = len(dtc_response.value)
                reading.sensor_fault = 1 if reading.dtc_count > 0 else 0
        except Exception as e:
            logger.debug(f"Could not read DTCs: {e}")

        # Derive battery health from voltage
        # 12.6V = 100% SoH, 11.8V = ~50%, below 11.5V = critical
        if reading.battery_voltage is not None:
            v = reading.battery_voltage
            reading.battery_health = min(1.0, max(0.0, (v - 11.5) / (12.8 - 11.5)))

        # Normalise fuel level to 0–1
        if reading.fuel_level is not None:
            reading.fuel_level = reading.fuel_level / 100.0

        return reading

    def stream(self, interval: float = 1.0):
        """
        Generator that yields OBDReading snapshots at a fixed interval.

        Parameters
        ----------
        interval : float
            Seconds between readings. Default 1.0.

        Example
        -------
        >>> with OBDReader() as reader:
        ...     for reading in reader.stream(interval=2.0):
        ...         print(reading.engine_temp)
        """
        while True:
            yield self.read()
            time.sleep(interval)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()
