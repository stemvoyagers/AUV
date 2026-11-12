"""AUV ground-truth simulator and synthetic sensor generator.

Public surface
--------------
``VehicleState``        ground-truth state container
``Trajectory``          analytic 120 s scenario
``SimulationClock``     fixed-step clock with multi-rate sensor scheduling
``Simulator``           trajectory + sensors + logging + export
``Accelerometer``       specific force in the body frame
``Gyroscope``           body-frame angular rate
``Magnetometer``        Earth field in the body frame
``PressureSensor``      hydrostatic pressure (-> depth)
"""

from .state import VehicleState
from .trajectory import Trajectory, Stage
from .simulation_clock import SimulationClock, ClockTick
from .simulator import Simulator, LogEntry
from .sensors import (
    Accelerometer,
    BaseSensor,
    Gyroscope,
    Magnetometer,
    PressureSensor,
    SensorReading,
)

__version__ = "0.2.0"

__all__ = [
    "VehicleState",
    "Trajectory", "Stage",
    "SimulationClock", "ClockTick",
    "Simulator", "LogEntry",
    "BaseSensor", "SensorReading",
    "Accelerometer", "Gyroscope", "Magnetometer", "PressureSensor",
]
