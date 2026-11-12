from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .sensors.base import BaseSensor, SensorReading
from .simulation_clock import SimulationClock
from .state import VehicleState
from .trajectory import Trajectory

GROUND_TRUTH_COLUMNS = (
    "t", "north", "east", "depth",
    "v_n", "v_e", "v_d",
    "yaw_rad", "pitch_rad", "roll_rad",
)


@dataclass
class LogEntry:
    t: float
    state: VehicleState
    readings: Dict[str, SensorReading] = field(default_factory=dict)


class Simulator:
    def __init__(self,
                 trajectory: Optional[Trajectory] = None,
                 dt_physics: float = 0.005,
                 seed: Optional[int] = 42):
        self.trajectory = trajectory or Trajectory()
        self.dt_physics = float(dt_physics)

        self._sensors: Dict[str, tuple[BaseSensor, float]] = {}
        self._rng = np.random.default_rng(seed)

        self.history: List[LogEntry] = []

    def add_sensor(self, sensor: BaseSensor, frequency_hz: float) -> None:
        if frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive")
        sensor._rng = self._rng
        self._sensors[sensor.name] = (sensor, 1.0 / float(frequency_hz))

    @property
    def sensor_names(self) -> list[str]:
        return list(self._sensors)

    def run(self, max_time: Optional[float] = None) -> List[LogEntry]:
        self.history.clear()
        for sensor, _ in self._sensors.values():
            sensor.reset()

        duration = self.trajectory.duration if max_time is None else float(max_time)
        clock = SimulationClock(
            duration=duration,
            dt_physics=self.dt_physics,
            sensor_periods={name: period for name, (_, period) in self._sensors.items()},
        )

        for tick in clock:
            state = self.trajectory.state_at(tick.t)
            readings: Dict[str, SensorReading] = {}

            for name in tick.due_sensors:
                sensor, period = self._sensors[name]
                reading = sensor.read(state, period)
                if reading is not None:
                    readings[name] = reading

            self.history.append(LogEntry(t=tick.t, state=state, readings=readings))

        return self.history

    def ground_truth_arrays(self) -> dict:
        states = [h.state for h in self.history]
        return {
            "t": np.array([s.t for s in states]),
            "position": np.vstack([s.position for s in states]),
            "velocity": np.vstack([s.velocity for s in states]),
            "depth": np.array([s.depth for s in states]),
            "yaw": np.array([s.yaw_pitch_roll[0] for s in states]),
            "pitch": np.array([s.yaw_pitch_roll[1] for s in states]),
            "roll": np.array([s.yaw_pitch_roll[2] for s in states]),
        }

    def sensor_arrays(self, name: str) -> Optional[dict]:
        entries = [h for h in self.history if name in h.readings]
        if not entries:
            return None
        first = entries[0].readings[name]
        return {
            "t": np.array([h.readings[name].t for h in entries]),
            "data": np.vstack([h.readings[name].data for h in entries]),
            "ideal": (np.vstack([h.readings[name].data_ideal for h in entries])
                      if first.data_ideal is not None else None),
        }

    def export_ground_truth_csv(self, path) -> None:
        gt = self.ground_truth_arrays()
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(GROUND_TRUTH_COLUMNS)
            for i in range(len(gt["t"])):
                writer.writerow([
                    f"{gt['t'][i]:.6f}",
                    f"{gt['position'][i, 0]:.8f}",
                    f"{gt['position'][i, 1]:.8f}",
                    f"{gt['position'][i, 2]:.8f}",
                    f"{gt['velocity'][i, 0]:.8f}",
                    f"{gt['velocity'][i, 1]:.8f}",
                    f"{gt['velocity'][i, 2]:.8f}",
                    f"{gt['yaw'][i]:.10f}",
                    f"{gt['pitch'][i]:.10f}",
                    f"{gt['roll'][i]:.10f}",
                ])

    def export_sensor_csv(self, name: str, path) -> None:
        arr = self.sensor_arrays(name)
        if arr is None:
            raise KeyError(f"No readings for sensor '{name}' in history")

        data = arr["data"]
        n_cols = 1 if data.ndim == 1 else data.shape[1]
        columns = ["t"] + [f"{name}_{axis}" for axis in range(n_cols)]

        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(columns)
            for t_row, d_row in zip(arr["t"], data):
                writer.writerow([f"{float(t_row):.6f}"]
                                + [f"{float(v):.8f}" for v in np.atleast_1d(d_row)])

    def export_json(self, path) -> None:
        gt = self.ground_truth_arrays()
        bundle = {
            "physics_dt": self.dt_physics,
            "trajectory_duration": self.trajectory.duration,
            "ground_truth": {
                "t": gt["t"].tolist(),
                "position": gt["position"].tolist(),
                "velocity": gt["velocity"].tolist(),
                "yaw": gt["yaw"].tolist(),
                "pitch": gt["pitch"].tolist(),
                "roll": gt["roll"].tolist(),
            },
            "sensors": {},
        }

        for name, (_, period) in self._sensors.items():
            arr = self.sensor_arrays(name)
            if arr is None:
                continue
            bundle["sensors"][name] = {
                "frequency_hz": 1.0 / period,
                "t": arr["t"].tolist(),
                "data": arr["data"].tolist(),
            }

        with open(path, "w") as fh:
            json.dump(bundle, fh, indent=2)
