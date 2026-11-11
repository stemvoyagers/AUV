"""Central simulator.

The simulator owns:
    * the analytic Trajectory (ground-truth state generator),
    * every configured sensor with its own nominal sampling frequency,
    * a fixed physics tick dt_physics.

On each physics tick we advance simulation time by dt_physics and:
    1. Ask the Trajectory for the new ground-truth state.
    2. For each sensor whose sample timer has expired, call its ``read()``.
    3. Collect every SensorReading alongside the ground-truth snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..core.state import VehicleState
from ..core.trajectory import Trajectory
from ..sensors.base import BaseSensor, SensorReading


@dataclass
class LogEntry:
    t: float
    state: VehicleState
    readings: Dict[str, SensorReading] = field(default_factory=dict)


class Simulator:
    """Orchestrates trajectory + multi-rate sensors."""

    def __init__(self,
                 trajectory: Optional[Trajectory] = None,
                 dt_physics: float = 0.005,
                 seed: Optional[int] = 42):
        self.trajectory = trajectory or Trajectory()
        self.dt_physics = dt_physics

        self._sensors: Dict[str, tuple[BaseSensor, float, float]] = {}   # (sensor, dt_sample, next_t)
        self._rng = np.random.default_rng(seed)

        self.history: List[LogEntry] = []

    # ------------------------------------------------------------------
    # Sensor registration
    # ------------------------------------------------------------------
    def add_sensor(self, sensor: BaseSensor, frequency_hz: float) -> None:
        dt_sample = 1.0 / frequency_hz
        # Use the sensor's own RNG seed for reproducibility per-sensor
        sensor._rng = self._rng
        self._sensors[sensor.name] = (sensor, dt_sample, 0.0)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self, max_time: Optional[float] = None) -> List[LogEntry]:
        """Execute the full simulation.  Returns the log."""
        self.history.clear()
        for sensor, _, _ in self._sensors.values():
            sensor.reset()

        t = 0.0
        duration = max_time or self.trajectory.duration
        # Initial state
        state = self.trajectory.state_at(t)

        # initialise sensor sample timers to 0 so they fire on t=0
        timers = {name: 0.0 for name in self._sensors}

        while t <= duration + 1e-9:
            state = self.trajectory.state_at(t)
            readings: Dict[str, SensorReading] = {}

            for name, (sensor, dt_sample, _) in self._sensors.items():
                if t >= timers[name] - 1e-12:
                    reading = sensor.read(state, dt_sample)
                    # A dropped sample (fault injection) is simply absent from
                    # the log — identical to a sensor that produced no output.
                    if reading is not None:
                        readings[name] = reading
                    timers[name] = t + dt_sample

            self.history.append(LogEntry(t=t, state=state, readings=readings))

            t += self.dt_physics

        return self.history

    # ------------------------------------------------------------------
    # Convenience: ground-truth arrays
    # ------------------------------------------------------------------
    def ground_truth_arrays(self) -> dict:
        states = [h.state for h in self.history]
        return {
            "t":       np.array([s.t for s in states]),
            "position": np.vstack([s.position for s in states]),   # N x 3
            "velocity": np.vstack([s.velocity for s in states]),
            "depth":    np.array([s.depth for s in states]),
            "yaw":      np.array([s.yaw_pitch_roll[0] for s in states]),
            "pitch":    np.array([s.yaw_pitch_roll[1] for s in states]),
            "roll":     np.array([s.yaw_pitch_roll[2] for s in states]),
        }

    # ------------------------------------------------------------------
    # Export: one CSV per sensor + ground-truth CSV + JSON bundle
    # ------------------------------------------------------------------
    def export_sensor_csv(self, name: str, path: str) -> None:
        """Write per-sample timestamps + data columns for a named sensor."""
        import csv
        arr = self.sensor_arrays(name)
        if arr is None:
            raise KeyError(f"No readings for sensor '{name}' in history")

        data = arr["data"]
        n_cols = 1 if data.ndim == 1 else data.shape[1]
        cols = ["t"] + [f"{name}_{axis}" for axis in range(n_cols)]

        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for t_row, d_row in zip(arr["t"], data):
                w.writerow([f"{float(t_row):.6f}"] + [f"{float(v):.8f}" for v in np.atleast_1d(d_row)])

    def export_ground_truth_csv(self, path: str) -> None:
        """Write ground-truth NED state per physics tick."""
        import csv
        gt = self.ground_truth_arrays()

        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t", "north", "east", "depth", "v_n", "v_e", "v_d",
                         "yaw_rad", "pitch_rad", "roll_rad"])
            for i in range(len(gt["t"])):
                w.writerow([
                    f"{gt['t'][i]:.6f}",
                    f"{gt['position'][i,0]:.8f}", f"{gt['position'][i,1]:.8f}", f"{gt['position'][i,2]:.8f}",
                    f"{gt['velocity'][i,0]:.8f}", f"{gt['velocity'][i,1]:.8f}", f"{gt['velocity'][i,2]:.8f}",
                    f"{gt['yaw'][i]:.10f}", f"{gt['pitch'][i]:.10f}", f"{gt['roll'][i]:.10f}",
                ])

    def export_json(self, path: str) -> None:
        """Single JSON bundle: ground truth + every sensor readings."""
        import json
        out = {"physics_dt": self.dt_physics, "trajectory": self.trajectory.duration}

        gt = self.ground_truth_arrays()
        out["ground_truth"] = {
            "t": gt["t"].tolist(),
            "position": gt["position"].tolist(),
            "velocity": gt["velocity"].tolist(),
            "yaw": gt["yaw"].tolist(),
            "pitch": gt["pitch"].tolist(),
            "roll": gt["roll"].tolist(),
        }

        sensors = {}
        for name in self._sensors:
            arr = self.sensor_arrays(name)
            if arr is None:
                continue
            sensors[name] = {
                "frequency_hz": 1.0 / self._sensors[name][1],
                "t": arr["t"].tolist(),
                "data": arr["data"].tolist(),
            }
        out["sensors"] = sensors

        with open(path, "w") as fh:
            json.dump(out, fh, indent=2)

    # Convenience: sensor arrays
    def sensor_arrays(self, name: str) -> Optional[dict]:
        """Collect all readings for a sensor, indexed by name."""
        entries = [h for h in self.history if name in h.readings]
        if not entries:
            return None
        r0 = entries[0].readings[name]
        return {
            "t":        np.array([h.readings[name].t for h in entries]),
            "data":     np.vstack([h.readings[name].data for h in entries]),
            "ideal":    np.vstack([h.readings[name].data_ideal for h in entries])
                        if r0.data_ideal is not None else None,
        }
