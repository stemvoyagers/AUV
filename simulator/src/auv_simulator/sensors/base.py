from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..state import VehicleState


@dataclass
class SensorReading:
    t: float
    data: np.ndarray
    data_ideal: np.ndarray | None = None
    name: str = "sensor"

    @property
    def noise(self) -> Optional[np.ndarray]:
        if self.data_ideal is None:
            return None
        return self.data - self.data_ideal


def _rotation_from_euler_deg(roll: float, pitch: float, yaw: float) -> np.ndarray:
    r, p, y = np.deg2rad([roll, pitch, yaw])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class BaseSensor:
    name: str = "sensor"
    axes: int = 3
    units: str = ""

    noise_std: np.ndarray
    bias: np.ndarray
    noise_density: float

    def __init__(self,
                 noise_std: float | np.ndarray = 0.0,
                 bias: float | np.ndarray = 0.0,
                 noise_density: float = 0.0,
                 seed: Optional[int] = None,
                 dropout_prob: float = 0.0,
                 stuck_at: float | np.ndarray | None = None,
                 scale_factor: float | np.ndarray = 1.0,
                 misalignment_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 fault_start: float = 0.0):
        self.noise_std = np.atleast_1d(np.asarray(noise_std, dtype=float))
        self.bias      = np.atleast_1d(np.asarray(bias, dtype=float))
        self.noise_density = noise_density
        self._rng = np.random.default_rng(seed)

        self._drift = np.zeros(self.axes)

        self.dropout_prob = float(dropout_prob)
        self.scale_factor = np.atleast_1d(np.asarray(scale_factor, dtype=float))
        self.fault_start = float(fault_start)
        self.stuck_at = None
        if stuck_at is not None:
            self.stuck_at = np.broadcast_to(
                np.atleast_1d(np.asarray(stuck_at, dtype=float)),
                (self.axes,),
            ).astype(float).copy()

        if any(v != 0.0 for v in misalignment_deg):
            R = _rotation_from_euler_deg(*misalignment_deg)
            self._misalignment = R if self.axes == 3 else None
        else:
            self._misalignment = None

    def _ideal(self, state: VehicleState) -> np.ndarray:
        raise NotImplementedError

    def ideal(self, state: VehicleState) -> np.ndarray:
        return self._ideal(state)

    def read(self, state: VehicleState, dt: float = 0.01) -> Optional[SensorReading]:
        ideal = np.asarray(self._ideal(state), dtype=float)

        if self.dropout_prob > 0.0 and self._rng.random() < self.dropout_prob:
            return None

        data = ideal.copy()

        if self._misalignment is not None:
            data = self._misalignment @ data

        data = data * self.scale_factor

        noise = self._rng.normal(0.0, self.noise_std, size=self.axes)

        if self.noise_density > 0:
            self._drift += self._rng.normal(0.0, self.noise_density * np.sqrt(dt), size=self.axes)

        data = data + noise + self.bias + self._drift

        if self.stuck_at is not None and state.t >= self.fault_start:
            data = self.stuck_at.copy()

        return SensorReading(
            t=state.t,
            data=data,
            data_ideal=ideal,
            name=self.name,
        )

    def reset(self) -> None:
        self._drift = np.zeros(self.axes)
