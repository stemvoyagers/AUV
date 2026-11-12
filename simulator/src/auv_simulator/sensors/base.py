"""Base sensor class.

Every sensor follows the same pattern:
    1. It has a nominal sampling rate (Hz).
    2. ``read(state, rng)`` returns a SensorReading derived from the current
       ground-truth VehicleState, optionally corrupted with noise / bias / drift
       and — optionally — with an injected *fault*.
    3. ``ideal(state)`` returns the noiseless "what the physics should give"
       value — used as a reference during validation.

Fault injection
---------------
Real missions fail.  Four fault classes are modelled here, all disabled by
default (``fault_start=0.0`` plus a no-op value):

``dropout_prob``
    Per-sample probability that the sensor produces **no reading at all**.
    ``read()`` returns ``None``; the simulator then simply omits that sample
    from the log, which is exactly what a DVL losing bottom-lock looks like.
    A navigation filter must therefore tolerate gaps, not zeros.

``stuck_at``
    Freeze the output at a constant value from ``fault_start`` onwards
    (classic "stuck-at" / latched ADC failure).  The reading looks plausible
    but carries no new information.

``scale_factor``
    Multiplicative gain error (e.g. 1.02 = +2 %).  This is the dominant
    error mode of an uncalibrated DVL scale factor.

``misalignment_deg``
    Sensor mounting misalignment, given as (roll, pitch, yaw) degrees.  The
    measured vector is rotated by this amount before noise is added, i.e. the
    sensor frame is not aligned with the body frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..state import VehicleState


@dataclass
class SensorReading:
    t: float                               # timestamp
    data: np.ndarray                       # raw sensor values
    data_ideal: np.ndarray | None = None   # ideal counterpart (optional, for validation)
    name: str = "sensor"

    @property
    def noise(self) -> Optional[np.ndarray]:
        if self.data_ideal is None:
            return None
        return self.data - self.data_ideal


def _rotation_from_euler_deg(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll), angles in degrees."""
    r, p, y = np.deg2rad([roll, pitch, yaw])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class BaseSensor:
    """Abstract base.  Subclasses must implement ``_ideal`` and ``_noisy``."""

    name: str = "sensor"
    axes: int = 3
    units: str = ""

    # Configurable non-idealities
    noise_std: np.ndarray
    bias: np.ndarray
    noise_density: float

    def __init__(self,
                 noise_std: float | np.ndarray = 0.0,
                 bias: float | np.ndarray = 0.0,
                 noise_density: float = 0.0,
                 seed: Optional[int] = None,
                 # --- fault injection (all inert by default) ---
                 dropout_prob: float = 0.0,
                 stuck_at: float | np.ndarray | None = None,
                 scale_factor: float | np.ndarray = 1.0,
                 misalignment_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 fault_start: float = 0.0):
        self.noise_std = np.atleast_1d(np.asarray(noise_std, dtype=float))
        self.bias      = np.atleast_1d(np.asarray(bias, dtype=float))
        self.noise_density = noise_density
        self._rng = np.random.default_rng(seed)

        # running drift state (integrates noise_density over time)
        self._drift = np.zeros(self.axes)

        # Fault configuration
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

    # ------------------------------------------------------------------
    # Physics — subclasses override
    # ------------------------------------------------------------------
    def _ideal(self, state: VehicleState) -> np.ndarray:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ideal(self, state: VehicleState) -> np.ndarray:
        """Noiseless value for the current state."""
        return self._ideal(state)

    def read(self, state: VehicleState, dt: float = 0.01) -> Optional[SensorReading]:
        """Return a possibly noisy reading, or ``None`` if the sample dropped.

        ``dt`` is used to update the drift (random walk) term.
        """
        ideal = np.asarray(self._ideal(state), dtype=float)

        # 0) Dropout — the sensor produced nothing this cycle.
        if self.dropout_prob > 0.0 and self._rng.random() < self.dropout_prob:
            return None

        data = ideal.copy()

        # 1) Mounting misalignment (vector sensors only)
        if self._misalignment is not None:
            data = self._misalignment @ data

        # 2) Gain error
        data = data * self.scale_factor

        # 3) Gaussian noise
        noise = self._rng.normal(0.0, self.noise_std, size=self.axes)

        # 4) Random walk (drift) — integrated white-noise process
        if self.noise_density > 0:
            self._drift += self._rng.normal(0.0, self.noise_density * np.sqrt(dt), size=self.axes)

        # 5) Combine
        data = data + noise + self.bias + self._drift

        # 6) Stuck-at fault (latched output, fault_start onwards)
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
