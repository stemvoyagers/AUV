"""Simulation clock.

One fixed physics step (``dt_physics``) drives the whole simulation.  Every
sensor has its own nominal period, and the clock reports which sensors are due
on each tick.  This keeps a single authoritative timeline:

    * the ground truth is evaluated at ``t = k * dt_physics``;
    * sensors sample at their own rates, aligned to that same timeline;
    * timestamps are always multiples of ``dt_physics``.

The clock holds no vehicle or sensor state, so it can be tested on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Mapping, Optional

import numpy as np

# Tolerance used when comparing floating-point times.
_TIME_EPS = 1e-9
_SCHEDULE_EPS = 1e-12


@dataclass(frozen=True)
class ClockTick:
    """One physics tick: its index, timestamp and which sensors are due."""
    index: int
    t: float
    due_sensors: tuple[str, ...]


class SimulationClock:
    """Fixed-step clock with multi-rate sensor scheduling.

    Parameters
    ----------
    duration : float
        Simulated seconds to run.  The tick at ``t == duration`` is included.
    dt_physics : float
        Physics time step in seconds (default 5 ms = 200 Hz).
    sensor_periods : mapping, optional
        ``{sensor_name: sample_period_seconds}``.  A sensor fires on every
        tick whose time has reached its next scheduled sample.
    """

    def __init__(self,
                 duration: float,
                 dt_physics: float = 0.005,
                 sensor_periods: Optional[Mapping[str, float]] = None):
        duration = float(duration)
        dt_physics = float(dt_physics)

        if not np.isfinite(duration) or duration < 0.0:
            raise ValueError("duration must be finite and non-negative")
        if not np.isfinite(dt_physics) or dt_physics <= 0.0:
            raise ValueError("dt_physics must be finite and positive")

        periods: Dict[str, float] = {}
        for name, period in (sensor_periods or {}).items():
            period = float(period)
            if not np.isfinite(period) or period <= 0.0:
                raise ValueError(f"sensor '{name}' must have a positive period")
            periods[str(name)] = period

        self.duration = duration
        self.dt_physics = dt_physics
        self.sensor_periods = periods

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    @property
    def n_ticks(self) -> int:
        """Number of physics ticks, including both endpoints."""
        return int(np.floor(self.duration / self.dt_physics + _TIME_EPS)) + 1

    def timestamp(self, index: int) -> float:
        """Timestamp of tick ``index``.  Always ``index * dt_physics``."""
        if index < 0 or index >= self.n_ticks:
            raise IndexError(f"tick {index} outside 0..{self.n_ticks - 1}")
        return index * self.dt_physics

    def expected_samples(self, name: str) -> int:
        """Nominal number of samples a sensor should produce."""
        if name not in self.sensor_periods:
            raise KeyError(f"unknown sensor '{name}'")
        return int(np.floor(self.duration / self.sensor_periods[name] + _TIME_EPS)) + 1

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[ClockTick]:
        next_due: Dict[str, float] = {name: 0.0 for name in self.sensor_periods}

        for index in range(self.n_ticks):
            t = index * self.dt_physics

            due = tuple(
                name for name, scheduled in next_due.items()
                if t >= scheduled - _SCHEDULE_EPS
            )
            for name in due:
                next_due[name] = t + self.sensor_periods[name]

            yield ClockTick(index=index, t=t, due_sensors=due)
