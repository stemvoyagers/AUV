from __future__ import annotations

from ..state import VehicleState
from .base import BaseSensor


class Gyroscope(BaseSensor):
    name = "gyroscope"
    axes = 3
    units = "rad/s"

    def _ideal(self, state: VehicleState) -> list:
        return list(state.angular_velocity)
