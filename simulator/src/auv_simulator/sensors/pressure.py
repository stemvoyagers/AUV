from __future__ import annotations

from ..state import VehicleState, depth_to_pressure
from .base import BaseSensor


class PressureSensor(BaseSensor):
    name = "pressure"
    axes = 1
    units = "Pa"

    def _ideal(self, state: VehicleState) -> list:
        return [depth_to_pressure(state.depth)]
