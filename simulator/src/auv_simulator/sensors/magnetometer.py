from __future__ import annotations

import numpy as np

from ..state import VehicleState, ned_to_body_rotation
from .base import BaseSensor


class Magnetometer(BaseSensor):
    name = "magnetometer"
    axes = 3
    units = "µT"

    EARTH_FIELD_NED = np.array([24.0, 0.0, 22.0])

    def _ideal(self, state: VehicleState) -> np.ndarray:
        R = ned_to_body_rotation(*state.yaw_pitch_roll)
        return R @ self.EARTH_FIELD_NED.astype(float)
