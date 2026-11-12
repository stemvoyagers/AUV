from __future__ import annotations

import numpy as np

from ..state import GRAVITY, VehicleState, ned_to_body_rotation
from .base import BaseSensor


class Accelerometer(BaseSensor):
    name = "accelerometer"
    axes = 3
    units = "m/s²"

    def _ideal(self, state: VehicleState) -> np.ndarray:
        R = ned_to_body_rotation(*state.yaw_pitch_roll)
        gravity_ned = np.array([0.0, 0.0, -GRAVITY])
        return R @ (state.acceleration_ned - gravity_ned)
