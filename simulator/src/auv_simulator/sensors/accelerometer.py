"""Accelerometer model.

Physics
-------
A practical accelerometer measures **specific force** = total linear
acceleration minus gravity, expressed in the body frame.

    f_body = R_NB @ (a_ned - g_ned)

where  g_ned = (0, 0, -g)  because in NED +z points *down* and gravity
pulls objects *down*, so gravity's contribution to the NED frame is
negative z.  The accelerometer output is therefore:

    accel_reading = f_body + noise + bias + drift
"""

from __future__ import annotations

import numpy as np

from ..state import GRAVITY, VehicleState, ned_to_body_rotation
from .base import BaseSensor


class Accelerometer(BaseSensor):
    name = "accelerometer"
    axes = 3
    units = "m/s²"

    def _ideal(self, state: VehicleState) -> np.ndarray:
        # Specific force = R_NB @ (a_ned - gravity_ned)
        R = ned_to_body_rotation(*state.yaw_pitch_roll)
        gravity_ned = np.array([0.0, 0.0, -GRAVITY])
        return R @ (state.acceleration_ned - gravity_ned)
