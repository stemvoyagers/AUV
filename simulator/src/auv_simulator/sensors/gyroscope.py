"""Gyroscope model.

Physics
-------
An ideal gyroscope directly reads the body-frame angular velocity
[p, q, r] = [roll_rate, pitch_rate, yaw_rate].

    gyro_reading = angular_velocity_body + noise + bias + drift
"""

from __future__ import annotations

from ..state import VehicleState
from .base import BaseSensor


class Gyroscope(BaseSensor):
    name = "gyroscope"
    axes = 3
    units = "rad/s"

    def _ideal(self, state: VehicleState) -> list:
        return list(state.angular_velocity)
