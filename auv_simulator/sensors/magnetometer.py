"""Magnetometer model.

Physics
-------
The Earth's magnetic field vector is measured in the body frame:

    m_body = R_NB @ m_earth_ned

where m_earth_ned is a simple local model (horizontal component pointing
North, a small downward dip).  A reading would be:

    mag_reading = m_body + noise + bias
"""

from __future__ import annotations

import numpy as np

from ..core.frames import ned_to_body_rotation
from ..core.state import VehicleState
from .base import BaseSensor


class Magnetometer(BaseSensor):
    name = "magnetometer"
    axes = 3
    units = "µT"

    # Approximate mid-latitude Earth magnetic field in NED (µT).
    # Field points Northward (+x) and Downward (+z, into Earth surface).
    EARTH_FIELD_NED = np.array([24.0, 0.0, 22.0])

    def _ideal(self, state: VehicleState) -> np.ndarray:
        R = ned_to_body_rotation(*state.yaw_pitch_roll)
        return R @ self.EARTH_FIELD_NED.astype(float)
