"""Doppler Velocity Log (DVL) model.

Physics
-------
A DVL measures the vehicle's velocity **relative to the seabed**, expressed
in the body frame:

    v_body = R_NB @ v_ned

This is the measurement that makes horizontal position observable.  Without
it an AUV can only integrate acceleration, and a constant accelerometer bias
b_a produces a position error that grows as 0.5 · b_a · t²  — i.e. it drifts
without bound.  With a DVL the velocity itself is measured, so position error
grows only linearly (and in practice stays bounded because the velocity is
continuously corrected).

Notes
-----
* The DVL only produces a valid reading inside its *bottom-lock* range
  (typically < 200 m altitude).  A loss of lock is modelled by the generic
  fault mechanism in :class:`~auv_simulator.sensors.base.BaseSensor`
  (``dropout_prob``) rather than here.
* The z component is only reliable for a 4-beam Janus configuration; the
  model still provides it, as most AUV DVLs do.
"""

from __future__ import annotations

import numpy as np

from ..core.state import VehicleState
from .base import BaseSensor


class DVL(BaseSensor):
    name = "dvl"
    axes = 3
    units = "m/s"

    def _ideal(self, state: VehicleState) -> np.ndarray:
        # Body-frame velocity, including the vertical component.
        return state.velocity_body()
