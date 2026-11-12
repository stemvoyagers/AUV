"""Pressure sensor model.

Physics
-------
Hydrostatic pressure (Pascal law):

    P(z) = P_atm + ρ g z

where z is positive down in NED.  We model the sensor as returning
pressure in Pa (internally) and the navigation system converts to
depth using the same formula.
"""

from __future__ import annotations

from ..state import VehicleState, depth_to_pressure
from .base import BaseSensor


class PressureSensor(BaseSensor):
    name = "pressure"
    axes = 1
    units = "Pa"

    def _ideal(self, state: VehicleState) -> list:
        return [depth_to_pressure(state.depth)]
