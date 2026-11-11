"""Vehicle state container.

All state is expressed in the NED world frame, except where body-frame
quantities are explicitly asked for.  Orientation is stored as a unit
quaternion (w, x, y, z) to avoid gimbal lock.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .frames import (
    GRAVITY,
    euler_to_quaternion,
    quaternion_to_euler,
    body_to_ned_rotation,
    ned_to_body_rotation,
)


@dataclass
class VehicleState:
    """Ground-truth state of the AUV at a single instant.

    Attributes
    ----------
    t : float
        Simulation time (seconds).
    position : np.ndarray
        NED position [north, east, depth] metres.
    velocity : np.ndarray
        NED velocity [v_n, v_e, v_d] m/s.
    acceleration_ned : np.ndarray
        NED acceleration [a_n, a_e, a_d] m/s² (excluding gravity).
    quaternion : np.ndarray
        (w, x, y, z) unit quaternion describing body→NED rotation.
    angular_velocity : np.ndarray
        Body-frame angular velocity [p, q, r] rad/s (roll, pitch, yaw rates).
    """
    t: float = 0.0
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def depth(self) -> float:
        return float(self.position[2])

    @property
    def yaw_pitch_roll(self) -> tuple[float, float, float]:
        """Euler angles (yaw, pitch, roll) in radians."""
        return quaternion_to_euler(self.quaternion)

    @property
    def heading_deg(self) -> float:
        """Compass heading, 0° = North, 90° = East."""
        yaw, *_ = self.yaw_pitch_roll
        return float(np.degrees((yaw + 2 * np.pi) % (2 * np.pi)))

    # ------------------------------------------------------------------
    # Body-frame projections (what the sensors actually see)
    # ------------------------------------------------------------------
    def velocity_body(self) -> np.ndarray:
        R = ned_to_body_rotation(*self.yaw_pitch_roll)
        return R @ self.velocity

    def linear_acceleration_body(self) -> np.ndarray:
        """Specific force in body frame = R_NB @ (a_ned - gravity).

        A real accelerometer does NOT measure gravity directly; it measures
        the non-gravitational specific force.
        """
        R = ned_to_body_rotation(*self.yaw_pitch_roll)
        gravity_ned = np.array([0.0, 0.0, -GRAVITY])   # gravity pulls DOWN → NED +z
        return R @ (self.acceleration_ned - gravity_ned)

    def rotation_matrix_ned_to_body(self) -> np.ndarray:
        return ned_to_body_rotation(*self.yaw_pitch_roll)

    # ------------------------------------------------------------------
    # Integration step (used by the simulator, not the trajectory)
    # ------------------------------------------------------------------
    def step(self, dt: float,
             acceleration_ned: Optional[np.ndarray] = None,
             angular_velocity_body: Optional[np.ndarray] = None) -> None:
        """Advance state by dt seconds using simple Euler integration.

        Provide *either* next acceleration in NED, or leave None (defaults to
        zero).  Orientation is advanced from the supplied body-frame angular
        velocity.
        """
        if acceleration_ned is not None:
            self.acceleration_ned = np.asarray(acceleration_ned, dtype=float)

        # Integrate velocity → position
        self.position = self.position + self.velocity * dt + 0.5 * self.acceleration_ned * dt ** 2
        self.velocity = self.velocity + self.acceleration_ned * dt

        # Integrate orientation from body-frame angular velocity
        if angular_velocity_body is not None:
            omega = np.asarray(angular_velocity_body, dtype=float)
            self.angular_velocity = omega
            # Quaternion derivative:  dq/dt = 0.5 * q ⊗ (0, omega)
            w, x, y, z = self.quaternion
            p, q_, r = omega
            dq = 0.5 * np.array([
                -x * p - y * q_ - z * r,
                 w * p + y * r - z * q_,
                 w * q_ - x * r + z * p,
                 w * r + x * q_ - y * p,
            ])
            self.quaternion = self.quaternion + dq * dt
            self.quaternion /= np.linalg.norm(self.quaternion)

        self.t += dt

    def copy(self) -> "VehicleState":
        return VehicleState(
            t=self.t,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            acceleration_ned=self.acceleration_ned.copy(),
            quaternion=self.quaternion.copy(),
            angular_velocity=self.angular_velocity.copy(),
        )
