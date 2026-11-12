"""Ground-truth state and the coordinate-frame maths behind it.

Conventions
-----------
World frame : North-East-Down (NED).  x = North, y = East, z = Down.
              Depth is therefore the +z coordinate and is positive downwards,
              so *descending* means the depth value increases.
Body frame  : x = forward (surge), y = starboard (sway), z = down (heave).
Euler angles follow the aerospace z-y-x (yaw-pitch-roll) convention:
    yaw   : rotation about world-Z, positive clockwise viewed from above
            (0 = North, +90 deg = East).
    pitch : rotation about body-Y, positive nose-up.
    roll  : rotation about body-X, positive right-wing-down.
Quaternions use the (w, x, y, z) Hamilton convention.

``VehicleState`` is the single source of truth for what the AUV is actually
doing at one instant.  Trajectory, sensors and exporters all read from it, so
the simulator only ever has one representation of the truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRAVITY = 9.80665           # m/s^2
EARTH_RADIUS = 6_371_000    # metres
SEA_WATER_DENSITY = 1025.0  # kg/m^3
ATMOSPHERIC_PRESSURE = 101_325.0  # Pa


# ---------------------------------------------------------------------------
# Euler <-> rotation matrix  (yaw-pitch-roll, NED)
# ---------------------------------------------------------------------------
def ned_to_body_rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Return R_NB such that  v_body = R_NB @ v_ned.

    R = Rx(-roll) @ Ry(-pitch) @ Rz(-yaw)
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    return np.array([
        [cp * cy, cp * sy, -sp],
        [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, sr * cp],
        [cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp],
    ])


def body_to_ned_rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Return R_BN such that  v_ned = R_BN @ v_body.  (Transpose of R_NB.)"""
    return ned_to_body_rotation(yaw, pitch, roll).T


# ---------------------------------------------------------------------------
# Euler <-> quaternion
# ---------------------------------------------------------------------------
def euler_to_quaternion(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Return (w, x, y, z) from aerospace z-y-x Euler angles."""
    cr2, sr2 = np.cos(roll / 2), np.sin(roll / 2)
    cp2, sp2 = np.cos(pitch / 2), np.sin(pitch / 2)
    cy2, sy2 = np.cos(yaw / 2), np.sin(yaw / 2)

    w = cr2 * cp2 * cy2 + sr2 * sp2 * sy2
    x = sr2 * cp2 * cy2 - cr2 * sp2 * sy2
    y = cr2 * sp2 * cy2 + sr2 * cp2 * sy2
    z = cr2 * cp2 * sy2 - sr2 * sp2 * cy2
    return np.array([w, x, y, z], dtype=float)


def quaternion_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    """Return (yaw, pitch, roll) from (w, x, y, z)."""
    w, x, y, z = q

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * (np.pi / 2)     # clamp to +/-90 deg, avoids gimbal lock
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return float(yaw), float(pitch), float(roll)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 (x) q2.  Both (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate (w, -x, -y, -z)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def rotate_vector(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate vector v by unit quaternion q (v' = q (x) v (x) q^-1)."""
    qn = quaternion_normalize(q)
    p = np.array([0.0, v[0], v[1], v[2]])
    q_inv = quaternion_conjugate(qn)
    result = quaternion_multiply(quaternion_multiply(qn, p), q_inv)
    return result[1:]


# ---------------------------------------------------------------------------
# Pressure <-> depth (sea-water approximation)
# ---------------------------------------------------------------------------
def depth_to_pressure(depth_m, rho: float = SEA_WATER_DENSITY):
    """Hydrostatic pressure in Pascals for a depth in metres (positive down).

    P = P_atm + rho * g * h
    """
    return ATMOSPHERIC_PRESSURE + rho * GRAVITY * np.asarray(depth_m, dtype=float)


def pressure_to_depth(pressure_pa, rho: float = SEA_WATER_DENSITY):
    """Invert :func:`depth_to_pressure`.  Scalar or array input."""
    return np.maximum(0.0, (np.asarray(pressure_pa, dtype=float) - ATMOSPHERIC_PRESSURE)
                      / (rho * GRAVITY))


# ---------------------------------------------------------------------------
# Ground-truth state
# ---------------------------------------------------------------------------
@dataclass
class VehicleState:
    """Ground-truth state of the AUV at a single instant.

    Attributes
    ----------
    t : float
        Simulation time in seconds.
    position : np.ndarray
        NED position [north, east, depth] in metres.
    velocity : np.ndarray
        NED velocity [v_n, v_e, v_d] in m/s (v_d positive downwards).
    acceleration_ned : np.ndarray
        NED acceleration [a_n, a_e, a_d] in m/s^2 (gravity excluded).
    quaternion : np.ndarray
        (w, x, y, z) unit quaternion describing the body-to-NED rotation.
    angular_velocity : np.ndarray
        Body-frame angular velocity [p, q, r] in rad/s.
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
    def north(self) -> float:
        return float(self.position[0])

    @property
    def east(self) -> float:
        return float(self.position[1])

    @property
    def depth(self) -> float:
        return float(self.position[2])

    @property
    def yaw_pitch_roll(self) -> tuple[float, float, float]:
        """Euler angles (yaw, pitch, roll) in radians."""
        return quaternion_to_euler(self.quaternion)

    @property
    def heading_deg(self) -> float:
        """Compass heading in degrees: 0 = North, 90 = East."""
        yaw, *_ = self.yaw_pitch_roll
        return float(np.degrees((yaw + 2 * np.pi) % (2 * np.pi)))

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    # ------------------------------------------------------------------
    # Body-frame projections (what the sensors actually see)
    # ------------------------------------------------------------------
    def velocity_body(self) -> np.ndarray:
        return ned_to_body_rotation(*self.yaw_pitch_roll) @ self.velocity

    def linear_acceleration_body(self) -> np.ndarray:
        """Specific force in the body frame = R_NB @ (a_ned - gravity_ned).

        An accelerometer measures non-gravitational specific force, not
        gravity itself; at rest and level it therefore reads +g on body-z.
        """
        R = ned_to_body_rotation(*self.yaw_pitch_roll)
        gravity_ned = np.array([0.0, 0.0, -GRAVITY])
        return R @ (self.acceleration_ned - gravity_ned)

    def rotation_matrix_ned_to_body(self) -> np.ndarray:
        return ned_to_body_rotation(*self.yaw_pitch_roll)

    # ------------------------------------------------------------------
    # Integration step (used for state propagation, not by the trajectory)
    # ------------------------------------------------------------------
    def step(self, dt: float,
             acceleration_ned: Optional[np.ndarray] = None,
             angular_velocity_body: Optional[np.ndarray] = None) -> None:
        """Advance the state by dt seconds with simple Euler integration."""
        if acceleration_ned is not None:
            self.acceleration_ned = np.asarray(acceleration_ned, dtype=float)

        self.position = self.position + self.velocity * dt + 0.5 * self.acceleration_ned * dt ** 2
        self.velocity = self.velocity + self.acceleration_ned * dt

        if angular_velocity_body is not None:
            omega = np.asarray(angular_velocity_body, dtype=float)
            self.angular_velocity = omega
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
