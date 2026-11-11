"""Coordinate-frame utilities.

Conventions
-----------
World frame : North-East-Down (NED).  x = North, y = East, z = Down (positive into water).
Body frame  : x = forward (surge), y = starboard (sway), z = down (heave).
Euler angles follow the aerospace z-y-x (yaw-pitch-roll) convention:
    yaw   : rotation about world-Z (NED), positive clockwise when viewed from above.
    pitch : rotation about body-Y, positive nose-up.
    roll  : rotation about body-X, positive right-wing-down.

Quaternions use the (w, x, y, z) Hamilton convention.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRAVITY = 9.80665       # m/s² in NED (z-down), so gravity vector is (0, 0, +g)
EARTH_RADIUS = 6_371_000  # metres, used for pressure -> depth


# ---------------------------------------------------------------------------
# Euler <-> rotation matrix  (yaw-pitch-roll, NED)
# ---------------------------------------------------------------------------
def ned_to_body_rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Return R_NB such that  v_body = R_NB @ v_ned.

    R = R_x(-roll) @ R_y(-pitch) @ R_z(-yaw)
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    return np.array([
        [ cp * cy,  cp * sy, -sp     ],
        [ sr * sp * cy - cr * sy,  sr * sp * sy + cr * cy,  sr * cp ],
        [ cr * sp * cy + sr * sy,  cr * sp * sy - sr * cy,  cr * cp ],
    ])


def body_to_ned_rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Return R_BN such that  v_ned = R_BN @ v_body.  (Transpose of R_NB.)"""
    return ned_to_body_rotation(yaw, pitch, roll).T


# ---------------------------------------------------------------------------
# Euler <-> quaternion
# ---------------------------------------------------------------------------
def euler_to_quaternion(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Return (w, x, y, z) from aerospace z-y-x Euler angles."""
    cr2 = np.cos(roll / 2);   sr2 = np.sin(roll / 2)
    cp2 = np.cos(pitch / 2);  sp2 = np.sin(pitch / 2)
    cy2 = np.cos(yaw / 2);    sy2 = np.sin(yaw / 2)

    w = cr2 * cp2 * cy2 + sr2 * sp2 * sy2
    x = sr2 * cp2 * cy2 - cr2 * sp2 * sy2
    y = cr2 * sp2 * cy2 + sr2 * cp2 * sy2
    z = cr2 * cp2 * sy2 - sr2 * sp2 * cy2
    return np.array([w, x, y, z], dtype=float)


def quaternion_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    """Return (yaw, pitch, roll) from (w, x, y, z)."""
    w, x, y, z = q

    # Roll (X-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (Y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * (np.pi / 2)     # clamp to ±90° (avoids gimbal lock)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (Z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return float(yaw), float(pitch), float(roll)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 ⊗ q2.  Both (w, x, y, z)."""
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
    """Rotate vector v by unit quaternion q (v' = q ⊗ v ⊗ q⁻¹).

    v is treated as a pure quaternion (0, vx, vy, vz).  Returns (vx', vy', vz').
    """
    qn = quaternion_normalize(q)
    p = np.array([0.0, v[0], v[1], v[2]])
    q_inv = quaternion_conjugate(qn)
    result = quaternion_multiply(quaternion_multiply(qn, p), q_inv)
    return result[1:]


# ---------------------------------------------------------------------------
# Pressure <-> depth (freshwater approximation)
# ---------------------------------------------------------------------------
def depth_to_pressure(depth_m: float, rho: float = 1025.0) -> float:
    """Hydrostatic pressure in Pascals for a given depth in NED (positive down).

    P = P_atm + ρ g h
    """
    P_atm = 101_325.0          # Pa at sea level
    return P_atm + rho * GRAVITY * depth_m


def pressure_to_depth(pressure_pa, rho: float = 1025.0):
    """Invert depth_to_pressure.  Scalar or array input."""
    P_atm = 101_325.0
    return np.maximum(0.0, (np.asarray(pressure_pa) - P_atm) / (rho * GRAVITY))
