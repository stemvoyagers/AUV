from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

GRAVITY = 9.80665
EARTH_RADIUS = 6_371_000
SEA_WATER_DENSITY = 1025.0
ATMOSPHERIC_PRESSURE = 101_325.0


def ned_to_body_rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    return np.array([
        [cp * cy, cp * sy, -sp],
        [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, sr * cp],
        [cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp],
    ])


def body_to_ned_rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    return ned_to_body_rotation(yaw, pitch, roll).T


def euler_to_quaternion(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cr2, sr2 = np.cos(roll / 2), np.sin(roll / 2)
    cp2, sp2 = np.cos(pitch / 2), np.sin(pitch / 2)
    cy2, sy2 = np.cos(yaw / 2), np.sin(yaw / 2)

    w = cr2 * cp2 * cy2 + sr2 * sp2 * sy2
    x = sr2 * cp2 * cy2 - cr2 * sp2 * sy2
    y = cr2 * sp2 * cy2 + sr2 * cp2 * sy2
    z = cr2 * cp2 * sy2 - sr2 * sp2 * cy2
    return np.array([w, x, y, z], dtype=float)


def quaternion_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = q

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * (np.pi / 2)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return float(yaw), float(pitch), float(roll)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def rotate_vector(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    qn = quaternion_normalize(q)
    p = np.array([0.0, v[0], v[1], v[2]])
    q_inv = quaternion_conjugate(qn)
    result = quaternion_multiply(quaternion_multiply(qn, p), q_inv)
    return result[1:]


def depth_to_pressure(depth_m, rho: float = SEA_WATER_DENSITY):
    return ATMOSPHERIC_PRESSURE + rho * GRAVITY * np.asarray(depth_m, dtype=float)


def pressure_to_depth(pressure_pa, rho: float = SEA_WATER_DENSITY):
    return np.maximum(0.0, (np.asarray(pressure_pa, dtype=float) - ATMOSPHERIC_PRESSURE)
                      / (rho * GRAVITY))


@dataclass
class VehicleState:
    t: float = 0.0
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

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
        return quaternion_to_euler(self.quaternion)

    @property
    def heading_deg(self) -> float:
        yaw, *_ = self.yaw_pitch_roll
        return float(np.degrees((yaw + 2 * np.pi) % (2 * np.pi)))

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    def velocity_body(self) -> np.ndarray:
        return ned_to_body_rotation(*self.yaw_pitch_roll) @ self.velocity

    def linear_acceleration_body(self) -> np.ndarray:
        R = ned_to_body_rotation(*self.yaw_pitch_roll)
        gravity_ned = np.array([0.0, 0.0, -GRAVITY])
        return R @ (self.acceleration_ned - gravity_ned)

    def rotation_matrix_ned_to_body(self) -> np.ndarray:
        return ned_to_body_rotation(*self.yaw_pitch_roll)

    def step(self, dt: float,
             acceleration_ned: Optional[np.ndarray] = None,
             angular_velocity_body: Optional[np.ndarray] = None) -> None:
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
