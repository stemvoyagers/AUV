"""Tests for core frame math: rotations, quaternions, pressure<->depth."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from auv_simulator.core.frames import (
    GRAVITY,
    ned_to_body_rotation,
    body_to_ned_rotation,
    euler_to_quaternion,
    quaternion_to_euler,
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_normalize,
    rotate_vector,
    depth_to_pressure,
    pressure_to_depth,
)


class TestRotations:
    """R_NB @ v_ned = v_body and R_BN @ v_body = v_ned for identity rotation."""

    def test_zero_angles_identity(self):
        R = ned_to_body_rotation(0.0, 0.0, 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_transpose_inverse(self):
        for yaw in [0.0, np.pi / 4, np.pi / 2]:
            for pitch in [-np.pi / 6, 0.0, np.pi / 6]:
                for roll in [-np.pi / 3, 0.0, np.pi / 3]:
                    R_NB = ned_to_body_rotation(yaw, pitch, roll)
                    R_BN = body_to_ned_rotation(yaw, pitch, roll)
                    np.testing.assert_allclose(R_BN, R_NB.T, atol=1e-12)
                    np.testing.assert_allclose(R_NB @ R_BN, np.eye(3), atol=1e-12)

    def test_yaw_only_rotates_north_into_body_starboard(self):
        """Yaw +π/2 = heading East. Starboard from East = South = world (0,-1,0)."""
        R = ned_to_body_rotation(np.pi / 2, 0.0, 0.0)
        v_north = np.array([1.0, 0.0, 0.0])
        v_body = R @ v_north
        # North→body_x should be 0, North→body_y should be -1 (southward starboard)
        np.testing.assert_allclose(v_body, np.array([0.0, -1.0, 0.0]), atol=1e-9)


class TestQuaternions:
    """Round-trip euler <-> quaternion."""

    @pytest.mark.parametrize("yaw,pitch,roll", [
        (0.0, 0.0, 0.0),
        (np.pi, 0.0, 0.0),
        (0.0, np.pi / 4, np.pi / 6),
        (np.pi / 3, -np.pi / 5, np.pi / 7),
    ])
    def test_round_trip(self, yaw, pitch, roll):
        q = euler_to_quaternion(yaw, pitch, roll)
        y2, p2, r2 = quaternion_to_euler(q)
        np.testing.assert_allclose([y2, p2, r2], [yaw, pitch, roll], atol=1e-9)

    def test_multiply_by_conjugate_gives_identity(self):
        q = euler_to_quaternion(0.3, -0.1, 0.2)
        q_inv = quaternion_conjugate(q)
        prod = quaternion_multiply(q, q_inv)
        prod = prod / np.linalg.norm(prod)
        np.testing.assert_allclose(prod, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_rotate_vector_round_trip(self):
        q = euler_to_quaternion(0.7, -0.3, 0.5)
        v = np.array([3.0, -2.0, 1.5])
        v_rot = rotate_vector(v, q)
        # inverse rotation should give v back
        q_inv = quaternion_conjugate(q)
        v_back = rotate_vector(v_rot, q_inv)
        np.testing.assert_allclose(v_back, v, atol=1e-10)


class TestPressureDepth:
    def test_zero_depth_is_atmospheric(self):
        P = depth_to_pressure(0.0)
        assert abs(P - 101_325.0) < 1.0

    def test_round_trip(self):
        for d in [0.0, 1.0, 10.0, 100.0]:
            P = depth_to_pressure(d)
            d_back = pressure_to_depth(P)
            np.testing.assert_allclose(d_back, d, atol=1e-6)

    def test_pressure_increases_with_depth(self):
        assert depth_to_pressure(10.0) > depth_to_pressure(0.0)


class TestGravityConvention:
    """Sanity: gravity in NED points +z (down), so accelerometer at rest reads (0,0,-g) body-frame when level."""

    def test_level_stationary_accel_reading(self):
        R = ned_to_body_rotation(0.0, 0.0, 0.0)
        gravity_ned = np.array([0.0, 0.0, -GRAVITY])  # note sign
        a_ned = np.zeros(3)                            # stationary, no accel
        f_body = R @ (a_ned - gravity_ned)
        np.testing.assert_allclose(f_body, np.array([0.0, 0.0, GRAVITY]), atol=1e-9)
