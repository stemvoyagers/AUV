"""Tests for the ground-truth state container and the frame maths it relies on."""

import numpy as np
import pytest

from auv_simulator.state import (
    GRAVITY,
    VehicleState,
    body_to_ned_rotation,
    depth_to_pressure,
    euler_to_quaternion,
    ned_to_body_rotation,
    pressure_to_depth,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_euler,
    rotate_vector,
)


class TestStateContainer:
    def test_default_state_is_identity_at_origin(self):
        s = VehicleState()
        np.testing.assert_allclose(s.position, [0, 0, 0])
        np.testing.assert_allclose(s.velocity, [0, 0, 0])
        np.testing.assert_allclose(s.quaternion, [1, 0, 0, 0])
        assert s.t == 0.0
        assert s.depth == 0.0

    def test_position_components_are_north_east_depth(self):
        s = VehicleState(position=np.array([12.0, -4.0, 20.0]))
        assert s.north == 12.0
        assert s.east == -4.0
        assert s.depth == 20.0

    def test_heading_of_level_north_state(self):
        s = VehicleState(quaternion=euler_to_quaternion(0.0, 0.0, 0.0))
        assert s.heading_deg == pytest.approx(0.0, abs=1e-9)

    def test_heading_of_eastward_state(self):
        s = VehicleState(quaternion=euler_to_quaternion(np.pi / 2, 0.0, 0.0))
        assert s.heading_deg == pytest.approx(90.0, abs=1e-9)

    def test_speed_is_velocity_magnitude(self):
        s = VehicleState(velocity=np.array([3.0, 4.0, 0.0]))
        assert s.speed == pytest.approx(5.0)

    def test_copy_is_independent(self):
        s = VehicleState(position=np.array([1.0, 2.0, 3.0]))
        c = s.copy()
        c.position[0] = 99.0
        assert s.position[0] == 1.0

    def test_level_resting_vehicle_reads_gravity_on_body_z(self):
        """Specific force at rest and level is +g on the down axis."""
        s = VehicleState()
        np.testing.assert_allclose(s.linear_acceleration_body(), [0.0, 0.0, GRAVITY], atol=1e-9)

    def test_velocity_body_of_northward_motion(self):
        s = VehicleState(velocity=np.array([2.0, 0.0, 0.0]))
        np.testing.assert_allclose(s.velocity_body(), [2.0, 0.0, 0.0], atol=1e-9)

    def test_step_integrates_position_and_velocity(self):
        s = VehicleState()
        s.step(0.5, acceleration_ned=np.array([1.0, 0.0, 0.0]))
        assert s.t == pytest.approx(0.5)
        assert s.velocity[0] == pytest.approx(0.5)
        assert s.position[0] == pytest.approx(0.125)


class TestRotations:
    def test_zero_angles_give_identity(self):
        np.testing.assert_allclose(ned_to_body_rotation(0.0, 0.0, 0.0), np.eye(3), atol=1e-12)

    def test_transpose_is_inverse(self):
        for yaw in (0.0, np.pi / 4, np.pi / 2):
            for pitch in (-np.pi / 6, 0.0, np.pi / 6):
                for roll in (-np.pi / 3, 0.0, np.pi / 3):
                    r_nb = ned_to_body_rotation(yaw, pitch, roll)
                    r_bn = body_to_ned_rotation(yaw, pitch, roll)
                    np.testing.assert_allclose(r_bn, r_nb.T, atol=1e-12)
                    np.testing.assert_allclose(r_nb @ r_bn, np.eye(3), atol=1e-12)

    def test_yaw_90_maps_north_to_starboard_negative(self):
        """Heading east: world north projects onto body -y."""
        r = ned_to_body_rotation(np.pi / 2, 0.0, 0.0)
        np.testing.assert_allclose(r @ np.array([1.0, 0.0, 0.0]), [0.0, -1.0, 0.0], atol=1e-9)


class TestQuaternions:
    @pytest.mark.parametrize("yaw,pitch,roll", [
        (0.0, 0.0, 0.0),
        (np.pi, 0.0, 0.0),
        (0.0, np.pi / 4, np.pi / 6),
        (np.pi / 3, -np.pi / 5, np.pi / 7),
    ])
    def test_euler_round_trip(self, yaw, pitch, roll):
        q = euler_to_quaternion(yaw, pitch, roll)
        np.testing.assert_allclose(quaternion_to_euler(q), [yaw, pitch, roll], atol=1e-9)

    def test_multiply_by_conjugate_gives_identity(self):
        q = euler_to_quaternion(0.3, -0.1, 0.2)
        prod = quaternion_multiply(q, quaternion_conjugate(q))
        prod = prod / np.linalg.norm(prod)
        np.testing.assert_allclose(prod, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_rotate_vector_round_trip(self):
        q = euler_to_quaternion(0.7, -0.3, 0.5)
        v = np.array([3.0, -2.0, 1.5])
        v_rot = rotate_vector(v, q)
        v_back = rotate_vector(v_rot, quaternion_conjugate(q))
        np.testing.assert_allclose(v_back, v, atol=1e-10)


class TestPressureDepth:
    def test_zero_depth_is_atmospheric(self):
        assert abs(depth_to_pressure(0.0) - 101_325.0) < 1.0

    def test_round_trip(self):
        for d in (0.0, 1.0, 10.0, 20.0, 24.0, 100.0):
            np.testing.assert_allclose(pressure_to_depth(depth_to_pressure(d)), d, atol=1e-6)

    def test_pressure_increases_with_depth(self):
        assert depth_to_pressure(10.0) > depth_to_pressure(0.0)

    def test_twenty_metres_is_about_two_bar_gauge(self):
        gauge = depth_to_pressure(20.0) - 101_325.0
        assert gauge == pytest.approx(1025.0 * GRAVITY * 20.0, rel=1e-12)
