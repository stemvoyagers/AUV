"""Tests for sensor models — ideal physics should match expectations."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from auv_simulator.core.frames import GRAVITY, euler_to_quaternion
from auv_simulator.core.state import VehicleState
from auv_simulator.sensors import (
    Accelerometer, DVL, Gyroscope, Magnetometer, PressureSensor,
)
from auv_simulator.simulator import Simulator


@pytest.fixture
def level_state() -> VehicleState:
    return VehicleState(t=0.0, position=np.zeros(3), velocity=np.zeros(3),
                        acceleration_ned=np.zeros(3),
                        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
                        angular_velocity=np.zeros(3))


class TestAccelerometerIdeal:
    def test_level_static_reads_gravity(self, level_state):
        ideal = Accelerometer().ideal(level_state)
        # body-z should read +g (we're level, gravity is (0,0,-g) NED, R_NB=I,
        # specific force = I @ (0,0,0) - (0,0,-g) = (0,0,g))
        np.testing.assert_allclose(ideal, [0.0, 0.0, GRAVITY], atol=1e-9)

    def test_level_constant_accel_north(self, level_state):
        level_state.acceleration_ned = np.array([2.0, 0.0, 0.0])
        ideal = Accelerometer().ideal(level_state)
        np.testing.assert_allclose(ideal, [2.0, 0.0, GRAVITY], atol=1e-9)

    def test_noise_zero_gives_ideal(self, level_state):
        s = Accelerometer(noise_std=0.0, bias=0.0, noise_density=0.0, seed=0)
        r1 = s.read(level_state)
        r2 = s.read(level_state)
        np.testing.assert_allclose(r1.data, ideal_forward_only(level_state), atol=1e-15)
        np.testing.assert_allclose(r2.data, r1.data, atol=1e-15)


def ideal_forward_only(state):
    """Helper — exactly what accel ideal() returns."""
    from auv_simulator.core.frames import ned_to_body_rotation
    R = ned_to_body_rotation(*state.yaw_pitch_roll)
    gravity_ned = np.array([0.0, 0.0, -GRAVITY])
    return R @ (state.acceleration_ned - gravity_ned)


class TestGyroscopeIdeal:
    def test_zero_rate_zero_reading(self, level_state):
        ideal = Gyroscope().ideal(level_state)
        np.testing.assert_allclose(ideal, [0, 0, 0], atol=1e-12)

    def test_records_yaw_rate(self, level_state):
        level_state.angular_velocity = np.array([0.0, 0.0, 0.1571])
        ideal = Gyroscope().ideal(level_state)
        np.testing.assert_allclose(ideal, [0.0, 0.0, 0.1571], atol=1e-9)


class TestPressureIdeal:
    def test_zero_depth_is_atmospheric(self, level_state):
        P = PressureSensor().ideal(level_state)[0]
        assert abs(P - 101_325.0) < 1.0

    def test_positive_depth_reads_higher(self):
        s = VehicleState(position=np.array([0.0, 0.0, 10.0]))
        P = PressureSensor().ideal(s)[0]
        assert P > 101_325.0


class TestSensorNonIdealities:
    def test_noise_adds_variation(self, level_state):
        s = Accelerometer(noise_std=1.0, seed=42)
        samples = np.array([s.read(level_state).data for _ in range(1000)])
        # std of samples > 0
        assert np.std(samples, axis=0).min() > 0.5

    def test_bias_is_persistent(self, level_state):
        s = Accelerometer(noise_std=0.0, bias=np.array([0.1, 0.2, -0.3]), seed=0)
        readings = [s.read(level_state).data for _ in range(10)]
        ideal = Accelerometer().ideal(level_state)
        for r in readings:
            np.testing.assert_allclose(r - ideal, [0.1, 0.2, -0.3], atol=1e-9)

    def test_drift_walks_over_time(self, level_state):
        s = Gyroscope(noise_std=0.0, bias=0.0, noise_density=0.1, seed=42)
        # reset drift then take several readings
        s.reset()
        readings = []
        for _ in range(200):
            readings.append(s.read(level_state, dt=0.01).data.copy())
        # drift should accumulate → readings diverge from ideal
        diffs = np.linalg.norm(np.array(readings), axis=1)
        assert diffs[-1] > diffs[0]


class TestFaultInjection:
    """Faults are inert by default and fully opt-in per sensor."""

    def test_default_sensor_has_no_faults(self, level_state):
        s = Accelerometer(seed=0)
        assert s.read(level_state) is not None
        assert s.stuck_at is None
        assert s.dropout_prob == 0.0

    def test_full_dropout_returns_none(self, level_state):
        s = Accelerometer(dropout_prob=1.0, seed=0)
        assert s.read(level_state) is None

    def test_dropout_rate_matches_probability(self, level_state):
        s = Accelerometer(dropout_prob=0.25, seed=1234)
        dropped = sum(s.read(level_state) is None for _ in range(4000))
        assert 0.20 < dropped / 4000 < 0.30

    def test_stuck_at_latches_after_start(self, level_state):
        s = Accelerometer(stuck_at=[1.0, 2.0, 3.0], fault_start=0.5, seed=0)
        level_state.t = 0.0
        np.testing.assert_allclose(s.read(level_state).data, [0.0, 0.0, GRAVITY], atol=1e-12)
        level_state.t = 0.6
        np.testing.assert_allclose(s.read(level_state).data, [1.0, 2.0, 3.0], atol=1e-12)
        level_state.t = 10.0
        np.testing.assert_allclose(s.read(level_state).data, [1.0, 2.0, 3.0], atol=1e-12)

    def test_scale_factor_is_multiplicative(self, level_state):
        s = Accelerometer(noise_std=0.0, scale_factor=1.05, seed=0)
        np.testing.assert_allclose(s.read(level_state).data, np.array([0, 0, GRAVITY]) * 1.05,
                                   atol=1e-9)

    def test_misalignment_rotates_measurement(self, level_state):
        """A 90° yaw mounting error swaps the x and y channels."""
        level_state.acceleration_ned = np.array([1.0, 0.0, 0.0])
        level_state.t = 0.0
        clean = Accelerometer(noise_std=0.0).read(level_state).data
        rotated = Accelerometer(noise_std=0.0, misalignment_deg=(0.0, 0.0, 90.0)).read(level_state).data
        # Rz(90°) @ [1, 0, g] = [0, 1, g]
        np.testing.assert_allclose(rotated, [0.0, 1.0, GRAVITY], atol=1e-9)
        assert not np.allclose(clean, rotated)

    def test_dropped_samples_are_absent_from_log(self):
        """The simulator omits dropped samples rather than logging zeros."""
        sim = Simulator(dt_physics=0.01, seed=5)
        sim.add_sensor(DVL(dropout_prob=1.0), frequency_hz=10)
        history = sim.run(max_time=2.0)
        assert all("dvl" not in entry.readings for entry in history)
        assert sim.sensor_arrays("dvl") is None
