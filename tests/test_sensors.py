
import numpy as np
import pytest

from auv_simulator.sensors import (
    Accelerometer,
    Gyroscope,
    Magnetometer,
    PressureSensor,
)
from auv_simulator.simulator import Simulator
from auv_simulator.state import (
    GRAVITY,
    VehicleState,
    depth_to_pressure,
    ned_to_body_rotation,
    pressure_to_depth,
)


@pytest.fixture
def level_state() -> VehicleState:
    return VehicleState(t=0.0, position=np.zeros(3), velocity=np.zeros(3),
                        acceleration_ned=np.zeros(3),
                        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
                        angular_velocity=np.zeros(3))


@pytest.fixture
def submerged_state() -> VehicleState:
    return VehicleState(t=0.0, position=np.array([0.0, 0.0, 20.0]))


def _specific_force(state: VehicleState) -> np.ndarray:
    r = ned_to_body_rotation(*state.yaw_pitch_roll)
    gravity_ned = np.array([0.0, 0.0, -GRAVITY])
    return r @ (state.acceleration_ned - gravity_ned)


class TestAccelerometerIdeal:
    def test_level_at_rest_reads_gravity(self, level_state):
        np.testing.assert_allclose(Accelerometer().ideal(level_state),
                                   [0.0, 0.0, GRAVITY], atol=1e-9)

    def test_level_accelerating_north(self, level_state):
        level_state.acceleration_ned = np.array([2.0, 0.0, 0.0])
        np.testing.assert_allclose(Accelerometer().ideal(level_state),
                                   [2.0, 0.0, GRAVITY], atol=1e-9)

    def test_matches_state_helper(self, level_state):
        level_state.acceleration_ned = np.array([0.0, 1.5, 0.0])
        np.testing.assert_allclose(Accelerometer().ideal(level_state),
                                   _specific_force(level_state), atol=1e-12)

    def test_zero_noise_repeats_exactly(self, level_state):
        sensor = Accelerometer(noise_std=0.0, bias=0.0, noise_density=0.0, seed=0)
        first = sensor.read(level_state).data
        second = sensor.read(level_state).data
        np.testing.assert_allclose(first, second, atol=1e-15)
        np.testing.assert_allclose(first, _specific_force(level_state), atol=1e-15)


class TestGyroscopeIdeal:
    def test_zero_rate_reads_zero(self, level_state):
        np.testing.assert_allclose(Gyroscope().ideal(level_state), [0, 0, 0], atol=1e-12)

    def test_reads_body_rate(self, level_state):
        level_state.angular_velocity = np.array([0.0, 0.0, 0.1571])
        np.testing.assert_allclose(Gyroscope().ideal(level_state), [0.0, 0.0, 0.1571], atol=1e-12)


class TestMagnetometerIdeal:
    def test_level_north_reading_is_earth_field(self, level_state):
        np.testing.assert_allclose(Magnetometer().ideal(level_state),
                                   Magnetometer.EARTH_FIELD_NED, atol=1e-12)

    def test_heading_east_swaps_horizontal_component(self, level_state):
        level_state.quaternion = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
        reading = Magnetometer().ideal(level_state)
        assert reading[0] == pytest.approx(0.0, abs=1e-9)
        assert reading[1] == pytest.approx(-Magnetometer.EARTH_FIELD_NED[0], abs=1e-9)

    def test_reading_magnitude_is_preserved(self, level_state):
        level_state.quaternion = np.array([np.cos(0.3), 0.0, np.sin(0.3), 0.0])
        assert np.linalg.norm(Magnetometer().ideal(level_state)) == pytest.approx(
            np.linalg.norm(Magnetometer.EARTH_FIELD_NED), rel=1e-12)


class TestPressureIdeal:
    def test_surface_reads_atmospheric(self, level_state):
        assert PressureSensor().ideal(level_state)[0] == pytest.approx(101_325.0, abs=1.0)

    def test_twenty_metres_reads_about_three_bar(self, submerged_state):
        expected = depth_to_pressure(20.0)
        assert PressureSensor().ideal(submerged_state)[0] == pytest.approx(expected, rel=1e-12)
        assert expected > 3.0e5

    def test_pressure_increases_with_depth(self):
        shallow = PressureSensor().ideal(VehicleState(position=np.array([0.0, 0.0, 5.0])))[0]
        deep = PressureSensor().ideal(VehicleState(position=np.array([0.0, 0.0, 24.0])))[0]
        assert deep > shallow

    def test_pressure_can_be_converted_back_to_depth(self, submerged_state):
        reading = PressureSensor().ideal(submerged_state)[0]
        assert pressure_to_depth(reading) == pytest.approx(20.0, abs=1e-9)


class TestSensorNonIdealities:
    def test_noise_adds_variation(self, level_state):
        sensor = Accelerometer(noise_std=1.0, seed=42)
        samples = np.array([sensor.read(level_state).data for _ in range(1000)])
        assert np.std(samples, axis=0).min() > 0.5

    def test_bias_is_persistent(self, level_state):
        sensor = Accelerometer(noise_std=0.0, bias=np.array([0.1, 0.2, -0.3]), seed=0)
        ideal = Accelerometer().ideal(level_state)
        for _ in range(10):
            np.testing.assert_allclose(sensor.read(level_state).data - ideal,
                                       [0.1, 0.2, -0.3], atol=1e-9)

    def test_drift_walks_over_time(self, level_state):
        sensor = Gyroscope(noise_std=0.0, bias=0.0, noise_density=0.1, seed=42)
        sensor.reset()
        magnitudes = [np.linalg.norm(sensor.read(level_state, dt=0.01).data)
                      for _ in range(200)]
        assert magnitudes[-1] > magnitudes[0]

    def test_same_seed_reproduces_the_same_noise(self, level_state):
        a = Accelerometer(noise_std=0.5, seed=123)
        b = Accelerometer(noise_std=0.5, seed=123)
        np.testing.assert_allclose([a.read(level_state).data for _ in range(5)],
                                   [b.read(level_state).data for _ in range(5)], atol=0)

    def test_ideal_is_unaffected_by_noise(self, level_state):
        noisy = Accelerometer(noise_std=5.0, bias=3.0, seed=1)
        reading = noisy.read(level_state)
        np.testing.assert_allclose(reading.data_ideal, Accelerometer().ideal(level_state), atol=1e-12)
        assert reading.noise is not None


class TestFaultInjection:
    def test_default_sensor_has_no_faults(self, level_state):
        sensor = Accelerometer(seed=0)
        assert sensor.read(level_state) is not None
        assert sensor.stuck_at is None
        assert sensor.dropout_prob == 0.0

    def test_full_dropout_returns_none(self, level_state):
        assert Accelerometer(dropout_prob=1.0, seed=0).read(level_state) is None

    def test_dropout_rate_matches_probability(self, level_state):
        sensor = Accelerometer(dropout_prob=0.25, seed=1234)
        dropped = sum(sensor.read(level_state) is None for _ in range(4000))
        assert 0.20 < dropped / 4000 < 0.30

    def test_stuck_at_latches_after_start(self, level_state):
        sensor = Accelerometer(stuck_at=[1.0, 2.0, 3.0], fault_start=0.5, seed=0)
        level_state.t = 0.0
        np.testing.assert_allclose(sensor.read(level_state).data, [0.0, 0.0, GRAVITY], atol=1e-12)
        level_state.t = 0.6
        np.testing.assert_allclose(sensor.read(level_state).data, [1.0, 2.0, 3.0], atol=1e-12)
        level_state.t = 10.0
        np.testing.assert_allclose(sensor.read(level_state).data, [1.0, 2.0, 3.0], atol=1e-12)

    def test_scale_factor_is_multiplicative(self, level_state):
        sensor = Accelerometer(noise_std=0.0, scale_factor=1.05, seed=0)
        np.testing.assert_allclose(sensor.read(level_state).data,
                                   np.array([0, 0, GRAVITY]) * 1.05, atol=1e-9)

    def test_misalignment_rotates_measurement(self, level_state):
        level_state.acceleration_ned = np.array([1.0, 0.0, 0.0])
        clean = Accelerometer(noise_std=0.0).read(level_state).data
        rotated = Accelerometer(noise_std=0.0, misalignment_deg=(0.0, 0.0, 90.0)).read(level_state).data
        np.testing.assert_allclose(rotated, [0.0, 1.0, GRAVITY], atol=1e-9)
        assert not np.allclose(clean, rotated)

    def test_dropped_samples_are_absent_from_the_log(self):
        sim = Simulator(dt_physics=0.01, seed=5)
        sim.add_sensor(Accelerometer(dropout_prob=1.0), frequency_hz=10)
        history = sim.run(max_time=2.0)
        assert all("accelerometer" not in entry.readings for entry in history)
        assert sim.sensor_arrays("accelerometer") is None


class TestSensorsInSimulation:
    def test_pressure_tracks_trajectory_depth(self, reference_sim):
        reference_sim.run()
        name = "pressure"
        data = reference_sim.sensor_arrays(name)
        truth = reference_sim.ground_truth_arrays()

        depth_from_pressure = pressure_to_depth(data["data"][:, 0])
        truth_at_samples = np.interp(data["t"], truth["t"], truth["depth"])

        np.testing.assert_allclose(depth_from_pressure, truth_at_samples, atol=0.05)

    def test_ideal_readings_are_logged_alongside_noisy_ones(self, reference_sim):
        reference_sim.run()
        for name in reference_sim.sensor_names:
            arrays = reference_sim.sensor_arrays(name)
            assert arrays is not None
            assert arrays["ideal"] is not None
            assert arrays["ideal"].shape == arrays["data"].shape

    def test_every_sensor_reports_at_its_nominal_rate(self, reference_sim):
        reference_sim.run()
        expected = {"accelerometer": 100.0, "gyroscope": 100.0,
                    "magnetometer": 50.0, "pressure": 10.0}
        for name, hz in expected.items():
            count = len(reference_sim.sensor_arrays(name)["t"])
            assert count == pytest.approx(120.0 * hz + 1, abs=1)

    def test_ground_truth_reflects_the_scenario(self, reference_sim):
        reference_sim.run()
        truth = reference_sim.ground_truth_arrays()
        assert truth["depth"][0] == pytest.approx(20.0, abs=1e-9)
        assert truth["depth"][-1] == pytest.approx(24.0, abs=1e-9)
        assert truth["position"][-1, 1] > truth["position"][0, 1]
