"""Tests for the simulation clock and its multi-rate sensor scheduling."""

import numpy as np
import pytest

from auv_simulator.simulation_clock import SimulationClock


class TestTickGeneration:
    def test_tick_count_includes_both_endpoints(self):
        clock = SimulationClock(duration=1.0, dt_physics=0.01)
        ticks = list(clock)
        assert len(ticks) == 101
        assert clock.n_ticks == 101

    def test_timestamps_start_at_zero_and_end_at_duration(self):
        ticks = list(SimulationClock(duration=2.0, dt_physics=0.5))
        assert ticks[0].t == 0.0
        assert ticks[-1].t == pytest.approx(2.0)

    def test_timestamps_are_uniform_multiples_of_dt(self):
        ticks = list(SimulationClock(duration=0.2, dt_physics=0.02))
        np.testing.assert_allclose([k.t for k in ticks], np.arange(11) * 0.02, atol=1e-12)

    def test_timestamps_are_monotonic(self):
        times = [k.t for k in SimulationClock(duration=5.0, dt_physics=0.005)]
        assert all(b > a for a, b in zip(times, times[1:]))

    def test_dt_changes_resolution(self):
        coarse = len(list(SimulationClock(duration=10.0, dt_physics=0.1)))
        fine = len(list(SimulationClock(duration=10.0, dt_physics=0.01)))
        assert coarse == 101
        assert fine == 1001

    def test_index_matches_position_in_sequence(self):
        for expected, tick in enumerate(SimulationClock(duration=0.5, dt_physics=0.1)):
            assert tick.index == expected
            assert tick.t == pytest.approx(expected * 0.1)

    def test_timestamp_helper_rejects_out_of_range(self):
        clock = SimulationClock(duration=1.0, dt_physics=0.1)
        assert clock.timestamp(3) == pytest.approx(0.3)
        with pytest.raises(IndexError):
            clock.timestamp(clock.n_ticks)


class TestValidation:
    @pytest.mark.parametrize("kwargs", [
        dict(duration=-1.0, dt_physics=0.01),
        dict(duration=1.0, dt_physics=0.0),
        dict(duration=1.0, dt_physics=-0.5),
    ])
    def test_invalid_time_arguments_raise(self, kwargs):
        with pytest.raises(ValueError):
            SimulationClock(**kwargs)

    def test_non_positive_sensor_period_raises(self):
        with pytest.raises(ValueError):
            SimulationClock(duration=1.0, dt_physics=0.01, sensor_periods={"bad": 0.0})


class TestSensorScheduling:
    def test_no_sensors_means_no_due_entries(self):
        due = [k.due_sensors for k in SimulationClock(duration=0.1, dt_physics=0.01)]
        assert set(due) == {()}

    def test_slow_sensor_samples_at_its_own_rate(self):
        clock = SimulationClock(duration=10.0, dt_physics=0.005,
                                sensor_periods={"pressure": 0.1})
        fired = [k.t for k in clock if "pressure" in k.due_sensors]
        assert len(fired) == 101
        np.testing.assert_allclose(np.diff(fired), 0.1, atol=1e-12)

    def test_fast_and_slow_sensors_coexist(self):
        clock = SimulationClock(duration=5.0, dt_physics=0.005,
                                sensor_periods={"imu": 0.01, "mag": 0.02,
                                                "pressure": 0.1})
        counts = {"imu": 0, "mag": 0, "pressure": 0}
        for tick in clock:
            for name in tick.due_sensors:
                counts[name] += 1
        assert counts == {"imu": 501, "mag": 251, "pressure": 51}

    def test_expected_sample_counts_match_observed(self):
        clock = SimulationClock(duration=3.0, dt_physics=0.01,
                                sensor_periods={"imu": 0.01, "pressure": 0.25})
        observed = {name: 0 for name in clock.sensor_periods}
        for tick in clock:
            for name in tick.due_sensors:
                observed[name] += 1
        for name, count in observed.items():
            assert count == clock.expected_samples(name)

    def test_sensor_fires_on_the_very_first_tick(self):
        first = next(iter(SimulationClock(duration=1.0, dt_physics=0.01,
                                          sensor_periods={"imu": 0.1})))
        assert "imu" in first.due_sensors

    def test_schedule_order_is_registration_order(self):
        clock = SimulationClock(duration=1.0, dt_physics=0.01,
                                sensor_periods={"imu": 0.01, "mag": 0.01})
        assert next(iter(clock)).due_sensors == ("imu", "mag")
