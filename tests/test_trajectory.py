"""Tests for the analytic 120-second trajectory.

The scenario under test:
    0-10 s    stationary at 20 m, heading north
    10-20 s   smooth acceleration to 0.8 m/s
    20-35 s   constant 0.8 m/s north
    35-65 s   smooth 90 deg turn towards the east
    65-75 s   constant 0.8 m/s east
    75-105 s  descent from 20 m to 24 m
    105-120 s constant 0.8 m/s at 24 m
"""

import numpy as np
import pytest

from auv_simulator.state import depth_to_pressure, pressure_to_depth
from auv_simulator.trajectory import (
    CRUISE_SPEED,
    DESCENT_DEPTH_CHANGE,
    TURN_ANGLE,
    TURN_DURATION,
    Trajectory,
)

TURN_RADIUS = CRUISE_SPEED / (TURN_ANGLE / TURN_DURATION)


class TestStageTable:
    def test_duration_is_120_seconds(self, trajectory):
        assert trajectory.duration == pytest.approx(120.0)

    def test_has_seven_stages(self, trajectory):
        assert len(trajectory.stages()) == 7

    def test_stages_are_contiguous_and_cover_the_whole_run(self, trajectory):
        stages = trajectory.stages()
        assert stages[0].t_start == 0.0
        assert stages[-1].t_end == pytest.approx(120.0)
        for current, nxt in zip(stages, stages[1:]):
            assert current.t_end == pytest.approx(nxt.t_start)

    def test_stage_names_describe_the_brief(self, trajectory):
        names = [s.name for s in trajectory.stages()]
        assert names == ["Stationary", "Acceleration", "North cruise", "Turning",
                         "East cruise", "Descending", "Hold"]


class TestInitialConditions:
    def test_starts_stationary_at_twenty_metres(self, trajectory):
        s = trajectory.state_at(0.0)
        np.testing.assert_allclose(s.position, [0.0, 0.0, 20.0], atol=1e-12)
        np.testing.assert_allclose(s.velocity, [0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(s.acceleration_ned, [0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(s.angular_velocity, [0.0, 0.0, 0.0], atol=1e-12)

    def test_faces_north(self, trajectory):
        assert trajectory.state_at(0.0).heading_deg == pytest.approx(0.0, abs=1e-9)

    def test_initial_depth_is_configurable(self):
        deep = Trajectory(initial_depth=35.0)
        assert deep.state_at(0.0).depth == pytest.approx(35.0)
        assert deep.state_at(120.0).depth == pytest.approx(35.0 + DESCENT_DEPTH_CHANGE)

    def test_initial_depth_does_not_alter_horizontal_motion(self):
        default = Trajectory()
        deeper = Trajectory(initial_depth=50.0)
        for t in (20.0, 40.0, 65.0, 90.0, 120.0):
            np.testing.assert_allclose(deeper.state_at(t).position[:2],
                                       default.state_at(t).position[:2], atol=1e-9)


class TestStationaryStage:
    def test_holds_depth_and_position_until_ten_seconds(self, trajectory):
        for t in (0.0, 3.0, 9.999):
            s = trajectory.state_at(t)
            np.testing.assert_allclose(s.position, [0.0, 0.0, 20.0], atol=1e-9)
            assert s.speed == pytest.approx(0.0, abs=1e-12)


class TestAccelerationStage:
    def test_velocity_starts_from_rest(self, trajectory):
        assert trajectory.state_at(10.0).speed == pytest.approx(0.0, abs=1e-9)

    def test_reaches_target_speed_at_twenty_seconds(self, trajectory):
        assert trajectory.state_at(20.0).speed == pytest.approx(CRUISE_SPEED, rel=1e-9)

    def test_moves_north_only(self, trajectory):
        for t in np.linspace(10.1, 19.9, 12):
            s = trajectory.state_at(t)
            assert s.velocity[0] > 0.0
            assert s.velocity[1] == pytest.approx(0.0, abs=1e-12)

    def test_acceleration_is_positive_north_and_bounded(self, trajectory):
        for t in np.linspace(10.1, 19.9, 12):
            acc = trajectory.state_at(t).acceleration_ned
            assert acc[0] > 0.0
            assert acc[0] < CRUISE_SPEED
            assert abs(acc[1]) < 1e-12 and abs(acc[2]) < 1e-12

    def test_acceleration_is_zero_at_both_ends_for_smoothness(self, trajectory):
        assert trajectory.state_at(10.0).acceleration_ned[0] == pytest.approx(0.0, abs=1e-9)
        assert trajectory.state_at(20.0).acceleration_ned[0] == pytest.approx(0.0, abs=1e-9)

    def test_distance_travelled_is_four_metres(self, trajectory):
        # Half the duration at the average of a smooth 0 -> 0.8 ramp.
        assert trajectory.state_at(20.0).north == pytest.approx(4.0, abs=1e-9)


class TestNorthCruiseStage:
    def test_constant_velocity_and_zero_acceleration(self, trajectory):
        for t in (25.0, 30.0, 34.9):
            s = trajectory.state_at(t)
            np.testing.assert_allclose(s.velocity, [CRUISE_SPEED, 0.0, 0.0], atol=1e-9)
            np.testing.assert_allclose(s.acceleration_ned, [0.0, 0.0, 0.0], atol=1e-12)

    def test_position_at_stage_end(self, trajectory):
        # 4 m from the acceleration stage + 15 s at 0.8 m/s
        assert trajectory.state_at(35.0).north == pytest.approx(16.0, abs=1e-9)

    def test_depth_is_held(self, trajectory):
        assert trajectory.state_at(35.0).depth == pytest.approx(20.0, abs=1e-9)


class TestTurningStage:
    def test_yaw_rate_is_constant_three_degrees_per_second(self, trajectory):
        expected = np.deg2rad(3.0)
        for t in (36.0, 50.0, 64.0):
            assert trajectory.state_at(t).angular_velocity[2] == pytest.approx(expected, rel=1e-9)

    def test_heading_starts_north_and_ends_east(self, trajectory):
        assert trajectory.state_at(35.0).heading_deg == pytest.approx(0.0, abs=1e-6)
        assert trajectory.state_at(65.0).heading_deg == pytest.approx(90.0, abs=1e-6)

    def test_turn_is_exactly_ninety_degrees(self, trajectory):
        yaw_start = trajectory.state_at(35.0).yaw_pitch_roll[0]
        yaw_end = trajectory.state_at(65.0).yaw_pitch_roll[0]
        assert yaw_end - yaw_start == pytest.approx(np.pi / 2, abs=1e-9)

    def test_speed_is_constant_through_the_arc(self, trajectory):
        for t in (36.0, 45.0, 55.0, 64.0):
            assert trajectory.state_at(t).speed == pytest.approx(CRUISE_SPEED, rel=1e-9)

    def test_arc_geometry(self, trajectory):
        s = trajectory.state_at(65.0)
        assert s.north == pytest.approx(16.0 + TURN_RADIUS, abs=1e-6)
        assert s.east == pytest.approx(TURN_RADIUS, abs=1e-6)

    def test_turn_goes_east_not_back_south(self, trajectory):
        assert trajectory.state_at(50.0).east > 0.0
        assert trajectory.state_at(65.0).north > 30.0


class TestEastCruiseStage:
    def test_heading_is_east_and_speed_is_constant(self, trajectory):
        for t in (66.0, 70.0, 74.9):
            s = trajectory.state_at(t)
            assert s.heading_deg == pytest.approx(90.0, abs=1e-6)
            np.testing.assert_allclose(s.velocity, [0.0, CRUISE_SPEED, 0.0], atol=1e-9)

    def test_depth_is_still_twenty_metres(self, trajectory):
        assert trajectory.state_at(75.0).depth == pytest.approx(20.0, abs=1e-9)

    def test_east_advance_is_eight_metres(self, trajectory):
        start_east = trajectory.state_at(65.0).east
        assert trajectory.state_at(75.0).east == pytest.approx(start_east + 8.0, abs=1e-9)


class TestDescentStage:
    def test_depth_rises_from_twenty_to_twenty_four(self, trajectory):
        assert trajectory.state_at(75.0).depth == pytest.approx(20.0, abs=1e-9)
        assert trajectory.state_at(105.0).depth == pytest.approx(24.0, abs=1e-9)

    def test_depth_increases_monotonically(self, trajectory):
        depths = [trajectory.state_at(t).depth for t in np.linspace(75.0, 105.0, 61)]
        assert all(b >= a for a, b in zip(depths, depths[1:]))

    def test_descends_exactly_four_metres(self, trajectory):
        change = trajectory.state_at(105.0).depth - trajectory.state_at(75.0).depth
        assert change == pytest.approx(DESCENT_DEPTH_CHANGE, abs=1e-9)

    def test_horizontal_speed_is_maintained(self, trajectory):
        for t in (76.0, 90.0, 104.0):
            v = trajectory.state_at(t).velocity
            assert np.hypot(v[0], v[1]) == pytest.approx(CRUISE_SPEED, rel=1e-9)

    def test_still_travelling_east_while_descending(self, trajectory):
        assert trajectory.state_at(90.0).velocity[1] == pytest.approx(CRUISE_SPEED, rel=1e-9)

    def test_pitch_is_nose_down(self, trajectory):
        pitch = trajectory.state_at(90.0).yaw_pitch_roll[1]
        assert pitch < 0.0
        assert pitch > -np.deg2rad(25.0)

    def test_pitch_starts_and_ends_level(self, trajectory):
        assert trajectory.state_at(75.0).yaw_pitch_roll[1] == pytest.approx(0.0, abs=1e-9)
        assert trajectory.state_at(105.0).yaw_pitch_roll[1] == pytest.approx(0.0, abs=1e-9)

    def test_vertical_speed_is_zero_at_both_ends(self, trajectory):
        assert trajectory.state_at(75.0).velocity[2] == pytest.approx(0.0, abs=1e-9)
        assert trajectory.state_at(105.0).velocity[2] == pytest.approx(0.0, abs=1e-9)


class TestHoldStage:
    def test_depth_is_held_at_twenty_four_metres(self, trajectory):
        for t in (105.0, 112.0, 120.0):
            assert trajectory.state_at(t).depth == pytest.approx(24.0, abs=1e-9)

    def test_speed_and_heading_are_maintained(self, trajectory):
        s = trajectory.state_at(120.0)
        assert s.speed == pytest.approx(CRUISE_SPEED, rel=1e-9)
        assert s.heading_deg == pytest.approx(90.0, abs=1e-6)

    def test_final_position(self, trajectory):
        s = trajectory.state_at(120.0)
        expected_north = 16.0 + TURN_RADIUS
        expected_east = TURN_RADIUS + 8.0 + 24.0 + 12.0
        assert s.north == pytest.approx(expected_north, abs=1e-6)
        assert s.east == pytest.approx(expected_east, abs=1e-6)
        assert s.depth == pytest.approx(24.0, abs=1e-9)


class TestContinuity:
    @pytest.mark.parametrize("t_boundary", [10.0, 20.0, 35.0, 65.0, 75.0, 105.0])
    def test_position_and_velocity_are_continuous(self, trajectory, t_boundary):
        before = trajectory.state_at(t_boundary - 1e-6)
        after = trajectory.state_at(t_boundary + 1e-6)
        np.testing.assert_allclose(before.position, after.position, atol=1e-5)
        np.testing.assert_allclose(before.velocity, after.velocity, atol=1e-5)

    def test_heading_does_not_jump(self, trajectory):
        for t_boundary in (35.0, 65.0):
            before = trajectory.state_at(t_boundary - 1e-6).heading_deg
            after = trajectory.state_at(t_boundary + 1e-6).heading_deg
            assert abs(after - before) < 1e-4


class TestSampling:
    def test_sample_returns_the_expected_number_of_states(self, trajectory):
        assert len(trajectory.sample(dt=0.1)) == 1201

    def test_sampling_resolution_does_not_change_the_truth(self, trajectory):
        coarse = trajectory.sample(dt=1.0)
        fine = trajectory.sample(dt=0.5)
        by_time = {round(s.t, 6): s for s in fine}
        for state in coarse:
            np.testing.assert_allclose(state.position, by_time[round(state.t, 6)].position,
                                       atol=1e-12)

    def test_sample_rejects_non_positive_dt(self, trajectory):
        with pytest.raises(ValueError):
            trajectory.sample(dt=0.0)

    def test_states_are_clamped_outside_the_window(self, trajectory):
        np.testing.assert_allclose(trajectory.state_at(-5.0).position,
                                   trajectory.state_at(0.0).position, atol=1e-12)
        np.testing.assert_allclose(trajectory.state_at(500.0).position,
                                   trajectory.state_at(120.0).position, atol=1e-12)


class TestDepthVsPressureConsistency:
    def test_final_depth_round_trips_through_pressure(self, trajectory):
        depth = trajectory.state_at(120.0).depth
        np.testing.assert_allclose(pressure_to_depth(depth_to_pressure(depth)), depth, atol=1e-9)
