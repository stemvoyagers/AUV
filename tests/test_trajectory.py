"""Tests for the analytic 120-second trajectory."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from auv_simulator.core.trajectory import Trajectory, STAGES, TURN_YAW_RATE
from auv_simulator.core.state import VehicleState


@pytest.fixture
def traj() -> Trajectory:
    return Trajectory()


class TestStageBoundaries:
    def test_duration_is_120_seconds(self, traj):
        assert traj.duration == pytest.approx(120.0)

    def test_stage_times_are_monotonic(self):
        for i in range(len(STAGES) - 1):
            assert STAGES[i].t_end == pytest.approx(STAGES[i + 1].t_start)

    def test_all_stages_cover_zero_to_120(self):
        assert STAGES[0].t_start == 0.0
        assert STAGES[-1].t_end == 120.0


class TestStationaryStage:
    def test_starts_at_origin(self, traj):
        s = traj.state_at(0.0)
        np.testing.assert_allclose(s.position, [0, 0, 0], atol=1e-9)
        np.testing.assert_allclose(s.velocity, [0, 0, 0], atol=1e-9)

    def test_remains_stationary_at_stage_end(self, traj):
        s = traj.state_at(10.0 - 1e-6)
        np.testing.assert_allclose(s.position, [0, 0, 0], atol=1e-6)
        np.testing.assert_allclose(s.velocity, [0, 0, 0], atol=1e-6)
        np.testing.assert_allclose(s.angular_velocity, [0, 0, 0], atol=1e-9)


class TestAccelerationStage:
    def test_acceleration_is_constant_north(self, traj):
        s = traj.state_at(15.0)
        np.testing.assert_allclose(s.acceleration_ned, [0.5, 0, 0], atol=1e-9)
        np.testing.assert_allclose(s.velocity[1:], [0, 0], atol=1e-9)

    def test_ends_at_cruise_speed(self, traj):
        s = traj.state_at(20.0 - 1e-6)
        assert np.linalg.norm(s.velocity) == pytest.approx(5.0, rel=1e-3)
        # position = 0.5 * 0.5 * 10² = 25 m north
        np.testing.assert_allclose(s.position, [25.0, 0, 0], atol=1e-3)


class TestStraightRunStage:
    def test_constant_velocity(self, traj):
        s = traj.state_at(35.0)
        np.testing.assert_allclose(s.velocity, [5.0, 0, 0], atol=1e-6)
        np.testing.assert_allclose(s.acceleration_ned, [0, 0, 0], atol=1e-6)

    def test_position_at_stage_end(self, traj):
        # 25 m + 30 s * 5 m/s = 175 m north
        s = traj.state_at(50.0 - 1e-6)
        np.testing.assert_allclose(s.position, [175.0, 0, 0], atol=1e-2)


class TestTurningStage:
    def test_yaw_rate_is_constant_and_correct(self, traj):
        # Stage 3 = 50..70 s, turn right at 9°/s = 0.1571 rad/s
        s = traj.state_at(60.0)
        assert s.angular_velocity[2] == pytest.approx(TURN_YAW_RATE, rel=1e-6)

    def test_heading_changes_by_180_degrees(self, traj):
        # 20 s × 9°/s = 180° right turn
        s0 = traj.state_at(50.0)
        s1 = traj.state_at(70.0 - 1e-6)
        yaw0, _, _ = s0.yaw_pitch_roll
        yaw1, _, _ = s1.yaw_pitch_roll
        delta = yaw1 - yaw0
        # 180° right turn
        assert abs((delta % (2 * np.pi)) - np.pi) < 1e-4


class TestDescendingStage:
    def test_depth_becomes_positive(self, traj):
        s = traj.state_at(85.0)
        assert s.depth > 0.0

    def test_pitch_is_down(self, traj):
        s = traj.state_at(80.0)
        _, pitch, _ = s.yaw_pitch_roll
        assert pitch < 0.0   # nose down


class TestFinalState:
    def test_final_depth_is_positive(self, traj):
        s = traj.state_at(120.0)
        assert s.depth > 0.0
        # Final depth is analytically determined by descent speed + pitch
        # with 5 s linear ramps at stage boundaries (not instantaneous pitch)
        s90 = traj.state_at(90.0)
        assert s.position[2] == pytest.approx(s90.position[2], rel=1e-4)
        # Depth should be a meaningful ~5-10 m value
        assert 1.0 < s.position[2] < 20.0

    def test_final_heading_is_180_degrees(self, traj):
        s = traj.state_at(120.0)
        yaw, _, _ = s.yaw_pitch_roll
        assert abs((yaw % (2 * np.pi)) - np.pi) < 1e-3

    def test_final_depth_matches_pressure(self, traj):
        from auv_simulator.core.frames import depth_to_pressure, pressure_to_depth
        s = traj.state_at(120.0)
        P = depth_to_pressure(s.depth)
        d_back = pressure_to_depth(P)
        np.testing.assert_allclose(d_back, s.depth, atol=1e-6)


class TestStateAtBoundaries:
    """Trajectory should be continuous across stage boundaries."""

    @pytest.mark.parametrize("t_boundary", [10.0, 20.0, 50.0, 70.0, 90.0])
    def test_continuity(self, traj, t_boundary):
        """Position must be continuous across stage boundaries.

        Velocity CAN change discontinuously — e.g. descending stage (pitched
        body) has a different NED velocity vector than straight run (level).
        Position continuity is the hard requirement for ground truth.
        """
        s_before = traj.state_at(t_boundary - 0.001)
        s_after = traj.state_at(t_boundary + 0.001)
        np.testing.assert_allclose(s_before.position, s_after.position, atol=0.1)
