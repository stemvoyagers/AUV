"""Tests for the DVL sensor and its role in making position observable."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from auv_simulator.core.frames import GRAVITY, euler_to_quaternion
from auv_simulator.core.state import VehicleState
from auv_simulator.navigation import ESKF
from auv_simulator.sensors import DVL


@pytest.fixture
def moving_state() -> VehicleState:
    """Heading 90° (east), moving 5 m/s east, level."""
    return VehicleState(
        t=0.0,
        position=np.zeros(3),
        velocity=np.array([0.0, 5.0, 0.0]),
        quaternion=euler_to_quaternion(np.pi / 2, 0.0, 0.0),
    )


class TestDVLPhysics:
    def test_ideal_is_body_frame_velocity(self, moving_state):
        """Heading east at 5 m/s → the DVL sees its velocity along body-x."""
        np.testing.assert_allclose(DVL().ideal(moving_state), [5.0, 0.0, 0.0], atol=1e-9)

    def test_ideal_rotates_with_pitch(self, moving_state):
        """Nose-down 30° with a 5 m/s horizontal velocity → surge < 5, heave < 0."""
        moving_state.quaternion = euler_to_quaternion(np.pi / 2, np.deg2rad(-30), 0.0)
        v = DVL().ideal(moving_state)
        assert v[0] == pytest.approx(5.0 * np.cos(np.deg2rad(30)), abs=1e-9)
        # Nose-down: the water flows "up" past the down-pointing axis.
        assert v[2] == pytest.approx(-5.0 * np.sin(np.deg2rad(30)), abs=1e-9)

    def test_noise_and_bias_are_applied(self, moving_state):
        s = DVL(noise_std=0.0, bias=0.05, seed=0)
        np.testing.assert_allclose(s.read(moving_state).data, [5.05, 0.05, 0.05], atol=1e-9)


class TestDVLCompletesObservability:
    def test_body_velocity_converges_to_measurement(self):
        """The DVL observes v_body, so that is precisely what must converge.

        Note the world-frame velocity may keep a yaw offset: rotating the
        attitude and the NED velocity together leaves ``R_NB·v_ned`` unchanged,
        so a DVL alone cannot observe heading.  That is why the mag stays.
        """
        from auv_simulator.core.frames import ned_to_body_rotation

        eskf = ESKF()
        for _ in range(200):                       # 20 s at 10 Hz
            eskf.propagate(0.1, [0.0, 0.0, GRAVITY], [0.0, 0.0, 0.0])
            eskf.update_dvl([5.0, 0.0, 0.0])

        R_NB = ned_to_body_rotation(*eskf.yaw_pitch_roll)
        np.testing.assert_allclose(R_NB @ eskf.velocity, [5.0, 0.0, 0.0], atol=0.05)
        assert np.linalg.norm(eskf.velocity) == pytest.approx(5.0, abs=0.05)

    def test_position_stays_bounded_with_dvl(self):
        """Full 120 s trajectory: DVL keeps horizontal error small.

        Without a DVL the same run drifts to hundreds of metres; this is the
        regression test for that defect class.
        """
        from .conftest import build_reference_sim

        sim = build_reference_sim()
        history = sim.run()
        est = ESKF().run(history)

        truth = np.vstack([h.state.position for h in history])
        got = np.vstack([e.p for e in est])
        horiz_err = np.linalg.norm((got - truth)[:, :2], axis=1)

        assert horiz_err.max() < 5.0, f"horizontal error grew to {horiz_err.max():.1f} m"
        assert horiz_err[-1] < 2.0, f"final horizontal error {horiz_err[-1]:.2f} m"

    def test_dvl_dropout_degrades_gracefully(self):
        """Losing 50 % of DVL samples must not destabilise the filter."""
        from .conftest import build_reference_sim

        sim = build_reference_sim(dvl_dropout=0.5)
        history = sim.run(max_time=60.0)
        est = ESKF().run(history)

        truth = np.vstack([h.state.position for h in history])
        got = np.vstack([e.p for e in est])
        horiz_err = np.linalg.norm((got - truth)[:, :2], axis=1)

        assert np.all(np.isfinite(got))
        assert horiz_err[-1] < 3.0, f"error after dropout {horiz_err[-1]:.2f} m"

    def test_run_survives_missing_all_aiding(self):
        """No mag / no pressure / no DVL for 30 s — filter must still return."""
        from .conftest import build_reference_sim

        sim = build_reference_sim(mag_dropout=1.0, pressure_dropout=1.0, dvl_dropout=1.0)
        history = sim.run(max_time=30.0)
        est = ESKF().run(history)

        assert len(est) > 0
        assert np.all(np.isfinite(np.vstack([e.p for e in est])))
