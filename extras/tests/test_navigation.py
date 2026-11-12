"""Tests for the ESKF navigation stub — simple sanity checks."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from auv_simulator.navigation import ESKF
from auv_simulator.core.frames import GRAVITY
from auv_simulator.core.state import VehicleState


class TestESKFPropagation:
    def test_static_imu_drifts_slowly(self):
        """Propagating with zero IMU should leave position almost unchanged."""
        eskf = ESKF()
        initial_pos = eskf.position.copy()
        dt = 0.01
        for _ in range(1000):
            eskf.propagate(dt, accel_body=[0, 0, GRAVITY], gyro_body=[0, 0, 0])
        displacement = np.linalg.norm(eskf.position - initial_pos)
        # small noise-only drift → < 1 m
        assert displacement < 1.0

    def test_accel_north_moves_north(self):
        eskf = ESKF()
        dt = 0.01
        for _ in range(500):  # 5 s
            eskf.propagate(dt, accel_body=[1.0, 0.0, GRAVITY], gyro_body=[0, 0, 0])
        # body-x is forward (north), accel 1 m/s² for 5 s → ~12.5 m north
        assert eskf.position[0] > 0.0
        assert eskf.position[0] < 50.0   # safety bound


class TestESKFMeasurementUpdate:
    def test_pressure_corrects_depth(self):
        eskf = ESKF()
        # Inject wrong depth
        eskf.nominal.p[2] = 50.0
        eskf.update_pressure(depth_meas=10.0, noise_r=0.01)
        # Should now be much closer to 10 m
        assert abs(eskf.position[2] - 10.0) < 5.0

    def test_magnetometer_constrains_yaw(self):
        eskf = ESKF()
        # Large initial yaw uncertainty
        eskf.P[6, 6] = 1.0   # ~57° variance
        # Magnetometer reading for North-facing vehicle (body-frame mag = NED field rotated to body)
        from auv_simulator.sensors.magnetometer import Magnetometer
        m = Magnetometer.EARTH_FIELD_NED   # (24, 0, -22) in NED
        # zero yaw → R_NB = I → mag_body should ≈ m
        eskf.update_magnetometer(m, noise_r=0.01)
        # P should shrink
        assert eskf.P[6, 6] < 1.0


class TestTiltAid:
    def test_accelerometer_pulls_pitch_toward_gravity(self):
        """With the truth level, the tilt aid must drive an initial pitch error to zero."""
        from auv_simulator.core.frames import euler_to_quaternion

        eskf = ESKF()
        eskf.nominal.q = euler_to_quaternion(0.0, 0.1, 0.0)   # 5.7° nose-up error
        before = abs(eskf.yaw_pitch_roll[1])
        for _ in range(100):
            eskf.update_accelerometer([0.0, 0.0, GRAVITY])     # truth is level
        after = abs(eskf.yaw_pitch_roll[1])
        assert after < 0.5 * before, f"tilt aid failed: {before:.4f} → {after:.4f} rad"

    def test_accelerometer_does_not_touch_yaw(self):
        """Gravity carries no heading information; yaw must stay put."""
        from auv_simulator.core.frames import euler_to_quaternion

        eskf = ESKF()
        eskf.nominal.q = euler_to_quaternion(0.7, 0.0, 0.0)
        yaw_before = eskf.yaw_pitch_roll[0]
        for _ in range(100):
            eskf.update_accelerometer([0.0, 0.0, GRAVITY])
        assert abs(eskf.yaw_pitch_roll[0] - yaw_before) < 1e-3


class TestESKFPipeline:
    def _run(self, perfect: bool):
        from auv_simulator.simulator import Simulator
        from auv_simulator.sensors import Accelerometer, Gyroscope, Magnetometer, PressureSensor

        sim = Simulator(dt_physics=0.005, seed=7)
        if perfect:
            sim.add_sensor(Accelerometer(noise_std=0.0, bias=0.0, noise_density=0.0), 100)
            sim.add_sensor(Gyroscope(noise_std=0.0, bias=0.0, noise_density=0.0),     100)
            sim.add_sensor(Magnetometer(noise_std=0.0, bias=0.0),                      50)
            sim.add_sensor(PressureSensor(noise_std=0.0, bias=0.0),                    10)
        else:
            sim.add_sensor(Accelerometer(noise_std=0.01, bias=0.005, noise_density=1e-4), 100)
            sim.add_sensor(Gyroscope(noise_std=0.002, bias=0.001, noise_density=1e-5),    100)
            sim.add_sensor(Magnetometer(noise_std=0.1, bias=0.0),                         50)
            sim.add_sensor(PressureSensor(noise_std=5.0, bias=1.0),                       10)
        history = sim.run()
        eskf = ESKF()
        est = eskf.run(history)
        return history, est, eskf

    @staticmethod
    def _errors(history, est, t_min=20.0):
        t = np.array([h.t for h in history])
        late = t > t_min
        gt_depth = np.array([h.state.depth for h in history])
        es_depth = np.array([e.depth for e in est])
        gt_ypr = np.array([h.state.yaw_pitch_roll for h in history])
        es_ypr = np.array([e.yaw_pitch_roll for e in est])
        yaw_d = np.arctan2(np.sin(es_ypr[:, 0] - gt_ypr[:, 0]),
                           np.cos(es_ypr[:, 0] - gt_ypr[:, 0]))
        rmse = lambda a, b: float(np.sqrt(np.mean((a[late] - b[late]) ** 2)))
        return {
            "depth": rmse(gt_depth, es_depth),
            "yaw": float(np.degrees(rmse(yaw_d, np.zeros_like(yaw_d)))),
            "pitch": float(np.degrees(rmse(gt_ypr[:, 1], es_ypr[:, 1]))),
            "roll": float(np.degrees(rmse(gt_ypr[:, 2], es_ypr[:, 2]))),
        }

    def test_full_run_reasonable_accuracy(self):
        """IMU + accel-tilt + magnetometer + pressure over the full 120 s run.

        Absolute position is *unobservable* without DVL/GPS/USBL and is
        expected to drift.  Attitude and depth must be tightly tracked.
        """
        history, est, eskf = self._run(perfect=False)
        err = self._errors(history, est)

        assert err["depth"] < 0.02, f"depth RMSE {err['depth']:.4f} m"
        assert err["yaw"] < 0.5, f"yaw RMSE {err['yaw']:.3f}°"
        assert err["pitch"] < 0.3, f"pitch RMSE {err['pitch']:.3f}°"
        assert err["roll"] < 0.3, f"roll RMSE {err['roll']:.3f}°"

        # attitude covariance must have collapsed
        assert np.max(np.degrees(np.sqrt(np.diag(eskf.P)[6:9]))) < 1.0

    def test_truth_only_tracking(self):
        """With noiseless, bias-free sensors the filter must track truth almost exactly."""
        history, est, _ = self._run(perfect=True)
        err = self._errors(history, est)

        assert err["depth"] < 0.01, f"depth RMSE {err['depth']:.4f} m"
        assert err["yaw"] < 0.2, f"yaw RMSE {err['yaw']:.3f}°"
        assert err["pitch"] < 0.2, f"pitch RMSE {err['pitch']:.3f}°"
        assert err["roll"] < 0.2, f"roll RMSE {err['roll']:.3f}°"
