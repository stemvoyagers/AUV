"""Shared pytest fixtures/helpers.

``build_reference_sim`` must stay in sync with ``examples/run_pipeline.py``:
it is the documented reference sensor suite (README → "Python API").
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auv_simulator.sensors import (          # noqa: E402
    Accelerometer, DVL, Gyroscope, Magnetometer, PressureSensor,
)
from auv_simulator.simulator import Simulator  # noqa: E402


def build_reference_sim(seed: int = 7,
                        accel_dropout: float = 0.0,
                        gyro_dropout: float = 0.0,
                        mag_dropout: float = 0.0,
                        pressure_dropout: float = 0.0,
                        dvl_dropout: float = 0.0) -> Simulator:
    """Reference 5-sensor AUV suite used by the pipeline, tests and goldens."""
    sim = Simulator(dt_physics=0.005, seed=seed)

    sim.add_sensor(Accelerometer(noise_std=0.01, bias=0.005, noise_density=1e-4,
                                 dropout_prob=accel_dropout), frequency_hz=100)
    sim.add_sensor(Gyroscope(noise_std=0.002, bias=0.001, noise_density=1e-5,
                             dropout_prob=gyro_dropout), frequency_hz=100)
    sim.add_sensor(Magnetometer(noise_std=0.1,
                                dropout_prob=mag_dropout), frequency_hz=50)
    sim.add_sensor(PressureSensor(noise_std=5.0, bias=1.0,
                                  dropout_prob=pressure_dropout), frequency_hz=10)
    sim.add_sensor(DVL(noise_std=0.01, bias=0.005, noise_density=1e-4,
                       dropout_prob=dvl_dropout), frequency_hz=5)
    return sim


def estimate_errors(history) -> dict:
    """Run the ESKF over ``history`` and return position/attitude error arrays."""
    from auv_simulator.navigation import ESKF

    est = ESKF().run(history)
    truth_pos = np.vstack([h.state.position for h in history])
    est_pos = np.vstack([e.p for e in est])
    truth_ypr = np.vstack([h.state.yaw_pitch_roll for h in history])
    est_ypr = np.vstack([e.yaw_pitch_roll for e in est])

    yaw_err = np.arctan2(np.sin(est_ypr[:, 0] - truth_ypr[:, 0]),
                         np.cos(est_ypr[:, 0] - truth_ypr[:, 0]))

    return {
        "t": np.array([h.t for h in history]),
        "position": est_pos - truth_pos,
        "yaw": yaw_err,
        "pitch": est_ypr[:, 1] - truth_ypr[:, 1],
        "roll": est_ypr[:, 2] - truth_ypr[:, 2],
    }
