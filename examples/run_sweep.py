"""Sweep sensor noise/bias/drift through the ESKF to stress-test navigation.

For every (accel_noise, gyro_noise, mag_noise, press_noise, bias_scale)
combination we:
    1. Run the simulator.
    2. Run the ESKF.
    3. Record depth / attitude / position RMSE (t > 20 s).
    4. Print a sorted table + write results CSV.

The suite includes a DVL, so position is observable and the ``pos`` column is a
meaningful accuracy metric rather than an unbounded drift number.
"""

from __future__ import annotations

import sys
from pathlib import Path
from itertools import product

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from auv_simulator.simulator import Simulator
from auv_simulator.sensors import Accelerometer, DVL, Gyroscope, Magnetometer, PressureSensor
from auv_simulator.navigation import ESKF

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# Sweep grids
ACCEL_NOISES  = [0.01, 0.05, 0.10, 0.20]      # m/s²
GYRO_NOISES   = [0.001, 0.005, 0.01, 0.02]    # rad/s
MAG_NOISES    = [0.25, 0.50, 1.0, 2.0]        # µT
PRESS_NOISES  = [5.0, 20.0, 50.0, 100.0]      # Pa
BIAS_SCALES   = [0.0, 0.25, 0.50, 1.00]       # bias / σ


def run_one(accel_noise: float, gyro_noise: float, mag_noise: float,
            press_noise: float, bias_scale: float = 0.25,
            seed: int = 7) -> dict:
    sim = Simulator(dt_physics=0.005, seed=seed)

    # Biases = noise * bias_scale (a fraction of σ, representing a real sensor)
    sim.add_sensor(Accelerometer(noise_std=accel_noise,  bias=accel_noise  * bias_scale, noise_density=accel_noise * 0.01),  100)
    sim.add_sensor(Gyroscope(noise_std=gyro_noise,     bias=gyro_noise   * bias_scale, noise_density=gyro_noise  * 0.01),  100)
    sim.add_sensor(Magnetometer(noise_std=mag_noise,   bias=mag_noise    * bias_scale),                                   50)
    sim.add_sensor(PressureSensor(noise_std=press_noise, bias=press_noise * bias_scale),                                    10)
    sim.add_sensor(DVL(noise_std=0.01, bias=0.005, noise_density=1e-4),                                                     5)

    history = sim.run()
    eskf = ESKF()
    est = eskf.run(history)

    t = np.array([h.t for h in history])
    mask = t > 20.0

    gt_pos = np.vstack([h.state.position for h in history])
    est_pos = np.vstack([e.p for e in est])
    err = np.linalg.norm(est_pos - gt_pos, axis=1)

    gt_ypr = np.vstack([h.state.yaw_pitch_roll for h in history])
    es_ypr = np.vstack([e.yaw_pitch_roll for e in est])
    yaw_d = np.arctan2(np.sin(es_ypr[:, 0] - gt_ypr[:, 0]),
                       np.cos(es_ypr[:, 0] - gt_ypr[:, 0]))

    rmse = lambda v: float(np.sqrt(np.mean(v[mask] ** 2)))
    return {
        "accel_noise": accel_noise, "gyro_noise": gyro_noise,
        "mag_noise": mag_noise, "press_noise": press_noise,
        "bias_scale": bias_scale,
        "rmse_depth": rmse((est_pos - gt_pos)[:, 2]),
        "rmse_yaw": float(np.degrees(rmse(yaw_d))),
        "rmse_pitch": float(np.degrees(rmse(es_ypr[:, 1] - gt_ypr[:, 1]))),
        "rmse_roll": float(np.degrees(rmse(es_ypr[:, 2] - gt_ypr[:, 2]))),
        "rmse_pos": rmse(err),
    }


def main() -> None:
    # Full combinatorial sweep (4 × 4 × 4 × 4 = 256 runs, each ~1s) → too much.
    # We keep the other three at a "nominal" value and vary one at a time.
    rows = []

    nominal = {"accel_noise": 0.05, "gyro_noise": 0.01, "mag_noise": 0.50, "press_noise": 20.0}

    # 1) Varying accelerometer noise
    print("Sweep 1 — Accelerometer noise")
    for an in ACCEL_NOISES:
        rows.append(run_one(an, nominal["gyro_noise"], nominal["mag_noise"], nominal["press_noise"]))
    _print_rows(rows[-len(ACCEL_NOISES):])

    # 2) Varying gyro noise
    print("\nSweep 2 — Gyro noise")
    for gn in GYRO_NOISES:
        rows.append(run_one(nominal["accel_noise"], gn, nominal["mag_noise"], nominal["press_noise"]))
    _print_rows(rows[-len(GYRO_NOISES):])

    # 3) Varying magnetometer noise
    print("\nSweep 3 — Magnetometer noise")
    for mn in MAG_NOISES:
        rows.append(run_one(nominal["accel_noise"], nominal["gyro_noise"], mn, nominal["press_noise"]))
    _print_rows(rows[-len(MAG_NOISES):])

    # 4) Varying pressure noise
    print("\nSweep 4 — Pressure noise")
    for pn in PRESS_NOISES:
        rows.append(run_one(nominal["accel_noise"], nominal["gyro_noise"], nominal["mag_noise"], pn))
    _print_rows(rows[-len(PRESS_NOISES):])

    # 5) Varying the bias level (noise held nominal) — "how much bias is tolerable?"
    print("\nSweep 5 — Sensor bias magnitude (bias = bias_scale × σ)")
    for bs in BIAS_SCALES:
        rows.append(run_one(nominal["accel_noise"], nominal["gyro_noise"],
                            nominal["mag_noise"], nominal["press_noise"], bias_scale=bs))
    _print_rows(rows[-len(BIAS_SCALES):])

    # Save full results
    import csv
    out_path = OUT / "sweep_results.csv"
    fields = ["accel_noise", "gyro_noise", "mag_noise", "press_noise", "bias_scale",
              "rmse_depth", "rmse_yaw", "rmse_pitch", "rmse_roll", "rmse_pos"]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nFull results → {out_path}")


def _print_rows(rows: list[dict]) -> None:
    header = (f"{'accel':>7} {'gyro':>7} {'mag':>6} {'press':>6} {'bias':>6} | "
              f"{'depth':>8} {'yaw':>7} {'pitch':>7} {'roll':>7} {'pos':>9}")
    print(header); print("-" * len(header))
    for r in rows:
        print(f"{r['accel_noise']:>7.3f} {r['gyro_noise']:>7.4f} "
              f"{r['mag_noise']:>6.2f} {r['press_noise']:>6.1f} {r['bias_scale']:>6.2f} | "
              f"{r['rmse_depth']:>7.4f}m {r['rmse_yaw']:>6.3f}° "
              f"{r['rmse_pitch']:>6.3f}° {r['rmse_roll']:>6.3f}° {r['rmse_pos']:>8.1f}m")


if __name__ == "__main__":
    main()
