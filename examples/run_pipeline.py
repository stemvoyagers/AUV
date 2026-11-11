"""End-to-end pipeline: trajectory → noisy sensors → ESKF → compare vs truth.

This exercises the full loop requested in the learning plan:

    Trajectory ──► Ground Truth ──► Noisy Sensors ──► ESKF ──► Compare vs truth

Prints navigation error statistics and saves comparison plots.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from auv_simulator.simulator import Simulator
from auv_simulator.sensors import Accelerometer, DVL, Gyroscope, Magnetometer, PressureSensor
from auv_simulator.navigation import ESKF
from auv_simulator.core.frames import pressure_to_depth
from auv_simulator.visualization import plot_trajectory_3d   # reuse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)
EXPORT_DIR = ROOT / "exports"
EXPORT_DIR.mkdir(exist_ok=True)


def build_simulator(seed: int = 7) -> Simulator:
    """Reference sensor suite — the same values documented in the README."""
    sim = Simulator(dt_physics=0.005, seed=seed)

    sim.add_sensor(Accelerometer(noise_std=0.01, bias=0.005, noise_density=1e-4), frequency_hz=100)
    sim.add_sensor(Gyroscope(noise_std=0.002, bias=0.001, noise_density=1e-5),     frequency_hz=100)
    sim.add_sensor(Magnetometer(noise_std=0.1),                                    frequency_hz=50)
    sim.add_sensor(PressureSensor(noise_std=5.0, bias=1.0),                        frequency_hz=10)
    # DVL: measures body-frame velocity over ground → observes horizontal motion.
    sim.add_sensor(DVL(noise_std=0.01, bias=0.005, noise_density=1e-4),            frequency_hz=5)
    return sim


def export_for_cpp(sim: Simulator) -> None:
    sim.export_ground_truth_csv(str(EXPORT_DIR / "ground_truth.csv"))
    for name in sim._sensors:
        sim.export_sensor_csv(name, str(EXPORT_DIR / f"{name}.csv"))
    sim.export_json(str(EXPORT_DIR / "full_simulation.json"))
    print(f"  → exported to {EXPORT_DIR}/")


def run_pipeline() -> None:
    sim = build_simulator()
    history = sim.run()
    print(f"Simulation done — {len(history)} ticks, {sim.trajectory.duration} s")

    # Export
    export_for_cpp(sim)

    # Run ESKF
    print("Running ESKF navigation filter...")
    eskf = ESKF()
    est = eskf.run(history)

    # Align times — estimate has one entry per history tick
    t = np.array([h.t for h in history])
    gt_pos = np.vstack([h.state.position for h in history])
    est_pos = np.vstack([e.p for e in est])
    gt_ypr = np.vstack([h.state.yaw_pitch_roll for h in history])
    est_ypr = np.vstack([e.yaw_pitch_roll for e in est])

    # Errors
    err = est_pos - gt_pos
    err_euclidean = np.linalg.norm(err, axis=1)

    # Late-time errors (filter converged)
    late = t > 20.0
    pos_rmse_late = float(np.sqrt(np.mean(err_euclidean[late] ** 2)))

    depth_rmse = float(np.sqrt(np.mean(err[late, 2] ** 2)))
    yaw_err = np.degrees(np.arctan2(np.sin(est_ypr[:, 0] - gt_ypr[:, 0]),
                                    np.cos(est_ypr[:, 0] - gt_ypr[:, 0])))
    pitch_err = np.degrees(est_ypr[:, 1] - gt_ypr[:, 1])
    roll_err = np.degrees(est_ypr[:, 2] - gt_ypr[:, 2])
    yaw_rmse = float(np.sqrt(np.mean(yaw_err[late] ** 2)))
    pitch_rmse = float(np.sqrt(np.mean(pitch_err[late] ** 2)))
    roll_rmse = float(np.sqrt(np.mean(roll_err[late] ** 2)))

    print("\nNavigation accuracy (ESKF, t>20s)")
    print("-" * 50)
    print(f"  depth RMSE  : {depth_rmse:8.4f} m")
    print(f"  yaw   RMSE  : {yaw_rmse:8.4f} deg")
    print(f"  pitch RMSE  : {pitch_rmse:8.4f} deg")
    print(f"  roll  RMSE  : {roll_rmse:8.4f} deg")
    print(f"  position    : {pos_rmse_late:8.3f} m  (DVL-aided)")

    # Plot 1: 3D comparison
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt_pos[:, 0],  gt_pos[:, 1],  -gt_pos[:, 2],  color="navy", lw=2, label="Ground truth")
    ax.plot(est_pos[:, 0], est_pos[:, 1], -est_pos[:, 2], color="crimson", lw=1.5,
            ls="--", label="ESKF estimate")
    ax.set_xlabel("North (m)"); ax.set_ylabel("East (m)"); ax.set_zlabel("Depth (m, + down)")
    ax.set_title("Ground truth vs ESKF navigation estimate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(OUT_DIR / "nav_compare_3d.png"), dpi=150)
    plt.close(fig)

    # Plot 2: position, depth and attitude error over time
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(t, err[:, 0], label="North err", color="tab:blue")
    axes[0].plot(t, err[:, 1], label="East err",  color="tab:orange")
    axes[0].set_ylabel("Error (m)"); axes[0].set_title("Horizontal position error (DVL-aided)"); axes[0].legend()

    axes[1].plot(t, err[:, 2], color="tab:green")
    axes[1].set_ylabel("Depth err (m)"); axes[1].set_title("Depth error (pressure-aided)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, yaw_err, label="Yaw", color="tab:red")
    axes[2].plot(t, pitch_err, label="Pitch", color="tab:purple")
    axes[2].plot(t, roll_err, label="Roll", color="tab:brown")
    axes[2].set_xlabel("Time (s)"); axes[2].set_ylabel("Error (deg)")
    axes[2].set_title("Attitude error (accel tilt + magnetometer aided)"); axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(OUT_DIR / "nav_error.png"), dpi=150)
    plt.close(fig)

    print(f"\nPlots → {OUT_DIR}/nav_compare_3d.png, nav_error.png")


if __name__ == "__main__":
    run_pipeline()
