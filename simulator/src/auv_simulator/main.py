from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .sensors import Accelerometer, Gyroscope, Magnetometer, PressureSensor
from .simulator import Simulator
from .trajectory import DEFAULT_INITIAL_DEPTH, Trajectory

PACKAGE_DIR = Path(__file__).resolve().parent
SIMULATOR_DIR = PACKAGE_DIR.parent.parent
REPO_ROOT = SIMULATOR_DIR.parent

DEFAULT_EXPORT_DIR = REPO_ROOT / "exports"
DEFAULT_PLOT_DIR = REPO_ROOT / "outputs"

SENSOR_SUITE = (
    ("accelerometer", Accelerometer, dict(noise_std=0.01, bias=0.005, noise_density=1e-4), 100.0),
    ("gyroscope", Gyroscope, dict(noise_std=0.002, bias=0.001, noise_density=1e-5), 100.0),
    ("magnetometer", Magnetometer, dict(noise_std=0.1), 50.0),
    ("pressure", PressureSensor, dict(noise_std=5.0, bias=1.0), 10.0),
)


def build_reference_simulator(seed: int = 42,
                              dt_physics: float = 0.005,
                              initial_depth: float = DEFAULT_INITIAL_DEPTH) -> Simulator:
    trajectory = Trajectory(initial_depth=initial_depth)
    sim = Simulator(trajectory=trajectory, dt_physics=dt_physics, seed=seed)
    for _, sensor_cls, params, frequency_hz in SENSOR_SUITE:
        sim.add_sensor(sensor_cls(**params), frequency_hz=frequency_hz)
    return sim


def export_dataset(sim: Simulator, export_dir: Path) -> list[Path]:
    export_dir.mkdir(parents=True, exist_ok=True)
    written = []

    truth_path = export_dir / "ground_truth.csv"
    sim.export_ground_truth_csv(truth_path)
    written.append(truth_path)

    for name in sim.sensor_names:
        path = export_dir / f"{name}.csv"
        sim.export_sensor_csv(name, path)
        written.append(path)

    return written


def plot_trajectory(sim: Simulator, save_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt = sim.ground_truth_arrays()
    pos = gt["position"]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pos[:, 0], pos[:, 1], -pos[:, 2], color="navy", lw=2, label="Trajectory")

    for stage in sim.trajectory.stages():
        state = sim.trajectory.state_at(stage.t_start)
        ax.scatter(state.position[0], state.position[1], -state.position[2],
                   s=55, label=f"{stage.t_start:0.0f}s  {stage.name}")

    ax.set_xlabel("North (m)")
    ax.set_ylabel("East (m)")
    ax.set_zlabel("Depth (m, positive down)")
    ax.set_title("AUV trajectory — 120 s reference scenario (NED)")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_time_series(sim: Simulator, save_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt = sim.ground_truth_arrays()
    t = gt["t"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(t, gt["position"][:, 0], label="North", color="tab:blue")
    ax1.plot(t, gt["position"][:, 1], label="East", color="tab:orange")
    ax1.plot(t, gt["position"][:, 2], label="Depth", color="tab:green")
    ax1.set_ylabel("Position (m)")
    ax1.set_title("Ground-truth position and orientation")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(t, np.degrees(gt["yaw"]), label="yaw", color="tab:red")
    ax2.plot(t, np.degrees(gt["pitch"]), label="pitch", color="tab:purple")
    ax2.plot(t, np.degrees(gt["roll"]), label="roll", color="tab:brown")
    ax2.set_ylabel("Angle (deg)")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def summarise(sim: Simulator) -> str:
    truth = sim.ground_truth_arrays()
    final = sim.history[-1].state
    initial = sim.history[0].state
    peak_speed = float(np.linalg.norm(truth["velocity"], axis=1).max())

    lines = [
        f"ticks              : {len(sim.history)}  (dt = {sim.dt_physics} s)",
        f"duration           : {sim.trajectory.duration:.0f} s",
        f"initial state      : depth {initial.depth:.2f} m, "
        f"heading {initial.heading_deg:.1f} deg",
        f"final state        : depth {final.depth:.2f} m, "
        f"heading {final.heading_deg:.1f} deg, "
        f"north {final.north:.2f} m, east {final.east:.2f} m",
        f"peak speed         : {peak_speed:.3f} m/s",
    ]
    for name in sim.sensor_names:
        arr = sim.sensor_arrays(name)
        if arr is not None:
            lines.append(f"{name:<19}: {len(arr['t'])} samples")
    return "\n".join(lines)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the synthetic dataset and inspect the trajectory.")
    parser.add_argument("--dt", type=float, default=0.005,
                        help="physics time step in seconds (default: 0.005)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for sensor noise (default: 42)")
    parser.add_argument("--initial-depth", type=float, default=DEFAULT_INITIAL_DEPTH,
                        help="starting depth in metres (default: 20.0)")
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR,
                        help="where the CSV dataset is written")
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR,
                        help="where the figures are written")
    parser.add_argument("--no-plots", action="store_true",
                        help="skip figure generation")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.dt <= 0:
        raise SystemExit("--dt must be positive")

    sim = build_reference_simulator(seed=args.seed,
                                    dt_physics=args.dt,
                                    initial_depth=args.initial_depth)
    sim.run()

    files = export_dataset(sim, args.export_dir)
    print(summarise(sim))
    print("\nexported:")
    for path in files:
        print(f"  {path}")

    if not args.no_plots:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        plot_trajectory(sim, args.plot_dir / "trajectory_3d.png")
        plot_time_series(sim, args.plot_dir / "ground_truth.png")
        print(f"\nplots:\n  {args.plot_dir / 'trajectory_3d.png'}"
              f"\n  {args.plot_dir / 'ground_truth.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
