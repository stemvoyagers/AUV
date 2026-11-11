"""Matplotlib plotting helpers for simulation validation."""

from __future__ import annotations

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend so this works head-less
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  (registers 3D projection)
except ImportError as exc:   # pragma: no cover
    raise ImportError("matplotlib is required for visualization") from exc

from ..simulator.simulator import Simulator


def plot_trajectory_3d(sim: Simulator, save_path: str | None = None) -> plt.Figure:
    """3D perspective view of the NED trajectory."""
    gt = sim.ground_truth_arrays()
    pos = gt["position"]   # columns: north, east, depth

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pos[:, 0], pos[:, 1], -pos[:, 2], color="navy", lw=2, label="Trajectory")

    # stage markers
    stages = sim.trajectory.stages()
    for s in stages:
        state = sim.trajectory.state_at(s.t_start)
        ax.scatter(state.position[0], state.position[1], -state.position[2],
                   s=60, label=f"{s.t_start:0.0f}s: {s.name}")

    ax.set_xlabel("North (m)")
    ax.set_ylabel("East (m)")
    ax.set_zlabel("Depth (m, positive down)")
    ax.set_title("AUV Simulator — 120 s Trajectory (NED)")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_ground_truth_overview(sim: Simulator, save_path: str | None = None) -> plt.Figure:
    """2-panel time-series: (top) NED position, (bottom) Euler angles."""
    gt = sim.ground_truth_arrays()
    t = gt["t"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(t, gt["position"][:, 0], label="North", color="tab:blue")
    ax1.plot(t, gt["position"][:, 1], label="East",  color="tab:orange")
    ax1.plot(t, gt["position"][:, 2], label="Depth", color="tab:green")
    ax1.set_ylabel("Position (m)")
    ax1.legend()
    ax1.set_title("Ground-truth position and orientation")

    ax2.plot(t, np.degrees(gt["yaw"]),   label="yaw",   color="tab:red")
    ax2.plot(t, np.degrees(gt["pitch"]), label="pitch", color="tab:purple")
    ax2.plot(t, np.degrees(gt["roll"]),  label="roll",  color="tab:brown")
    ax2.set_ylabel("Angle (deg)")
    ax2.set_xlabel("Time (s)")
    ax2.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_sensor_overview(sim: Simulator, save_path: str | None = None) -> plt.Figure:
    """Grid of subplots — one per sensor, noisy vs ideal."""
    sensor_names = list(sim._sensors.keys())
    n = len(sensor_names)
    if n == 0:
        raise ValueError("No sensors registered with simulator")

    fig, axes = plt.subplots(n, 1, figsize=(10, 3.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, sensor_names):
        data = sim.sensor_arrays(name)
        if data is None:
            continue

        t, d = data["t"], data["data"]
        ideal = data["ideal"]

        if d.ndim == 1 or d.shape[1] == 1:
            ax.plot(t, d.ravel(), label=f"{name} noisy", color="tab:blue", lw=1.2)
            if ideal is not None:
                ax.plot(t, ideal.ravel(), label=f"{name} ideal", color="black", lw=1.0)
        else:
            labels = ["x", "y", "z"] if d.shape[1] == 3 else [str(i) for i in range(d.shape[1])]
            for i, lbl in enumerate(labels):
                ax.plot(t, d[:, i], label=f"{name} noisy {lbl}", lw=1.2)
                if ideal is not None:
                    ax.plot(t, ideal[:, i], label=f"{name} ideal  {lbl}",
                            color="black", lw=0.8, ls="--")

        ax.set_title(f"{name}  ({len(d)} samples)")
        ax.set_ylabel("reading")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Sensor readings — noisy vs ideal", y=1.01)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
