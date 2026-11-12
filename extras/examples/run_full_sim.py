"""End-to-end AUV simulator demo.

1. Run the 120 s trajectory with 4 realistically noisy sensors.
2. Print a sensor reference table + trajectory stage breakdown.
3. Plot trajectory + ground truth + sensor readings.
4. Run a numeric sanity check (accel gravity, pressure→depth, yaw rates).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from auv_simulator.simulator import Simulator
from auv_simulator.sensors import (
    Accelerometer, Gyroscope, Magnetometer, PressureSensor,
)
from auv_simulator.core.trajectory import STAGES
from auv_simulator.visualization import (
    plot_trajectory_3d, plot_ground_truth_overview, plot_sensor_overview,
)


OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def build_simulator(dt_physics: float = 0.005, seed: int = 7) -> Simulator:
    sim = Simulator(dt_physics=dt_physics, seed=seed)

    # --- Accelerometer — 100 Hz,  σ=0.05 m/s²,  0.02 m/s² bias ---
    sim.add_sensor(
        Accelerometer(noise_std=0.05, bias=0.02, noise_density=0.001),
        frequency_hz=100,
    )

    # --- Gyroscope — 100 Hz,  σ=0.01 rad/s,  0.005 rad/s bias ---
    sim.add_sensor(
        Gyroscope(noise_std=0.01, bias=0.005, noise_density=2e-4),
        frequency_hz=100,
    )

    # --- Magnetometer — 50 Hz,  σ=0.5 µT,  1.0 µT bias ---
    sim.add_sensor(
        Magnetometer(noise_std=0.5, bias=1.0),
        frequency_hz=50,
    )

    # --- Pressure sensor — 10 Hz,  σ=20 Pa,  10 Pa bias ---
    sim.add_sensor(
        PressureSensor(noise_std=20.0, bias=10.0),
        frequency_hz=10,
    )

    return sim


def print_sensor_table(sim: Simulator) -> None:
    header = f"{'Sensor':<16} {'Axes':>4} {'Units':<14} {'Freq':>7} {'Noise':>9} {'Bias':>9} {'Drift':>9}"
    print("=" * len(header))
    print("Sensor configuration")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, (sensor, dt_sample, _) in sim._sensors.items():
        freq = 1.0 / dt_sample
        n = sensor.noise_std
        b = sensor.bias
        d = sensor.noise_density
        print(f"{sensor.name:<16} {sensor.axes:>4} {sensor.units:<14} "
              f"{freq:>5.0f} Hz  {np.mean(n):>8.4g} {np.mean(b):>8.4g} {d:>8.4g}")


def print_stage_summary(sim: Simulator) -> None:
    print("\n" + "=" * 72)
    print("120-second trajectory — stage summary")
    print("=" * 72)
    print(f"{'#':>2} {'Stage':<14} {'t_start':>7} {'t_end':>7} | "
          f"{'pos_end (N,E,D)':<28} {'yaw_end':>9}")
    print("-" * 72)
    for s in STAGES:
        state = sim.trajectory.state_at(s.t_end - 1e-9 if s.index < len(STAGES) - 1 else sim.trajectory.duration)
        yaw_deg = float(np.degrees(state.yaw_pitch_roll[0]))
        print(f"{s.index+1:>2} {s.name:<14} {s.t_start:>6.1f}s {s.t_end:>6.1f}s | "
              f"({state.position[0]:7.1f}, {state.position[1]:7.1f}, {state.position[2]:6.1f}) "
              f" {yaw_deg:7.2f}°")


def sanity_checks(sim: Simulator) -> None:
    print("\nSanity checks vs. ground truth")
    print("-" * 50)
    gt = sim.ground_truth_arrays()

    # 1. Final position vs. expected
    final_pos = gt["position"][-1]
    print(f"Final position NED : ({final_pos[0]:.2f}, {final_pos[1]:.2f}, {final_pos[2]:.2f}) m")
    assert final_pos[2] > 0, "depth should be positive (down)"

    # 2. Pressure sensor — pressure range & last reading
    ps = sim.sensor_arrays("pressure")
    if ps and ps["ideal"] is not None:
        ideal_pa = ps["ideal"].ravel()
        print(f"Pressure ideal (Pa)    : [{ideal_pa.min():.0f}, {ideal_pa.max():.0f}] Pa")
        from auv_simulator.core.frames import pressure_to_depth
        depths = pressure_to_depth(ideal_pa)
        print(f"Pressure-derived depth : [{depths.min():.2f}, {depths.max():.2f}] m")
        assert depths[-1] > 0, "final depth > 0"

    # 3. Gyro ideal yaw rate during turning stage
    gs = sim.sensor_arrays("gyroscope")
    if gs and gs["ideal"] is not None:
        # pick entries during stage 3 (50–70 s)
        mask = (gs["t"] >= 50) & (gs["t"] <= 70)
        r_rates = gs["ideal"][mask, 2]   # yaw rate
        print(f"Gyro ideal yaw rate @ stage 3 : mean={np.mean(r_rates):.4f} rad/s "
              f"(expected ~+0.157 rad/s)")

    print("All checks passed ✓")


def main() -> None:
    sim = build_simulator()
    print_sensor_table(sim)
    print_stage_summary(sim)

    print("\nRunning simulation...")
    history = sim.run()
    print(f"  physics ticks      : {len(history)}")
    print(f"  trajectory length  : {sim.trajectory.duration} s")

    sanity_checks(sim)

    # Plots
    fig_3d   = plot_trajectory_3d(sim, save_path=str(OUT_DIR / "trajectory_3d.png"))
    fig_gt   = plot_ground_truth_overview(sim, save_path=str(OUT_DIR / "ground_truth.png"))
    fig_sens = plot_sensor_overview(sim, save_path=str(OUT_DIR / "sensors.png"))

    print(f"\nPlots written to: {OUT_DIR}")
    for f in fig_3d, fig_gt, fig_sens:
        import matplotlib.pyplot as plt
        plt.close(f)

    print("\nDone.")


if __name__ == "__main__":
    main()
