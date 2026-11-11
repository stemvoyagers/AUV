"""Replay a CSV sensor export through the Python ESKF and check accuracy.

Why this exists
---------------
``exports/*.csv`` is the frozen contract between the Python truth model and the
C++ implementation.  Both sides consume the *same* sensor streams, so both can
be scored against the *same* ground truth.  This script is the reference
implementation of that scoring.

Usage
-----
    python tools/validate_export.py                     # default exports/
    python tools/validate_export.py --exports exports
    python tools/validate_export.py --exports exports --quiet

Exit code 0 → every required check passed.  1 → at least one failed.
The process is deterministic: the CSVs contain finished sensor streams, so the
same inputs always produce the same numbers.

Contract
--------
``ground_truth.csv``      t, north, east, depth, v_n, v_e, v_d, yaw_rad, pitch_rad, roll_rad
``accelerometer.csv``     t, accelerometer_0..2   (body specific force, m/s²)
``gyroscope.csv``         t, gyroscope_0..2       (body rate, rad/s)
``magnetometer.csv``      t, magnetometer_0..2    (body field, µT)
``pressure.csv``          t, pressure_0           (Pa)
``dvl.csv``               t, dvl_0..2             (body velocity, m/s)  [optional]

If ``dvl.csv`` is absent, horizontal position is unobservable by design and the
horizontal checks are reported as informational instead of failing.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auv_simulator.navigation import ESKF                      # noqa: E402
from auv_simulator.sensors.base import SensorReading            # noqa: E402
from auv_simulator.simulator.simulator import LogEntry          # noqa: E402

# Sensor columns we expect to find, in the order they are applied
VECTOR_SENSORS = ("accelerometer", "gyroscope", "magnetometer", "dvl")
SCALAR_SENSORS = ("pressure",)

# Convergence window: the filter starts with a deliberately loose prior, so
# early samples are dominated by initial-condition error, not filter quality.
SETTLE_TIME = 20.0

# Tolerances (one-sigma RMSE over the settled window).
LIMITS = {
    "depth_rmse":   0.05,    # m        pressure-aided
    "north_rmse":   2.0,     # m        DVL-aided
    "east_rmse":    2.0,     # m        DVL-aided
    "vn_rmse":      0.20,    # m/s      DVL-aided
    "ve_rmse":      0.20,    # m/s      DVL-aided
    "yaw_rmse_deg": 1.0,     # deg      magnetometer + gyro
    "pitch_rmse_deg": 1.0,   # deg      tilt aid
    "roll_rmse_deg": 1.0,    # deg      tilt aid
}


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------
def _load_csv(path: Path) -> dict:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        cols = [c for c in (reader.fieldnames or []) if c != "t"]

    if not rows:
        raise ValueError(f"{path} contains no data rows")

    t = np.array([float(r["t"]) for r in rows], dtype=float)
    data = np.array([[float(r[c]) for c in cols] for r in rows], dtype=float)
    return {"t": t, "data": data, "cols": cols}


def load_export(export_dir: Path) -> Dict[str, dict]:
    """Load every known sensor CSV plus ground truth from ``export_dir``."""
    out: Dict[str, dict] = {}

    gt = export_dir / "ground_truth.csv"
    if not gt.exists():
        raise FileNotFoundError(f"missing ground truth: {gt}")
    out["ground_truth"] = _load_csv(gt)

    for name in VECTOR_SENSORS + SCALAR_SENSORS:
        p = export_dir / f"{name}.csv"
        if p.exists():
            out[name] = _load_csv(p)
    return out


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
def build_history(export: Dict[str, dict]) -> list:
    """Merge per-sensor CSVs into a single time-ordered sensor log.

    Each timeline entry holds only the sensors that produced a sample at that
    instant, mirroring exactly what the simulator logs.
    """
    events: Dict[float, dict] = {}
    for name in VECTOR_SENSORS + SCALAR_SENSORS:
        if name not in export:
            continue
        s = export[name]
        for t_i, d_i in zip(s["t"], s["data"]):
            key = round(float(t_i), 9)
            events.setdefault(key, {})[name] = SensorReading(
                t=float(t_i), data=np.asarray(d_i, dtype=float), name=name
            )

    return [LogEntry(t=t, state=None, readings=events[t]) for t in sorted(events)]


def replay(export: Dict[str, dict]):
    """Run the reference ESKF over an export. Returns (t, est_pos, est_ypr, est_vel)."""
    history = build_history(export)
    eskf = ESKF()
    est = eskf.run(history)

    t = np.array([h.t for h in history], dtype=float)
    pos = np.vstack([e.p for e in est])
    vel = np.vstack([e.v for e in est])
    ypr = np.vstack([e.yaw_pitch_roll for e in est])
    return t, pos, vel, ypr


def _interp_gt(gt: dict, t: np.ndarray) -> dict:
    """Interpolate ground truth columns onto the estimate timeline."""
    cols = gt["cols"]
    idx = {c: i for i, c in enumerate(cols)}
    t_gt = gt["t"]

    def col(name: str) -> np.ndarray:
        return np.interp(t, t_gt, gt["data"][:, idx[name]])

    pos = np.column_stack([col("north"), col("east"), col("depth")])
    vel = np.column_stack([col("v_n"), col("v_e"), col("v_d")])
    ypr = np.column_stack([col("yaw_rad"), col("pitch_rad"), col("roll_rad")])
    return {"t": t, "position": pos, "velocity": vel, "ypr": ypr}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def compute_metrics(export: Dict[str, dict]) -> dict:
    t, pos, vel, ypr = replay(export)
    gt = _interp_gt(export["ground_truth"], t)

    late = t > SETTLE_TIME
    if not np.any(late):
        late = np.ones_like(t, dtype=bool)

    dp = pos - gt["position"]
    dv = vel - gt["velocity"]

    yaw_err = np.arctan2(np.sin(ypr[:, 0] - gt["ypr"][:, 0]),
                         np.cos(ypr[:, 0] - gt["ypr"][:, 0]))
    pitch_err = ypr[:, 1] - gt["ypr"][:, 1]
    roll_err = ypr[:, 2] - gt["ypr"][:, 2]

    m = {
        "samples": int(len(t)),
        "has_dvl": "dvl" in export,
        "north_rmse": _rmse(dp[late, 0]),
        "east_rmse": _rmse(dp[late, 1]),
        "depth_rmse": _rmse(dp[late, 2]),
        "vn_rmse": _rmse(dv[late, 0]),
        "ve_rmse": _rmse(dv[late, 1]),
        "vd_rmse": _rmse(dv[late, 2]),
        "yaw_rmse_deg": float(np.degrees(_rmse(yaw_err[late]))),
        "pitch_rmse_deg": float(np.degrees(_rmse(pitch_err[late]))),
        "roll_rmse_deg": float(np.degrees(_rmse(roll_err[late]))),
        "horizontal_max_m": float(np.max(np.linalg.norm(dp[late, :2], axis=1))),
    }
    return m


def check_metrics(m: dict) -> list[dict]:
    """Compare metrics against LIMITS. Horizontal checks are informational if no DVL."""
    dvl_dependent = {"north_rmse", "east_rmse", "vn_rmse", "ve_rmse"}
    checks = []
    for key, limit in LIMITS.items():
        value = m[key]
        required = m["has_dvl"] or key not in dvl_dependent
        checks.append({
            "name": key,
            "value": value,
            "limit": limit,
            "required": required,
            "passed": value <= limit,
        })
    return checks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _fmt_row(c: dict) -> str:
    status = "PASS" if c["passed"] else ("FAIL" if c["required"] else "INFO")
    return f"  [{status}] {c['name']:<16} {c['value']:10.5f}   (limit {c['limit']:g})"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exports", default=str(ROOT / "exports"),
                    help="directory holding the CSV export (default: exports/)")
    ap.add_argument("--quiet", action="store_true", help="only print the summary")
    args = ap.parse_args(argv)

    export_dir = Path(args.exports)
    try:
        export = load_export(export_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    m = compute_metrics(export)
    checks = check_metrics(m)

    if not args.quiet:
        found = ", ".join(sorted(k for k in export if k != "ground_truth"))
        print(f"Export  : {export_dir}")
        print(f"Sensors : {found}")
        print(f"Samples : {m['samples']}  (settled window: t > {SETTLE_TIME:g} s)")
        print("\nMetrics")
        print("-" * 56)
        for c in checks:
            print(_fmt_row(c))
        print(f"  [INFO] horizontal max   {m['horizontal_max_m']:10.3f}   (m)")

    failed = [c for c in checks if c["required"] and not c["passed"]]
    print()
    if failed:
        print(f"RESULT: FAIL — {len(failed)} check(s) outside tolerance")
        for c in failed:
            print(f"  {c['name']}: {c['value']:.5f} > {c['limit']:g}")
        return 1

    print("RESULT: PASS — all required checks within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
