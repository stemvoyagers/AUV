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

    # Score an external (e.g. C++) navigation output instead of replaying
    python tools/validate_export.py --estimates path/to/cpp_estimates.csv

    # Emit a reference estimates file the C++ side can imitate
    python tools/validate_export.py --write-sample exports/estimates_reference.csv

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

External estimates (``--estimates``) use the **same columns as
``ground_truth.csv``**, so the C++ filter has one schema to match and both
implementations are scored by identical code:

    t, north, east, depth, v_n, v_e, v_d, yaw_rad, pitch_rad, roll_rad

The estimates timeline need not match the truth timeline; it is interpolated
onto it. Yaw is unwrapped before interpolation, so a filter that reports
`-π…+π` is not penalised for crossing the ±180° boundary.
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

# External navigation output contract — identical to ground_truth.csv.
ESTIMATE_COLUMNS = ("t", "north", "east", "depth",
                    "v_n", "v_e", "v_d", "yaw_rad", "pitch_rad", "roll_rad")

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


def load_estimates(path: Path) -> dict:
    """Load an external estimates CSV (same columns as ``ground_truth.csv``)."""
    if not path.exists():
        raise FileNotFoundError(f"missing estimates file: {path}")

    try:
        raw = _load_csv(path)
    except KeyError:
        raise ValueError(
            f"{path} has no 't' column; expected: {', '.join(ESTIMATE_COLUMNS)}"
        ) from None

    missing = [c for c in ESTIMATE_COLUMNS[1:] if c not in raw["cols"]]
    if missing:
        raise ValueError(
            f"{path} is missing column(s): {', '.join(missing)}. "
            f"Expected: {', '.join(ESTIMATE_COLUMNS)}"
        )

    # Column order is part of the contract, not whatever order the file uses.
    order = [raw["cols"].index(c) for c in ESTIMATE_COLUMNS[1:]]
    data = raw["data"][:, order]

    # Accept unsorted / duplicated timestamps rather than silently mis-scoring.
    t = raw["t"]
    idx = np.argsort(t, kind="stable")
    t, data = t[idx], data[idx]
    keep = np.concatenate(([True], np.diff(t) > 0))
    t, data = t[keep], data[keep]

    if t.size < 2:
        raise ValueError(f"{path} needs at least two distinct timestamps")

    return {"t": t, "data": data}


def write_estimates_csv(path: Path, t: np.ndarray, pos: np.ndarray,
                        vel: np.ndarray, ypr: np.ndarray) -> None:
    """Write a reference estimates CSV for the C++ implementation to imitate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(ESTIMATE_COLUMNS)
        for i in range(len(t)):
            w.writerow([
                f"{float(t[i]):.6f}",
                f"{float(pos[i, 0]):.8f}", f"{float(pos[i, 1]):.8f}", f"{float(pos[i, 2]):.8f}",
                f"{float(vel[i, 0]):.8f}", f"{float(vel[i, 1]):.8f}", f"{float(vel[i, 2]):.8f}",
                f"{float(ypr[i, 0]):.10f}", f"{float(ypr[i, 1]):.10f}", f"{float(ypr[i, 2]):.10f}",
            ])


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


def _summarize(dp: np.ndarray, dv: np.ndarray, yaw_err: np.ndarray,
               pitch_err: np.ndarray, roll_err: np.ndarray, *,
               has_dvl: bool, samples: int,
               window_start: float, window_end: float) -> dict:
    """Metric block shared by replay scoring and external estimates scoring."""
    return {
        "samples": samples,
        "has_dvl": has_dvl,
        "window_start": window_start,
        "window_end": window_end,
        "north_rmse": _rmse(dp[:, 0]),
        "east_rmse": _rmse(dp[:, 1]),
        "depth_rmse": _rmse(dp[:, 2]),
        "vn_rmse": _rmse(dv[:, 0]),
        "ve_rmse": _rmse(dv[:, 1]),
        "vd_rmse": _rmse(dv[:, 2]),
        "yaw_rmse_deg": float(np.degrees(_rmse(yaw_err))),
        "pitch_rmse_deg": float(np.degrees(_rmse(pitch_err))),
        "roll_rmse_deg": float(np.degrees(_rmse(roll_err))),
        "horizontal_max_m": float(np.max(np.linalg.norm(dp[:, :2], axis=1))),
    }


def compute_metrics(export: Dict[str, dict]) -> dict:
    t, pos, vel, ypr = replay(export)
    gt = _interp_gt(export["ground_truth"], t)

    late = t > SETTLE_TIME
    if not np.any(late):
        late = np.ones_like(t, dtype=bool)

    return _summarize(
        pos[late] - gt["position"][late],
        vel[late] - gt["velocity"][late],
        np.arctan2(np.sin(ypr[late, 0] - gt["ypr"][late, 0]),
                   np.cos(ypr[late, 0] - gt["ypr"][late, 0])),
        ypr[late, 1] - gt["ypr"][late, 1],
        ypr[late, 2] - gt["ypr"][late, 2],
        has_dvl="dvl" in export,
        samples=int(late.sum()),
        window_start=float(t[late][0]),
        window_end=float(t[late][-1]),
    )


def compute_estimates_metrics(estimates: dict, ground_truth: dict,
                              has_dvl: bool) -> dict:
    """Score an external estimate stream against ground truth.

    Estimates are interpolated onto the truth timeline, restricted to the
    overlap of the estimate coverage and the settled window. Extrapolating
    beyond the estimates would clamp ``np.interp`` at its endpoints and
    silently score a flat line, so the overlap is enforced instead.
    """
    gt_t = ground_truth["t"]
    gt = ground_truth["data"]
    idx = {c: i for i, c in enumerate(ground_truth["cols"])}
    est_t = estimates["t"]
    est = estimates["data"]

    lo = max(float(est_t[0]), SETTLE_TIME)
    hi = float(est_t[-1])
    mask = (gt_t >= lo) & (gt_t <= hi)
    if int(mask.sum()) < 2:
        raise ValueError(
            f"estimates cover {est_t[0]:.2f}-{est_t[-1]:.2f} s, which does not overlap "
            f"the scored window (t > {SETTLE_TIME:g} s)"
        )

    t = gt_t[mask]

    def gt_col(name: str) -> np.ndarray:
        return gt[mask, idx[name]]

    def est_col(i: int) -> np.ndarray:
        return np.interp(t, est_t, est[:, i])

    pos_gt = np.column_stack([gt_col("north"), gt_col("east"), gt_col("depth")])
    pos_est = np.column_stack([est_col(0), est_col(1), est_col(2)])
    vel_gt = np.column_stack([gt_col("v_n"), gt_col("v_e"), gt_col("v_d")])
    vel_est = np.column_stack([est_col(3), est_col(4), est_col(5)])

    # Unwrap before interpolating: a ±π jump is a representation change, not error.
    yaw_est = np.interp(t, est_t, np.unwrap(est[:, 6]))
    yaw_gt = np.unwrap(gt_col("yaw_rad"))

    return _summarize(
        pos_est - pos_gt,
        vel_est - vel_gt,
        np.arctan2(np.sin(yaw_est - yaw_gt), np.cos(yaw_est - yaw_gt)),
        est_col(7) - gt_col("pitch_rad"),
        est_col(8) - gt_col("roll_rad"),
        has_dvl=has_dvl,
        samples=int(mask.sum()),
        window_start=float(t[0]),
        window_end=float(t[-1]),
    )


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
    ap.add_argument("--estimates", default=None, metavar="CSV",
                    help="score this external estimates CSV (same columns as "
                         "ground_truth.csv) instead of replaying the sensors")
    ap.add_argument("--write-sample", default=None, metavar="PATH",
                    help="write a reference estimates CSV from the Python filter, then exit")
    ap.add_argument("--quiet", action="store_true", help="only print the summary")
    args = ap.parse_args(argv)

    export_dir = Path(args.exports)
    try:
        export = load_export(export_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.write_sample:
        t, pos, vel, ypr = replay(export)
        out = Path(args.write_sample)
        write_estimates_csv(out, t, pos, vel, ypr)
        print(f"Wrote {len(t)} estimate rows to {out}")
        return 0

    source = "Python ESKF replay"
    try:
        if args.estimates:
            estimates = load_estimates(Path(args.estimates))
            m = compute_estimates_metrics(estimates, export["ground_truth"],
                                          has_dvl="dvl" in export)
            source = f"external estimates ({args.estimates})"
        else:
            m = compute_metrics(export)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    checks = check_metrics(m)

    if not args.quiet:
        print(f"Export  : {export_dir}")
        print(f"Source  : {source}")
        if not args.estimates:
            found = ", ".join(sorted(k for k in export if k != "ground_truth"))
            print(f"Sensors : {found}")
        print(f"Samples : {m['samples']} scored  "
              f"(window: {m['window_start']:.2f}-{m['window_end']:.2f} s)")
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
