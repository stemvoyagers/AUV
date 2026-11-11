"""Golden-file regression test.

Runs the reference simulation end-to-end — truth → sensors → CSV export →
replay through the ESKF — and compares the resulting accuracy metrics against
a committed baseline (``tests/golden/sim_metrics.json``).

This is the test that catches *silent quality regressions*: a filter can still
converge while being ten times less accurate, and unit tests alone would not
notice.

Regenerate the baseline deliberately (never automatically) with:

    python tests/test_golden.py --update
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.conftest import build_reference_sim       # noqa: E402
from tools.validate_export import compute_metrics, load_export  # noqa: E402

GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "sim_metrics.json"

# Metrics that must not regress. Tolerances are wide enough to absorb
# floating-point/platform differences and narrow enough to catch a real bug.
METRICS = (
    "north_rmse", "east_rmse", "depth_rmse",
    "vn_rmse", "ve_rmse", "vd_rmse",
    "yaw_rmse_deg", "pitch_rmse_deg", "roll_rmse_deg",
)
REL_TOL = 0.25
ABS_TOL = 1e-3


def _export_run(tmp_dir: Path) -> dict:
    """Simulate, export the CSV contract, replay it, and return the metrics."""
    sim = build_reference_sim()
    sim.run()

    sim.export_ground_truth_csv(str(tmp_dir / "ground_truth.csv"))
    for name in sim._sensors:
        sim.export_sensor_csv(name, str(tmp_dir / f"{name}.csv"))

    return compute_metrics(load_export(tmp_dir))


def test_export_contains_all_sensors(tmp_path):
    _export_run(tmp_path)
    for name in ("ground_truth", "accelerometer", "gyroscope",
                 "magnetometer", "pressure", "dvl"):
        assert (tmp_path / f"{name}.csv").exists(), f"missing {name}.csv"


def test_metrics_match_golden(tmp_path):
    golden = json.loads(GOLDEN_PATH.read_text())
    got = _export_run(tmp_path)

    failures = []
    for key in METRICS:
        want = golden[key]
        value = got[key]
        if abs(value - want) > max(ABS_TOL, REL_TOL * abs(want)):
            failures.append(f"{key}: {value:.5f} vs golden {want:.5f}")

    assert not failures, "navigation accuracy regressed:\n  " + "\n  ".join(failures)
