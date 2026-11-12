"""Tests for multi-seed validation.

Two layers:
  * pure aggregation tests (fast, no simulation) — pin the statistics and status
  * one integration run over two seeds — proves seeds change the noise and that
    the artefacts are written
"""

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.run_multiseed import aggregate, main, metric_keys, run_one, write_csv


def _row(seed: int, north: float, yaw: float) -> dict:
    """Minimal row shaped like ``run_one``'s output."""
    return {
        "seed": seed, "status": "PASS", "failed": [],
        "samples": 10000, "has_dvl": 1, "window_start": 20.0, "window_end": 120.0,
        "north_rmse": north, "east_rmse": north, "depth_rmse": 0.002,
        "vn_rmse": 0.05, "ve_rmse": 0.01, "vd_rmse": 0.02,
        "yaw_rmse_deg": yaw, "pitch_rmse_deg": 0.02, "roll_rmse_deg": 0.04,
        "horizontal_max_m": north * 2,
    }


class TestAggregate:
    def test_mean_std_min_max_and_worst_seed(self):
        rows = [_row(1, 0.30, 0.04), _row(2, 0.50, 0.06), _row(3, 0.40, 0.05)]
        s = aggregate(rows)
        north = s["metrics"]["north_rmse"]
        assert north["mean"] == pytest.approx(0.40)
        assert north["min"] == pytest.approx(0.30)
        assert north["max"] == pytest.approx(0.50)
        assert north["worst_seed"] == 2
        assert north["std"] == pytest.approx(0.08164966, rel=1e-4)
        assert s["status"] == "PASS"
        assert s["passed"] == 3

    def test_one_failing_seed_fails_the_run(self):
        rows = [_row(1, 0.30, 0.04), _row(2, 5.00, 0.04)]
        rows[1]["status"] = "FAIL"
        rows[1]["failed"] = ["north_rmse"]
        s = aggregate(rows)
        assert s["status"] == "FAIL"
        assert s["failed_seeds"] == [2]
        assert s["passed"] == 1

    def test_single_seed_has_zero_spread(self):
        s = aggregate([_row(7, 0.31, 0.04)])
        assert s["metrics"]["north_rmse"]["std"] == 0.0
        assert s["metrics"]["north_rmse"]["worst_seed"] == 7

    def test_metric_keys_come_from_the_check_list(self):
        keys = metric_keys([_row(1, 0.3, 0.04)])
        # Required checks only — horizontal_max_m is informational.
        assert "north_rmse" in keys and "yaw_rmse_deg" in keys
        assert "horizontal_max_m" not in keys
        assert "samples" not in keys


class TestArtefacts:
    def test_write_csv_round_trip(self, tmp_path):
        rows = [_row(1, 0.30, 0.04), _row(2, 0.50, 0.06)]
        path = tmp_path / "results.csv"
        write_csv(path, rows)

        with path.open(newline="") as fh:
            got = list(csv.DictReader(fh))

        assert [r["seed"] for r in got] == ["1", "2"]
        assert float(got[1]["north_rmse"]) == pytest.approx(0.50)
        assert got[0]["status"] == "PASS"


@pytest.fixture(scope="module")
def two_seed_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("multiseed")
    rc = main(["--seeds", "2", "--out", str(out)])
    return rc, out


def test_two_seeds_both_pass_and_write_artefacts(two_seed_run):
    rc, out = two_seed_run
    assert rc == 0

    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "PASS"
    assert summary["runs"] == 2
    assert summary["passed"] == 2
    assert (out / "results.csv").exists()

    with (out / "results.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2


def test_seeds_actually_change_the_noise(two_seed_run):
    """If two seeds produced identical numbers, the seed would not be wired through."""
    _, out = two_seed_run
    with (out / "results.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))

    north = [float(r["north_rmse"]) for r in rows]
    yaw = [float(r["yaw_rmse_deg"]) for r in rows]
    assert north[0] != north[1]
    assert yaw[0] != yaw[1]
    # ...but both stay in the same physical ballpark.
    assert all(0.0 < v < 2.0 for v in north)


def test_run_one_is_deterministic_per_seed():
    a = run_one(11)
    b = run_one(11)
    for k in metric_keys([a]):
        assert a[k] == b[k], f"{k} not reproducible for the same seed"


def test_cli_rejects_zero_seeds():
    assert main(["--seeds", "0"]) == 2
