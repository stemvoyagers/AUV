"""Tests for the evaluation report generator.

Pin three things: the files exist, the HTML is self-contained (base64 plots),
and the status badge reflects PASS/FAIL when estimates are good/bad.
"""

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.conftest import build_reference_sim                     # noqa: E402
from tools.generate_report import build_report, main               # noqa: E402
from tools.validate_export import (                                # noqa: E402
    check_metrics, compute_metrics, load_export, replay, write_estimates_csv,
)


@pytest.fixture(scope="module")
def export_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("exports")
    sim = build_reference_sim()
    sim.run()
    sim.export_ground_truth_csv(str(tmp / "ground_truth.csv"))
    for name in sim._sensors:
        sim.export_sensor_csv(name, str(tmp / f"{name}.csv"))
    return tmp


def test_replay_report_is_pass(export_dir, tmp_path):
    export = load_export(export_dir)
    out = tmp_path / "out"
    path, meta = build_report(export, estimates=None, out_dir=out,
                              source_label="Python ESKF replay")

    assert path.exists()
    assert (out / "metrics.json").exists()
    assert (out / "plots" / "trajectory.png").exists()
    assert (out / "plots" / "errors.png").exists()
    assert (out / "plots" / "accuracy_bars.png").exists()
    assert meta["status"] == "PASS"

    m = json.loads((out / "metrics.json").read_text())
    assert m["metrics"]["depth_rmse"] < 0.05
    assert m["status"] == "PASS"
    assert m["settle_time_s"] == 20.0
    assert any(s["name"] == "dvl" for s in m["sensors"])


def test_html_is_self_contained(export_dir, tmp_path):
    export = load_export(export_dir)
    path, _ = build_report(export, estimates=None, out_dir=tmp_path,
                           source_label="test")
    text = path.read_text(encoding="utf-8")
    # Three base64-embedded PNGs
    assert text.count("data:image/png;base64,") == 3
    # No external <script> or <img src=...> references
    assert "<script" not in text
    assert re.search(r'<img[^>]+src="(?!data:)', text) is None
    assert 'class="badge pass">PASS<' in text


def test_bad_estimates_report_is_fail(export_dir, tmp_path):
    export = load_export(export_dir)
    t, pos, vel, ypr = replay(export)
    bad_pos = pos.copy()
    bad_pos[:, 0] += 50.0   # 50 m north bias → FAIL north_rmse
    bad = tmp_path / "bad.csv"
    write_estimates_csv(bad, t, bad_pos, vel, ypr)

    out = tmp_path / "out"
    path, meta = build_report(export, estimates=bad, out_dir=out,
                              source_label="bad estimates")
    assert meta["status"] == "FAIL"
    text = path.read_text(encoding="utf-8")
    assert 'class="badge fail">FAIL<' in text


def test_main_cli(export_dir, tmp_path):
    rc = main(["--exports", str(export_dir), "--out", str(tmp_path / "r")])
    assert rc == 0
    assert (tmp_path / "r" / "report.html").exists()
    assert json.loads((tmp_path / "r" / "metrics.json").read_text())["status"] == "PASS"


def test_main_runs_pipeline_when_exports_missing(tmp_path):
    missing = tmp_path / "nope"
    out = tmp_path / "r2"
    rc = main(["--exports", str(missing), "--out", str(out), "--run-pipeline"])
    assert rc == 0
    assert (out / "report.html").exists()


def test_main_returns_2_for_missing_estimates(export_dir, tmp_path):
    rc = main(["--exports", str(export_dir), "--estimates", "no_such_file.csv",
               "--out", str(tmp_path)])
    assert rc == 2
