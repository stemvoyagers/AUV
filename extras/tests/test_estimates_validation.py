"""Tests for external (C++) estimates scoring in ``tools/validate_export.py``.

The estimates contract is the only interface where a C++ filter can be graded
against this simulator, so these tests pin three things: the schema, the
scoring maths, and the CLI exit codes.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.conftest import build_reference_sim                     # noqa: E402
from tools.validate_export import (                                # noqa: E402
    ESTIMATE_COLUMNS,
    check_metrics,
    compute_estimates_metrics,
    load_estimates,
    load_export,
    main,
    replay,
    write_estimates_csv,
)


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """One full mission → CSV contract → reference estimates file."""
    tmp = tmp_path_factory.mktemp("estimates")
    sim = build_reference_sim()
    sim.run()

    sim.export_ground_truth_csv(str(tmp / "ground_truth.csv"))
    for name in sim._sensors:
        sim.export_sensor_csv(name, str(tmp / f"{name}.csv"))

    export = load_export(tmp)
    t, pos, vel, ypr = replay(export)

    sample = tmp / "estimates.csv"
    write_estimates_csv(sample, t, pos, vel, ypr)

    return {
        "dir": tmp, "sample": sample, "ground_truth": export["ground_truth"],
        "t": t, "pos": pos, "vel": vel, "ypr": ypr,
    }


def test_sample_schema_matches_ground_truth(reference):
    """The emitted sample must be column-for-column the truth schema."""
    with reference["sample"].open(newline="") as fh:
        header = tuple(next(csv.reader(fh)))
    assert header == ESTIMATE_COLUMNS


def test_reference_estimates_pass_every_check(reference):
    """A perfect estimate stream must clear all limits — proves the pipeline is fair."""
    m = compute_estimates_metrics(load_estimates(reference["sample"]),
                                  reference["ground_truth"], has_dvl=True)
    failed = [c for c in check_metrics(m) if not c["passed"]]
    assert not failed, failed


def test_offset_estimates_are_rejected(reference, tmp_path):
    """A 50 m northern bias must be caught, not averaged away."""
    pos = reference["pos"].copy()
    pos[:, 0] += 50.0
    bad = tmp_path / "bad.csv"
    write_estimates_csv(bad, reference["t"], pos, reference["vel"], reference["ypr"])

    m = compute_estimates_metrics(load_estimates(bad), reference["ground_truth"],
                                  has_dvl=True)
    assert m["north_rmse"] > 40.0
    assert any(c["name"] == "north_rmse" and not c["passed"]
               for c in check_metrics(m))


def test_yaw_wraparound_is_not_penalised(reference, tmp_path):
    """A filter reporting yaw on a shifted branch is not making an error.

    The baseline carries the filter's own yaw error, so the test asserts the
    shifted stream scores the *same* as the unshifted one — if unwrapping were
    missing, the error would jump to ~360°.
    """
    baseline = compute_estimates_metrics(load_estimates(reference["sample"]),
                                         reference["ground_truth"], has_dvl=True)

    ypr = reference["ypr"].copy()
    ypr[:, 0] += 2.0 * np.pi
    shifted = tmp_path / "shifted.csv"
    write_estimates_csv(shifted, reference["t"], reference["pos"],
                        reference["vel"], ypr)

    m = compute_estimates_metrics(load_estimates(shifted),
                                  reference["ground_truth"], has_dvl=True)
    assert m["yaw_rmse_deg"] == pytest.approx(baseline["yaw_rmse_deg"], abs=1e-9)
    assert m["yaw_rmse_deg"] < 0.1


def test_missing_column_names_the_column(reference, tmp_path):
    with reference["sample"].open(newline="") as fh:
        rows = list(csv.reader(fh))

    keep = [i for i, c in enumerate(rows[0]) if c != "yaw_rad"]
    stripped = tmp_path / "missing_yaw.csv"
    with stripped.open("w", newline="") as fh:
        csv.writer(fh).writerows([[r[i] for i in keep] for r in rows])

    with pytest.raises(ValueError, match="yaw_rad"):
        load_estimates(stripped)


def test_estimates_that_miss_the_window_are_rejected(reference, tmp_path):
    """No overlap with the settled window must be an error, not a silent zero."""
    early = tmp_path / "early.csv"
    t = np.array([0.0, 1.0])
    write_estimates_csv(early, t, reference["pos"][:2],
                        reference["vel"][:2], reference["ypr"][:2])

    with pytest.raises(ValueError, match="does not overlap"):
        compute_estimates_metrics(load_estimates(early),
                                  reference["ground_truth"], has_dvl=True)


def test_cli_exit_codes(reference, tmp_path):
    good = main(["--exports", str(reference["dir"]),
                 "--estimates", str(reference["sample"]), "--quiet"])
    assert good == 0

    pos = reference["pos"].copy()
    pos[:, 0] += 50.0
    bad = tmp_path / "bad.csv"
    write_estimates_csv(bad, reference["t"], pos, reference["vel"], reference["ypr"])

    assert main(["--exports", str(reference["dir"]),
                 "--estimates", str(bad), "--quiet"]) == 1


def test_unreadable_estimates_exit_with_two(reference):
    assert main(["--exports", str(reference["dir"]),
                 "--estimates", "definitely_not_here.csv", "--quiet"]) == 2


def test_write_sample_produces_loadable_file(reference, tmp_path):
    out = tmp_path / "sample.csv"
    assert main(["--exports", str(reference["dir"]),
                 "--write-sample", str(out)]) == 0
    loaded = load_estimates(out)
    assert loaded["t"].size == reference["t"].size
