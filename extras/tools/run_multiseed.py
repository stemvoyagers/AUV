"""Multi-seed validation.

A single lucky noise realisation is not evidence. This runs the reference
mission across several seeds, scores each run with the same code and the same
tolerances as ``tools/validate_export.py``, and reports the spread — not just
one number.

Usage
-----
    python tools/run_multiseed.py                       # seeds 1..5
    python tools/run_multiseed.py --seeds 10            # seeds 1..10
    python tools/run_multiseed.py --first-seed 100 --seeds 3
    python tools/run_multiseed.py --out outputs/multiseed

Output (inside ``--out``):
    results.csv   — one row per seed, every metric, PASS/FAIL
    summary.json  — mean / std / min / max / worst seed, overall status

Exit code 0 → every seed passed.  1 → at least one seed broke tolerance.

Why report the spread and not the mean alone: a filter whose mean passes but
whose worst seed fails is not validated, it is lucky.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.conftest import build_reference_sim             # noqa: E402
from tools.validate_export import (                        # noqa: E402
    ESTIMATE_COLUMNS, SCALAR_SENSORS, VECTOR_SENSORS,
    check_metrics, compute_metrics,
)


def export_from_sim(sim) -> dict:
    """Build an in-memory equivalent of ``exports/*.csv``.

    Reusing ``compute_metrics`` means the multi-seed numbers are produced by the
    identical code path as the CSV harness — no second scoring implementation.
    """
    gt = sim.ground_truth_arrays()
    out = {
        "ground_truth": {
            "t": gt["t"],
            "cols": list(ESTIMATE_COLUMNS[1:]),
            "data": np.column_stack([
                gt["position"][:, 0], gt["position"][:, 1], gt["position"][:, 2],
                gt["velocity"][:, 0], gt["velocity"][:, 1], gt["velocity"][:, 2],
                gt["yaw"], gt["pitch"], gt["roll"],
            ]),
        }
    }

    for name in VECTOR_SENSORS + SCALAR_SENSORS:
        arr = sim.sensor_arrays(name)
        if arr is None:
            continue
        out[name] = {
            "t": arr["t"],
            "data": arr["data"],
            "cols": [f"{name}_{i}" for i in range(arr["data"].shape[1])],
        }
    return out


def run_one(seed: int) -> dict:
    """Simulate and score one seed. Returns a flat metrics row."""
    sim = build_reference_sim(seed=seed)
    sim.run()
    m = compute_metrics(export_from_sim(sim))
    checks = check_metrics(m)
    return {
        "seed": seed,
        "status": "PASS" if all(c["passed"] or not c["required"] for c in checks) else "FAIL",
        "failed": [c["name"] for c in checks if c["required"] and not c["passed"]],
        **{k: v for k, v in m.items() if isinstance(v, (int, float))},
    }


def metric_keys(rows: list[dict]) -> list[str]:
    """Required-check names, in the same order the harness prints them."""
    if not rows:
        return []
    checks = check_metrics(rows[0])
    return [c["name"] for c in checks]


def aggregate(rows: list[dict]) -> dict:
    """Mean / std / min / max / worst seed per metric, plus overall status."""
    keys = metric_keys(rows)
    metrics = {}
    for k in keys:
        values = [r[k] for r in rows]
        worst_i = int(np.argmax(values))
        metrics[k] = {
            "mean": float(np.mean(values)),
            "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "worst_seed": rows[worst_i]["seed"],
        }

    failed = [r["seed"] for r in rows if r["status"] != "PASS"]
    return {
        "runs": len(rows),
        "seeds": [r["seed"] for r in rows],
        "passed": len(rows) - len(failed),
        "failed_seeds": failed,
        "status": "PASS" if not failed else "FAIL",
        "metrics": metrics,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = metric_keys(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", *keys, "status", "failed"])
        for r in rows:
            w.writerow([r["seed"], *[f"{r[k]:.8f}" for k in keys],
                        r["status"], ";".join(r["failed"])])


def _print_table(rows: list[dict], summary: dict) -> None:
    keys = metric_keys(rows)
    header = ["seed"] + [k.replace("_rmse_deg", "_deg").replace("_rmse", "") for k in keys] + ["status"]
    print("  ".join(f"{h:>8}" if i else f"{h:>5}" for i, h in enumerate(header)))
    print("-" * (6 + 10 * len(keys) + 8))

    for r in rows:
        cells = [f"{r['seed']:>5}"]
        for k in keys:
            v = r[k]
            cells.append(f"{v:>8.4f}" if abs(v) < 100 else f"{v:>8.2f}")
        cells.append(f"{r['status']:>8}")
        print("  ".join(cells))

    print("-" * (6 + 10 * len(keys) + 8))
    cells = ["mean"]
    for k in keys:
        cells.append(f"{summary['metrics'][k]['mean']:>8.4f}")
    passed = f"{summary['passed']}/{summary['runs']}"
    cells.append(f"{passed:>8}")
    print("  ".join(cells))

    cells = ["worst"]
    for k in keys:
        cells.append(f"{summary['metrics'][k]['max']:>8.4f}")
    cells.append("")
    print("  ".join(cells))

    worst_parts = [f"{k}={summary['metrics'][k]['max']:.4f} (seed {summary['metrics'][k]['worst_seed']})"
                   for k in keys]
    print("\nWorst case: " + ", ".join(worst_parts))


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=5,
                    help="number of consecutive seeds to run (default: 5)")
    ap.add_argument("--first-seed", type=int, default=1,
                    help="first seed in the sequence (default: 1)")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "multiseed"),
                    help="output directory (default: outputs/multiseed)")
    args = ap.parse_args(argv)

    if args.seeds < 1:
        print("ERROR: --seeds must be >= 1", file=sys.stderr)
        return 2

    first = args.first_seed
    rows = []
    for seed in range(first, first + args.seeds):
        row = run_one(seed)
        rows.append(row)
        print(f"  seed {seed:>3}: {row['status']}"
              + (f"  ({', '.join(row['failed'])})" if row["failed"] else ""))

    summary = aggregate(rows)
    out_dir = Path(args.out)
    write_csv(out_dir / "results.csv", rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    _print_table(rows, summary)
    print(f"\nResults : {out_dir / 'results.csv'}")
    print(f"Summary : {out_dir / 'summary.json'}")

    if summary["status"] == "PASS":
        print(f"\nRESULT: PASS — {summary['passed']}/{summary['runs']} seeds within tolerance")
        return 0

    print(f"\nRESULT: FAIL — {len(summary['failed_seeds'])} seed(s) outside tolerance: "
          f"{summary['failed_seeds']}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
