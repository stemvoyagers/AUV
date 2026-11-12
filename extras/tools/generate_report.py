"""End-to-end evaluation report generator.

One artefact you can open, share, archive or feed to CI.

Usage
-----
    python tools/generate_report.py                          # run full pipeline + report
    python tools/generate_report.py --exports path/to/exports   # report only, reuse CSVs
    python tools/generate_report.py --estimates cpp.csv          # report the C++ filter
    python tools/generate_report.py --out outputs/my_report     # custom output dir

Output (inside ``--out``):
    report.html   — single self-contained page (plots base64-embedded, no external deps)
    metrics.json  — every number, machine-readable
    plots/*.png   — raw PNGs for reuse elsewhere

Why HTML, not PDF? PDF is heavier, requires a printer backend that often fails on
headless boxes, and the HTML version renders identically in any browser. Base64-
embedding means the file does not depend on a ``plots/`` subfolder — copy one
``.html`` and it works.
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import html
import io
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402

from tests.conftest import build_reference_sim             # noqa: E402
from auv_simulator.navigation import ESKF                   # noqa: E402
from tools.validate_export import (                         # noqa: E402
    LIMITS, SETTLE_TIME, check_metrics, compute_estimates_metrics,
    compute_metrics, load_estimates, load_export, replay,
)


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _plot_trajectory_3d(gt_t, gt_pos, est_t, est_pos):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], -gt_pos[:, 2],
            color="navy", lw=2, label="Ground truth")
    ax.plot(est_pos[:, 0], est_pos[:, 1], -est_pos[:, 2],
            color="crimson", lw=1.2, ls="--", label="Navigation estimate")
    ax.set_xlabel("North (m)"); ax.set_ylabel("East (m)"); ax.set_zlabel("Depth (m, + down)")
    ax.set_title("Trajectory comparison")
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_errors(t, gt_pos, est_pos, gt_ypr, est_ypr):
    pos_err = est_pos - gt_pos
    yaw_err = np.degrees(np.arctan2(
        np.sin(est_ypr[:, 0] - gt_ypr[:, 0]),
        np.cos(est_ypr[:, 0] - gt_ypr[:, 0])))
    pitch_err = np.degrees(est_ypr[:, 1] - gt_ypr[:, 1])
    roll_err = np.degrees(est_ypr[:, 2] - gt_ypr[:, 2])

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, pos_err[:, 0], label="North", color="tab:blue")
    axes[0].plot(t, pos_err[:, 1], label="East",  color="tab:orange")
    axes[0].set_ylabel("Error (m)"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_title("Horizontal position error")

    axes[1].plot(t, pos_err[:, 2], color="tab:green")
    axes[1].set_ylabel("Depth err (m)"); axes[1].grid(alpha=0.3)
    axes[1].set_title("Depth error")

    axes[2].plot(t, yaw_err, label="Yaw",   color="tab:red")
    axes[2].plot(t, pitch_err, label="Pitch", color="tab:purple")
    axes[2].plot(t, roll_err,  label="Roll",  color="tab:brown")
    axes[2].set_xlabel("Time (s)"); axes[2].set_ylabel("Error (deg)")
    axes[2].legend(); axes[2].grid(alpha=0.3)
    axes[2].set_title("Attitude error")

    fig.tight_layout()
    return fig


def _plot_final_error_bars(metrics: dict):
    names = ["depth", "north", "east", "vn", "ve", "yaw_deg", "pitch_deg", "roll_deg"]
    values = [
        metrics["depth_rmse"],
        metrics["north_rmse"], metrics["east_rmse"],
        metrics["vn_rmse"], metrics["ve_rmse"],
        metrics["yaw_rmse_deg"], metrics["pitch_rmse_deg"], metrics["roll_rmse_deg"],
    ]
    limits = [0.05, 2.0, 2.0, 0.2, 0.2, 1.0, 1.0, 1.0]

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(names))
    bars = ax.bar(x, values, color="#2196F3", edgecolor="black")
    ax.step(x - 0.45, limits, color="red", where="post", lw=2, label="Tolerance")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30)
    ax.set_ylabel("RMSE")
    ax.set_title("Accuracy vs tolerance (scored window)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar, v, lim in zip(bars, values, limits):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                f"{v:.4f}" if v < 0.1 else f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig


def _sensor_summary(export: dict) -> list[dict]:
    """Minimal "what was fed into the filter" table."""
    info = []
    for name, keys in (("accelerometer", "m/s²"), ("gyroscope", "rad/s"),
                       ("magnetometer", "µT"), ("pressure", "Pa"), ("dvl", "m/s")):
        if name not in export:
            continue
        arr = export[name]
        f_hz = round(1.0 / float(np.median(np.diff(arr["t"]))), 2)
        info.append({"name": name, "samples": int(arr["t"].size),
                     "rate_hz": f_hz, "units": keys})
    return info


def _build_html(title: str, generated_at: str, source: str,
                metrics: dict, checks: list[dict],
                sensor_info: list[dict],
                plot_b64: dict[str, str]) -> str:
    rows = "\n".join(
        f"""        <tr>
            <td><code>{html.escape(c['name'])}</code></td>
            <td>{c['value']:.5f}</td>
            <td>{c['limit']:g}</td>
            <td class="status-{'pass' if c['passed'] else ('fail' if c['required'] else 'info')}">
                {"PASS" if c['passed'] else ("FAIL" if c['required'] else "INFO")}
            </td>
        </tr>"""
        for c in checks
    )

    sensor_rows = "\n".join(
        f"""        <tr>
            <td>{s['name']}</td><td>{s['samples']}</td>
            <td>{s['rate_hz']} Hz</td><td>{s['units']}</td>
        </tr>"""
        for s in sensor_info
    )

    failed = [c for c in checks if c["required"] and not c["passed"]]
    status = "PASS" if not failed else "FAIL"
    status_cls = "pass" if status == "PASS" else "fail"
    summary = (f"{metrics['samples']} samples scored, window "
               f"{metrics['window_start']:.1f}–{metrics['window_end']:.1f} s<br>"
               f"Horizontal max: {metrics['horizontal_max_m']:.3f} m")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
  body {{ font-family: "Segoe UI", Arial, sans-serif; margin: 24px; color: #222; }}
  h1 {{ margin-bottom: 4px; }}
  .meta {{ color: #666; margin-bottom: 20px; font-size: 14px; }}
  .badge {{ display: inline-block; padding: 6px 14px; border-radius: 4px;
            color: white; font-weight: bold; font-size: 18px; margin-left: 10px; }}
  .badge.pass {{ background: #2e7d32; }}
  .badge.fail {{ background: #c62828; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #f5f5f5; }}
  .status-pass {{ color: #2e7d32; font-weight: bold; }}
  .status-fail {{ color: #c62828; font-weight: bold; }}
  .status-info {{ color: #757575; font-style: italic; }}
  img {{ max-width: 100%; border: 1px solid #e0e0e0; border-radius: 4px; margin: 6px 0; }}
  h2 {{ border-bottom: 1px solid #e0e0e0; padding-bottom: 4px; margin-top: 28px; }}
  code {{ background: #f0f0f0; padding: 1px 4px; border-radius: 3px; }}
</style>
</head>
<body>
<h1>{html.escape(title)}<span class="badge {status_cls}">{status}</span></h1>
<div class="meta">
  Generated {html.escape(generated_at)} · source: {html.escape(source)}<br>
  {summary}
</div>

<h2>Accuracy</h2>
<table>
  <tr><th>Metric</th><th>RMSE</th><th>Tolerance</th><th>Status</th></tr>
{rows}
</table>

<h2>Sensors used</h2>
<table>
  <tr><th>Sensor</th><th>Samples</th><th>Rate</th><th>Units</th></tr>
{sensor_rows}
</table>

<h2>Plots</h2>
<img alt="trajectory" src="data:image/png;base64,{plot_b64['trajectory']}">
<img alt="errors" src="data:image/png;base64,{plot_b64['errors']}">
<img alt="bars" src="data:image/png;base64,{plot_b64['bars']}">
</body>
</html>
"""


def build_report(export: dict, *, estimates: Optional[Path], out_dir: Path,
                 source_label: str) -> tuple[Path, dict]:
    """Assemble the HTML + JSON report from loaded CSVs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    gt_col_names = export["ground_truth"]["cols"]
    idx = {c: i for i, c in enumerate(gt_col_names)}
    gt_t = export["ground_truth"]["t"]

    def gt_at(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate truth pos and ypr onto an arbitrary timeline."""
        pos = np.column_stack([
            np.interp(t, gt_t, export["ground_truth"]["data"][:, idx["north"]]),
            np.interp(t, gt_t, export["ground_truth"]["data"][:, idx["east"]]),
            np.interp(t, gt_t, export["ground_truth"]["data"][:, idx["depth"]]),
        ])
        ypr = np.column_stack([
            np.interp(t, gt_t, np.unwrap(export["ground_truth"]["data"][:, idx["yaw_rad"]])),
            np.interp(t, gt_t, export["ground_truth"]["data"][:, idx["pitch_rad"]]),
            np.interp(t, gt_t, export["ground_truth"]["data"][:, idx["roll_rad"]]),
        ])
        # Re-wrap yaw to (-π, π] for plotting — unwrap is just for interpolation.
        yaw = ypr[:, 0] % (2 * np.pi)
        yaw[yaw > np.pi] -= 2 * np.pi
        ypr[:, 0] = yaw
        return pos, ypr

    if estimates:
        est = load_estimates(estimates)
        m = compute_estimates_metrics(est, export["ground_truth"],
                                      has_dvl="dvl" in export)
        t = est["t"]
        est_pos = np.column_stack([est["data"][:, 0], est["data"][:, 1], est["data"][:, 2]])
        est_ypr = np.column_stack([est["data"][:, 6], est["data"][:, 7], est["data"][:, 8]])
    else:
        t, pos, vel, ypr = replay(export)
        m = compute_metrics(export)
        est_pos = pos
        est_ypr = ypr

    gt_pos, gt_ypr = gt_at(t)

    checks = check_metrics(m)
    fig_traj = _plot_trajectory_3d(gt_t, gt_pos, t, est_pos)
    fig_err = _plot_errors(t, gt_pos, est_pos, gt_ypr, est_ypr)
    fig_bars = _plot_final_error_bars(m)

    # Save raw PNGs too — useful when someone wants just the image.
    fig_traj.savefig(plot_dir / "trajectory.png", dpi=150)
    fig_err.savefig(plot_dir / "errors.png", dpi=150)
    fig_bars.savefig(plot_dir / "accuracy_bars.png", dpi=150)
    plt.close("all")

    plot_b64 = {
        "trajectory": _fig_to_b64(_plot_trajectory_3d(gt_t, gt_pos, t, est_pos)),
        "errors":     _fig_to_b64(_plot_errors(t, gt_pos, est_pos, gt_ypr, est_ypr)),
        "bars":       _fig_to_b64(_plot_final_error_bars(m)),
    }

    failed = [c for c in checks if c["required"] and not c["passed"]]
    status = "PASS" if not failed else "FAIL"
    checks_out = [{
        "name": c["name"], "value": c["value"], "limit": c["limit"],
        "required": c["required"], "passed": c["passed"],
    } for c in checks]

    meta = {
        "generated_at": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source": source_label,
        "status": status,
        "metrics": m,
        "checks": checks_out,
        "limits": LIMITS,
        "settle_time_s": SETTLE_TIME,
        "sensors": _sensor_summary(export),
    }

    report_path = out_dir / "report.html"
    report_path.write_text(_build_html(
        title="AUV Navigator Evaluation Report",
        generated_at=meta["generated_at"],
        source=source_label,
        metrics=m, checks=checks,
        sensor_info=meta["sensors"],
        plot_b64=plot_b64,
    ), encoding="utf-8")

    (out_dir / "metrics.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    return report_path, meta


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exports", default=str(ROOT / "exports"),
                    help="directory of CSV exports (default: exports/)")
    ap.add_argument("--estimates", default=None, metavar="CSV",
                    help="score this external estimates CSV instead of replaying sensors")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "report"),
                    help="output directory (default: outputs/report)")
    ap.add_argument("--run-pipeline", action="store_true",
                    help="run the simulator first, overwriting exports/*.csv")
    args = ap.parse_args(argv)

    export_dir = Path(args.exports)

    if args.run_pipeline or not (export_dir / "ground_truth.csv").exists():
        sim = build_reference_sim()
        sim.run()
        export_dir.mkdir(parents=True, exist_ok=True)
        sim.export_ground_truth_csv(str(export_dir / "ground_truth.csv"))
        for name in sim._sensors:
            sim.export_sensor_csv(name, str(export_dir / f"{name}.csv"))

    try:
        export = load_export(export_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    source = (f"external estimates ({args.estimates})" if args.estimates
              else f"Python ESKF replay ({export_dir})")

    try:
        report_path, meta = build_report(
            export,
            estimates=Path(args.estimates) if args.estimates else None,
            out_dir=Path(args.out),
            source_label=source,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Report   : {report_path}")
    print(f"Metrics  : {Path(args.out) / 'metrics.json'}")
    print(f"Status   : {meta['status']}")
    return 0 if meta["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
