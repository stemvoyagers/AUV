# AUV Simulator + ESKF Navigation

A self-contained, **pure-Python** simulator for a torpedo-shaped AUV, paired with a
15-state Error-State Kalman Filter (ESKF). It provides a complete, reproducible
pipeline:

```
analytic ground truth  →  multi-rate noisy sensors  →  ESKF estimate  →  compare vs. truth
```

The same sensor streams can be exported to **CSV/JSON** so a C++ implementation can
consume byte-identical input and be validated against the reference filter here.

---

## Table of contents

1. [Highlights](#highlights)
2. [Quick start](#quick-start)
3. [Project layout](#project-layout)
4. [Coordinate & unit conventions](#coordinate--unit-conventions)
5. [Ground-truth trajectory](#ground-truth-trajectory)
6. [Sensor models](#sensor-models)
7. [ESKF design](#eskf-design)
8. [Python API](#python-api)
9. [Exports for the C++ side](#exports-for-the-c-side)
10. [Testing](#testing)
11. [Parameter sweep](#parameter-sweep)
12. [Accuracy results](#accuracy-results)
13. [Design decisions & limitations](#design-decisions--limitations)
14. [Troubleshooting](#troubleshooting)
15. [Roadmap](#roadmap)

---

## Highlights

- **Analytic trajectory** — exact position/velocity/acceleration/attitude at any `t`,
  so “ground truth” never depends on numerical integration of the simulator itself.
- **Five sensor models** with Gaussian noise, constant bias, random-walk drift and
  injectable faults: accelerometer, gyroscope, magnetometer, pressure, DVL.
- **Multi-rate scheduling** — each sensor samples at its own rate (100/100/50/10/5 Hz)
  on a 200 Hz physics tick.
- **15-state ESKF** — position, velocity, attitude, accelerometer bias, gyroscope bias;
  Joseph-form covariance updates and matrix-exponential discretisation.
- **Observability-aware** — accelerometer tilt aid observes roll/pitch, magnetometer
  observes heading, pressure observes depth, DVL observes velocity (and therefore
  bounds horizontal position). Heading stays unobservable from a DVL alone — rotating
  attitude and NED velocity together leaves `R_NB·v_ned` unchanged.
- **Reproducible** — every sensor is seeded; results are bit-stable for a given seed.
- **Regression-locked** — 68 pytest tests, a golden-metrics baseline tied to real accuracy
  numbers, a CSV replay harness, and CI.

---

## Quick start

```bash
# 1. Environment
python -m venv .venv
.\.venv\Scripts\activate        # Windows PowerShell
pip install -r requirements.txt

# 2. Run the tests
pytest tests/ -v

# 3. Full simulation + plots
python examples/run_full_sim.py

# 4. Ground truth → sensors → ESKF → comparison
python examples/run_pipeline.py

# 5. Sensor parameter sweep (stress test)
python examples/run_sweep.py
```

Outputs land in `outputs/` (plots, sweep CSV, simulation exports under `exports/`).

Requirements (`requirements.txt`):

```
numpy>=1.24
matplotlib>=3.7
scipy>=1.10
pytest>=7.0
```

> `scipy` is used only for `scipy.linalg.expm` (covariance discretisation).
> `matplotlib` is used only by the example scripts and is never imported by the core.

---

## Project layout

```
Submarine/
├── auv_simulator/
│   ├── core/
│   │   ├── frames.py        # NED↔body rotations, quaternions, pressure↔depth
│   │   ├── state.py         # VehicleState (ground truth container)
│   │   └── trajectory.py    # 120 s, 6-stage analytic trajectory
│   ├── sensors/
│   │   ├── base.py          # BaseSensor + SensorReading (noise/bias/drift/faults)
│   │   ├── accelerometer.py # specific force f = R_NB (a_ned − g_ned)
│   │   ├── gyroscope.py     # body rates [p, q, r]
│   │   ├── magnetometer.py  # m_body = R_NB · m_earth_ned
│   │   ├── pressure.py      # P = P_atm + ρ g z
│   │   └── dvl.py           # v_body = R_NB · v_ned (bottom-track velocity)
│   ├── simulator/
│   │   └── simulator.py     # multi-rate scheduler + CSV/JSON export
│   ├── navigation/
│   │   └── eskf.py          # 15-state ESKF
│   └── visualization/
│       └── plots.py         # matplotlib helpers
├── examples/
│   ├── run_full_sim.py      # demo + plots + sanity checks
│   ├── run_pipeline.py      # ground truth → sensors → ESKF → compare
│   └── run_sweep.py         # noise/bias stress sweep
├── tools/
│   └── validate_export.py   # replay exports/*.csv → PASS/FAIL accuracy report
├── tests/
│   ├── conftest.py          # shared reference simulator
│   ├── golden/sim_metrics.json
│   ├── test_frames.py
│   ├── test_trajectory.py
│   ├── test_sensors.py
│   ├── test_navigation.py
│   ├── test_dvl.py
│   └── test_golden.py
├── docs/
│   └── IMPLEMENTATION_LOG.md
├── .github/workflows/ci.yml
├── requirements.txt
└── README.md
```

---

## Coordinate & unit conventions

**Everything below is load-bearing.** The ESKF, sensors, and trajectory must agree or
the filter silently diverges.

| Item | Convention |
|------|-----------|
| World frame | **NED** — `x` North, `y` East, `z` **Down** (positive into the water) |
| Body frame | `x` forward (surge), `y` starboard (sway), `z` down (heave) |
| Euler order | aerospace **z-y-x** (yaw → pitch → roll) |
| Yaw | about NED `z`, positive clockwise seen from above |
| Pitch | about body `y`, positive **nose-up** |
| Roll | about body `x`, positive right-wing-down |
| Quaternion | `(w, x, y, z)`, Hamilton convention |
| Gravity | `g = 9.80665 m/s²`; gravity vector in NED is `[0, 0, −g]` (down) |
| Accelerometer | measures **specific force**: `f_body = R_NB · (a_ned − g_ned)` |
| Magnetometer | `m_body = R_NB · m_earth_ned`, `m_earth_ned = [24, 0, 22] µT` |
| Pressure | `P = P_atm + ρ g z`, `P_atm = 101325 Pa`, `ρ = 1025 kg/m³` |
| Depth | `z ≥ 0`; conversion `depth = (P − P_atm)/(ρ g)` |

Rotation helpers in [`core/frames.py`](auv_simulator/core/frames.py):

- `ned_to_body_rotation(yaw, pitch, roll) → R_NB` (`v_body = R_NB · v_ned`)
- `body_to_ned_rotation(yaw, pitch, roll) → R_BN` (= `R_NB.T`)
- `euler_to_quaternion` / `quaternion_to_euler` / `quaternion_multiply`
- `depth_to_pressure` / `pressure_to_depth`

### The sign trap worth remembering

A level, stationary accelerometer reads `[0, 0, +g]` **in body axes**
(not `−g`), because it measures the specific force that cancels gravity:

```
f_body = R_NB · (a_ned − g_ned) = I · (0 − [0,0,−g]) = [0, 0, +g]
```

Getting this backwards flips the tilt aid and the filter will confidently converge to
an upside-down attitude. The unit test `test_accelerometer_pitch_toward_gravity`
guards it.

---

## Ground-truth trajectory

`Trajectory.state_at(t)` returns a `VehicleState` for any `t ∈ [0, 120]`.
It is analytic per stage — no integration drift.

| # | Stage | t start | t end | End position (N, E, D) m | End yaw | Speed |
|---|-------|--------:|------:|--------------------------|--------:|------:|
| 1 | Stationary | 0 | 10 | (0.00, 0.00, 0.00) | 0° | 0 m/s |
| 2 | Acceleration | 10 | 20 | (25.00, 0.00, 0.00) | 0° | 5.0 m/s |
| 3 | Straight run | 20 | 50 | (175.00, 0.00, 0.00) | 0° | 5.0 m/s |
| 4 | Turning | 50 | 70 | (175.00, 63.66, 0.00) | 180° | 5.0 m/s |
| 5 | Descending | 70 | 90 | (135.91, 63.66, 7.78) | 180° | 2.0 m/s |
| 6 | Continued | 90 | 120 | (45.91, 63.66, 7.78) | 180° | 3.0 m/s |

- `SURGE_ACCEL = 0.5 m/s²`, `CRUISE_SPEED = 5 m/s`, `TURN_YAW_RATE = 9°/s` (right turn),
  `DESCENT_PITCH = −15°`, `DESCENT_SPEED = 2 m/s`, `FINAL_CRUISE_SPEED = 3 m/s`.
- Pitch changes are **linear over 5 s ramps** (`PITCH_RAMP_TIME`) at the start and end of
  the descent, exactly as a real vehicle would pitch over. This keeps attitude and
  angular rate continuous — a hard step would be invisible to the gyro.
- Descent position uses closed-form trigonometric integrals, so stage 5→6 uses
  `_descend_end_position()` to stay continuous to machine precision.

Final state: `(45.91, 63.66, 7.78) m`, depth **7.78 m**, heading **180°**.

---

## Sensor models

All sensors derive from [`BaseSensor`](auv_simulator/sensors/base.py):

```python
reading = ideal(state) + N(0, σ) + bias + drift
drift  += N(0, noise_density · √dt)      # integrated random walk
```

`SensorReading` carries both the noisy `data` and the noiseless `data_ideal`, so tests
and plots can separate physics from noise.

| Sensor | `name` | Axes | Units | Ideal model |
|--------|--------|-----:|-------|-------------|
| Accelerometer | `accelerometer` | 3 | m/s² | `R_NB · (a_ned − g_ned)` |
| Gyroscope | `gyroscope` | 3 | rad/s | `[p, q, r]` body rates |
| Magnetometer | `magnetometer` | 3 | µT | `R_NB · [24, 0, 22]` |
| Pressure | `pressure` | 1 | Pa | `P_atm + ρ g z` |
| DVL | `dvl` | 3 | m/s | `R_NB · v_ned` (body-frame velocity over ground) |

### Realistic reference set (used by the pipeline, tests and goldens)

| Sensor | Rate | Noise σ | Bias | Drift density |
|--------|-----:|--------:|-----:|--------------:|
| Accelerometer | 100 Hz | 0.01 m/s² | 0.005 m/s² | 1e−4 |
| Gyroscope | 100 Hz | 0.002 rad/s | 0.001 rad/s | 1e−5 |
| Magnetometer | 50 Hz | 0.1 µT | 0 | — |
| Pressure | 10 Hz | 5 Pa | 1 Pa | — |
| DVL | 5 Hz | 0.01 m/s | 0.005 m/s | 1e−4 |

> 5 Pa ≈ 0.0005 m of depth error — pressure is an extremely strong depth reference.

`examples/run_full_sim.py` deliberately uses a **worse** set (mag 0.5 µT + 1 µT bias,
pressure 20 Pa + 10 Pa bias, gyro 0.01 rad/s) to show a realistic low-cost unit.

### Fault injection

Every sensor accepts the same optional fault arguments; all are inert by default:

```python
DVL(noise_std=0.01,
    dropout_prob=0.05,               # 5 % of samples produce nothing at all
    stuck_at=[0.0, 0.0, 0.0],        # latched output (stuck-at fault)
    fault_start=60.0,                # faults only after 60 s
    scale_factor=1.02,               # +2 % gain error
    misalignment_deg=(0.0, 0.0, 1.5))  # 1.5° mounting boresight error
```

A dropped sample is simply **absent** from `LogEntry.readings` (never a zero-vector),
and `ESKF.run()` keeps propagating on the last good IMU values, so a DVL losing
bottom-lock degrades accuracy instead of crashing the filter.

---

## ESKF design

Implemented in [`navigation/eskf.py`](auv_simulator/navigation/eskf.py).

### State

Nominal state (propagated):

```
p (3)  NED position      v (3)  NED velocity      q (4)  body→NED quaternion
b_a (3) accel bias       b_g (3) gyro bias
```

Error state (estimated, injected, then reset):

```
δx = [ δp(3) | δv(3) | δθ(3) | δb_a(3) | δb_g(3) ]    # 15 states
```

Error convention: **body-frame right perturbation**, `R_true = R_nom · exp([δθ]×)`.
Every Jacobian in the file follows this one convention — mixing conventions is the
classic way to get a filter that “looks stable” but is wrong.

### Propagation (`propagate`)

```
δv̇ = −[R·a_body]× δθ − R δb_a
δθ̇ = −[ω_body]× δθ − δb_g
δṗ = δv
```

Covariance discretised with `Φ = expm(F·dt)` (not Euler `I + F·dt`) so the transition
matrix is always a proper rotation/flow — eigenvalues cannot exceed 1.

Process noise `Q = diag(qp,qp,qp, qv,..., qth,..., qba,..., qbg,...)`:

| Term | Value | Meaning |
|------|------:|---------|
| `qp` | 1e−4 | position random walk |
| `qv` | 1e−2 | velocity random walk |
| `qth` | 1e−6 | attitude random walk |
| `qba` | 1e−8 | accel-bias random walk |
| `qbg` | 1e−10 | gyro-bias random walk |

Initial covariance: position 10, velocity 5, attitude 30°, accel bias 0.5, gyro bias 0.1
(diagonal).

### Measurement updates

| Update | Measurement | Jacobian `H` | Observed |
|--------|-------------|--------------|----------|
| `update_magnetometer(m_body)` | `m_body` | `∂/∂δθ = [R_NB·m_earth]×` | heading (yaw) |
| `update_pressure(depth)` | scalar depth | `∂/∂δp_z = 1` | depth |
| `update_accelerometer(a_body, gyro_body)` | specific-force direction | `∂/∂δθ = g·[f_dir]×` | roll + pitch |
| `update_dvl(v_body)` | body-frame velocity | `∂/∂δv = R_NB`, `∂/∂δθ = [R_NB·v_ned]×` | velocity → horizontal position |

All updates use **Joseph form** for numerical stability:

```
K = P Hᵀ (H P Hᵀ + R)⁻¹
P = (I − K H) P (I − K H)ᵀ + K R Kᵀ
```

### Why the accelerometer tilt aid matters

A magnetometer measures a **single** vector. Rotation *about that vector* is invisible
to it, so yaw and one tilt combination are unobservable from mag alone. That is why a
mag-only filter sits at ~9° roll error forever. The tilt aid adds the gravity
direction, which observes the remaining two attitude DOF. Together:

| Source | Observes |
|--------|----------|
| Gyro + accel tilt | roll, pitch |
| Gyro + magnetometer | yaw |
| Pressure | depth |
| DVL | horizontal velocity, hence bounded horizontal position |

During a coordinated turn the specific force also contains `ω × v_body` (centripetal).
`update_accelerometer` subtracts this term using the estimated body velocity, so a
turn is not mistaken for a false tilt.

---

## Python API

### Simulate

```python
from auv_simulator.simulator import Simulator
from auv_simulator.sensors import (
    Accelerometer, DVL, Gyroscope, Magnetometer, PressureSensor,
)

sim = Simulator(dt_physics=0.005, seed=7)
sim.add_sensor(Accelerometer(noise_std=0.01, bias=0.005, noise_density=1e-4), 100)
sim.add_sensor(Gyroscope(noise_std=0.002, bias=0.001, noise_density=1e-5),    100)
sim.add_sensor(Magnetometer(noise_std=0.1),                                   50)
sim.add_sensor(PressureSensor(noise_std=5.0, bias=1.0),                       10)
sim.add_sensor(DVL(noise_std=0.01, bias=0.005, noise_density=1e-4),            5)

history = sim.run()            # list[LogEntry]
for entry in history[:3]:
    print(entry.t, entry.state.position, list(entry.readings))
```

`LogEntry` = `{ t, state: VehicleState, readings: dict[str, SensorReading] }`.

### Estimate

```python
from auv_simulator.navigation import ESKF

eskf = ESKF()
estimates = eskf.run(history)          # list[NominalState], one per physics tick

print(eskf.position)                   # NED
print(eskf.velocity)
print(eskf.yaw_pitch_roll)             # radians
print(estimates[-1].depth)
```

Lower-level control is available if you want a custom loop:

```python
eskf = ESKF()
for entry in history:
    if "accelerometer" in entry.readings:            # 100 Hz
        dt = 0.01
        eskf.propagate(dt,
                       entry.readings["accelerometer"].data,
                       entry.readings["gyroscope"].data)
        eskf.update_accelerometer(entry.readings["accelerometer"].data,
                                  gyro_body=entry.readings["gyroscope"].data)
    if "magnetometer" in entry.readings:             # 50 Hz
        eskf.update_magnetometer(entry.readings["magnetometer"].data)
    if "pressure" in entry.readings:                 # 10 Hz
        from auv_simulator.core.frames import pressure_to_depth
        depth = float(pressure_to_depth(entry.readings["pressure"].data[0]))
        eskf.update_pressure(depth)
```

### Trajectory directly

```python
from auv_simulator.core.trajectory import Trajectory, STAGES

tr = Trajectory()
s = tr.state_at(65.0)
print(s.position, s.yaw_pitch_roll, s.angular_velocity)
print([st.name for st in STAGES])
```

---

## Exports for the C++ side

`Simulator` writes C++-friendly flat files:

```python
sim.export_ground_truth_csv("exports/ground_truth.csv")
sim.export_sensor_csv("accelerometer", "exports/accelerometer.csv")
sim.export_sensor_csv("gyroscope",     "exports/gyroscope.csv")
sim.export_sensor_csv("magnetometer",  "exports/magnetometer.csv")
sim.export_sensor_csv("pressure",      "exports/pressure.csv")
sim.export_sensor_csv("dvl",           "exports/dvl.csv")
sim.export_json("exports/full_simulation.json")
```

### `ground_truth.csv`

```csv
t,north,east,depth,v_n,v_e,v_d,yaw_rad,pitch_rad,roll_rad
0.000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.0000000000,...
```

One row per physics tick (`dt = 0.005 s` → 24,001 samples over 120 s).

### `sensor_*.csv`

```csv
t,accelerometer_0,accelerometer_1,accelerometer_2
0.000000,0.00000000,0.00000000,9.80665000
```

One row per **sensor** sample, not per physics tick:

| File | Rate | Samples (120 s) |
|------|-----:|----------------:|
| `accelerometer.csv` | 100 Hz | 12,001 |
| `gyroscope.csv` | 100 Hz | 12,001 |
| `magnetometer.csv` | 50 Hz | 6,001 |
| `pressure.csv` | 10 Hz | 1,201 |
| `dvl.csv` | 5 Hz | 601 |

### `full_simulation.json`

```json
{
  "physics_dt": 0.005,
  "trajectory": 120.0,
  "ground_truth": {
    "t": [...], "position": [[n,e,d], ...], "velocity": [[...]],
    "yaw": [...], "pitch": [...], "roll": [...]
  },
  "sensors": {
    "accelerometer": { "frequency_hz": 100, "t": [...], "data": [[ax,ay,az], ...] },
    "gyroscope":     { ... },
    "magnetometer":  { ... },
    "pressure":      { "frequency_hz": 10, "t": [...], "data": [[pa], ...] },
    "dvl":           { "frequency_hz": 5,  "t": [...], "data": [[vx,vy,vz], ...] }
  }
}
```

### C++ consumption notes

- Plain CSV, `,` separated, header on line 1, all numbers `%.8f` (timestamps `%.6f`).
- Stream with `std::ifstream` + `std::getline`, split on `','`, `std::stod`.
- Time is monotonically increasing in every file; interpolate sensor rows onto the
  physics tick only if your implementation needs a single clock.
- All angles are radians; depth is metres positive-down; pressure is pascals.
- Do **not** re-scale: the same numbers, same order, same seed ⇒ identical input.

Minimal reader sketch:

```cpp
std::ifstream f("exports/accelerometer.csv");
std::string line; std::getline(f, line);              // header
while (std::getline(f, line)) {
    std::stringstream ss(line); std::string cell;
    std::getline(ss, cell, ','); double t = std::stod(cell);
    double ax = std::stod((std::getline(ss, cell, ','), cell));
    double ay = std::stod((std::getline(ss, cell, ','), cell));
    double az = std::stod((std::getline(ss, cell, ','), cell));
    // feed your filter...
}
```

---

## Testing

```bash
pytest tests/ -v          # 68 tests
pytest tests/ -q          # summary only
pytest tests/test_dvl.py -v
```

| File | Locks in |
|------|----------|
| `test_frames.py` | rotation orthonormality, round-trip Euler↔quaternion, pressure↔depth |
| `test_trajectory.py` | stage boundaries continuous in position/attitude, endpoint values, final depth |
| `test_sensors.py` | ideal physics, noise statistics, bias, drift, **fault injection** |
| `test_navigation.py` | propagation stability, update behaviour, tilt-aid direction, **accuracy** |
| `test_dvl.py` | DVL physics, velocity observability, bounded position, dropout tolerance |
| `test_golden.py` | end-to-end accuracy against a committed baseline (`tests/golden/sim_metrics.json`) |

Accuracy bounds enforced by `test_navigation.py` (t > 20 s):

| Test | Depth | Yaw | Pitch | Roll |
|------|------:|----:|------:|-----:|
| `test_full_run_reasonable_accuracy` | < 0.02 m | < 0.5° | < 0.3° | < 0.3° |
| `test_truth_only_tracking` (noiseless) | < 0.01 m | < 0.2° | < 0.2° | < 0.2° |

Plus behavioural guards:

- `test_accelerometer_pulls_pitch_toward_gravity` — tilt aid has the correct sign.
- `test_accelerometer_does_not_touch_yaw` — gravity never rotates the heading.
- `test_position_stays_bounded_with_dvl` — horizontal error < 5 m for the whole run
  (without a DVL the identical run drifts to hundreds of metres).
- `test_body_velocity_converges_to_measurement` — pins *what the DVL actually observes*,
  i.e. `R_NB·v`, not world velocity (a DVL cannot see heading).

### Validate a CSV export (no C++ build required)

`exports/*.csv` is the frozen contract between this repo and the C++ implementation.
`tools/validate_export.py` replays a directory of CSVs through the reference ESKF and
scores it against `ground_truth.csv`:

```bash
python tools/validate_export.py                    # defaults to exports/
python tools/validate_export.py --exports path/to/exports --quiet
```

```
Samples : 12001  (settled window: t > 20 s)

Metrics
--------------------------------------------------------
  [PASS] depth_rmse          0.00183   (limit 0.05)
  [PASS] north_rmse          0.31072   (limit 2)
  [PASS] east_rmse           0.26628   (limit 2)
  [PASS] vn_rmse             0.04961   (limit 0.2)
  [PASS] ve_rmse             0.00958   (limit 0.2)
  [PASS] yaw_rmse_deg        0.04202   (limit 1)
  [PASS] pitch_rmse_deg      0.02460   (limit 1)
  [PASS] roll_rmse_deg       0.04419   (limit 1)
  [INFO] horizontal max        0.612   (m)

RESULT: PASS — all required checks within tolerance
```

Exit code `0` = pass, `1` = a metric broke, `2` = the export is unreadable. Use it three
ways:

1. **Prove the Python side works today**, without waiting for the C++ merge.
2. **Score your C++ filter fairly**: have it emit the same CSV schema (an estimate per
   IMU tick), then compare against the same ground truth.
3. **Guard the contract**: if a simulator change silently degrades accuracy, CI fails.

If `dvl.csv` is missing (an older export), the horizontal checks are reported as `INFO`
instead of failing, because without a DVL horizontal position is genuinely unobservable.

### CI

`.github/workflows/ci.yml` runs, on every push and PR: `pytest tests -q` →
`examples/run_pipeline.py` → `tools/validate_export.py` → `examples/run_sweep.py`, and
uploads `outputs/` + `exports/` as artifacts.

---

## Parameter sweep

```bash
python examples/run_sweep.py        # writes outputs/sweep_results.csv
```

Five one-at-a-time sweeps around a nominal sensor set, each re-running the whole
simulation + filter:

| Sweep | Variable | Grid |
|-------|----------|------|
| 1 | Accelerometer noise | 0.01 → 0.20 m/s² |
| 2 | Gyroscope noise | 0.001 → 0.02 rad/s |
| 3 | Magnetometer noise | 0.25 → 2.0 µT |
| 4 | Pressure noise | 5 → 100 Pa |
| 5 | Bias magnitude | 0 → 1.0 × σ |

Example output (excerpt):

```
Sweep 3 — Magnetometer noise
  accel    gyro    mag  press   bias |    depth     yaw   pitch    roll       pos
  0.050  0.0100   0.25   20.0   0.25 |  0.0030m  0.120°  0.102°  0.207°      0.5m
  0.050  0.0100   0.50   20.0   0.25 |  0.0039m  0.221°  0.205°  0.253°      1.0m
  0.050  0.0100   1.00   20.0   0.25 |  0.0059m  0.510°  0.420°  0.368°      2.1m
  0.050  0.0100   2.00   20.0   0.25 |  0.0105m  1.111°  0.855°  0.630°      4.3m
```

Expected, physically sensible behaviour: depth tracks pressure noise almost exactly
(0.003 → 0.011 m across a 20× pressure-noise range), attitude degrades with IMU and
magnetometer noise, and position degrades *with* heading because a yaw error rotates the
DVL's body-frame velocity into a world-frame error. Position is no longer the meaningless
drift number it was before the DVL existed.

---

## Accuracy results

Reference sensor set (see [Sensor models](#sensor-models)), 120 s run, t > 20 s:

| Metric | Before fixes | After fixes (DVL off) | After fixes (DVL on) |
|--------|-------------:|----------------------:|---------------------:|
| Depth RMSE | 61 m → 23 m → 3 mm | **0.002 m** | **0.002 m** |
| Yaw RMSE | 9.0° | **0.047°** | **0.044°** |
| Pitch RMSE | 2.6° | **0.024°** | **0.025°** |
| Roll RMSE | 9.2° | **0.043°** | **0.044°** |
| Horizontal position RMSE | unbounded | 600–1200 m (drift) | **0.41 m** |

With noiseless sensors the filter tracks truth to < 0.2° / < 1 cm. Adding the DVL removes
the horizontal drift entirely and leaves attitude accuracy untouched — the expected result,
since velocity aiding is orthogonal to attitude aiding. See
[`docs/IMPLEMENTATION_LOG.md`](docs/IMPLEMENTATION_LOG.md) for the full diagnosis of each
defect.

---

## Design decisions & limitations

**Decisions**

- *Analytic ground truth.* Cheaper than integrating the same physics twice, and removes
  one source of drift from validation.
- *Error-state, not direct-state.* Attitude errors stay small, quaternion stays
  normalised, and the covariance has a clean linear meaning.
- *Joseph-form updates.* Keeps `P` symmetric positive-definite even with large gains.
- *Matrix-exponential discretisation.* A first-order Euler `Φ` has eigenvalues slightly
  outside the unit circle at 100 Hz; over 10 s this inflates covariance ~90× and makes
  the filter overreact to every measurement.
- *Body-frame error convention everywhere.* One convention, documented once.

**Limitations (deliberate)**

- No GPS / USBL ⇒ no absolute position fix. The DVL bounds *relative* drift; absolute
  position is still unbounded over long missions.
- Flat, non-rotating Earth; constant gravity; simple constant magnetic field.
- No hydrodynamic model, thrusters, drag, or buoyancy — the trajectory is prescribed.
- Pressure uses a uniform density `ρ = 1025 kg/m³` (no salinity/temperature profile).
- The DVL measures over ground: no ocean-current (water-track) error is modelled.
- Scale factor and misalignment are *injectable faults*, not estimated states.

---

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| Filter converges to upside-down attitude | Accelerometer gravity sign flipped. Level vehicle must read `+g` on body-z. |
| Attitude covariance `P[6:9,6:9]` grows without bound | `Φ` built as `I + F·dt`; use `expm`. |
| Yaw slowly spins, roll/pitch stuck near 9° | No accelerometer tilt aid; a single mag vector cannot observe roll. |
| Depth drifts to tens of metres | Pressure update missing, or `noise_r` for pressure wildly mismatched to the sensor σ. |
| Depth estimate biased by a few metres | Pressure `noise_r` too large relative to pressure σ; the filter ignores depth. |
| Position RMSE in the thousands of metres | No DVL in the loop — horizontal position is then unobservable. Add `DVL`, or compare attitude/depth instead. |
| Position wanders after a DVL dropout | Expected: the filter coasts on the IMU until bottom-lock returns. Check `dropout_prob` and that `ESKF.run()` sees gaps (missing keys), not zeros. |
| `KeyError: No readings for sensor ...` on export | You exported a sensor that was never `add_sensor`-ed, or every one of its samples dropped. |
| Plots fail with a backend error | Headless machine: set `MPLBACKEND=Agg`. |

---

## Roadmap

- ✅ DVL velocity update (body-frame `v_body`) → horizontal position is now observable.
- USBL / GPS fix update → bounded *absolute* position error.
- Gyro/accel scale-factor and misalignment states (currently injectable, not estimated).
- Current (ocean drift) disturbance model.
- C++ port of `eskf.py`, validated against `exports/*.csv` with `tools/validate_export.py`.

---

