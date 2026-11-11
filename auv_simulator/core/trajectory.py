"""120-second 6-stage AUV trajectory.

Stages (from the learning plan):
    1.  0–10 s   Stationary      – docked, zero motion
    2. 10–20 s   Acceleration    – surge forward (constant accel)
    3. 20–50 s   Straight run    – constant velocity, level
    4. 50–70 s   Turning         – smooth yaw arc (constant yaw rate)
    5. 70–90 s   Descending      – constant pitch down + increasing depth
    6. 90–120 s  Continued move  – level at new depth, constant velocity

The trajectory is **analytic**: we compute the exact position, velocity,
acceleration, orientation and angular rate at any time t, so the simulator
has a clean "ground truth" to follow.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable

from .state import VehicleState
from .frames import euler_to_quaternion


# ---------------------------------------------------------------------------
# Stage metadata
# ---------------------------------------------------------------------------
@dataclass
class Stage:
    index: int
    name: str
    t_start: float
    t_end: float

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


STAGES: list[Stage] = [
    Stage(0, "Stationary",     0.0, 10.0),
    Stage(1, "Acceleration", 10.0, 20.0),
    Stage(2, "Straight run",  20.0, 50.0),
    Stage(3, "Turning",       50.0, 70.0),
    Stage(4, "Descending",    70.0, 90.0),
    Stage(5, "Continued",     90.0, 120.0),
]


# ---------------------------------------------------------------------------
# Tunable trajectory parameters
# ---------------------------------------------------------------------------
SURGE_ACCEL       = 0.5      # m/s²  during acceleration stage
CRUISE_SPEED      = 5.0      # m/s   straight-line speed after accel
TURN_YAW_RATE     = np.deg2rad(9.0)   # rad/s   9°/s
TURN_DIRECTION    = +1.0     # +1 = turn right (clockwise viewed from above)
DESCENT_PITCH     = np.deg2rad(-15.0) # radians, nose-down
DESCENT_SPEED     = 2.0      # m/s   along body-x during descent
FINAL_CRUISE_SPEED = 3.0     # m/s   after leveling out
PITCH_RAMP_TIME   = 5.0      # seconds to ramp pitch at descent boundaries

# Initial heading (stage 0–2): pointing North (0 rad)
INITIAL_YAW = 0.0


def _smooth_step(x: float) -> float:
    """Cubic Hermite ramp: 0→1 over x∈[0,1], zero derivative at endpoints."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    return 3 * x ** 2 - 2 * x ** 3


def _descend_end_position() -> np.ndarray:
    """Compute the NED position at the end of the descending stage (t=90s).

    Accounts for linear pitch ramps at start/end using closed-form
    trigonometric integrals.  Must stay in sync with stage 4 logic.
    """
    omega = TURN_YAW_RATE
    r = CRUISE_SPEED / omega
    turn_end_pos = np.array([
        0.5 * SURGE_ACCEL * STAGES[1].duration ** 2 + CRUISE_SPEED * STAGES[2].duration + r * np.sin(omega * STAGES[3].duration),
        r * (1.0 - np.cos(omega * STAGES[3].duration)) * TURN_DIRECTION,
        0.0,
    ])
    yaw = INITIAL_YAW + TURN_DIRECTION * TURN_YAW_RATE * STAGES[3].duration
    V = DESCENT_SPEED
    ramp_T = PITCH_RAMP_TIME
    dur = STAGES[4].duration

    # Sub-phase 1: ramp down 0→DESCENT_PITCH over ramp_T seconds
    dp1 = DESCENT_PITCH / ramp_T
    c_y, s_y = np.cos(yaw), np.sin(yaw)
    pos = np.array([
        turn_end_pos[0] + V * c_y / dp1 * (np.sin(DESCENT_PITCH) - np.sin(0.0)),
        turn_end_pos[1] + V * s_y / dp1 * (np.sin(DESCENT_PITCH) - np.sin(0.0)),
        turn_end_pos[2] + V / dp1 * (np.cos(DESCENT_PITCH) - np.cos(0.0)),
    ])

    # Sub-phase 2: constant pitch DESCENT_PITCH for (dur - 2*ramp_T) seconds
    t_const = dur - 2 * ramp_T
    c_p, s_p = np.cos(DESCENT_PITCH), np.sin(DESCENT_PITCH)
    vel_const = np.array([V * c_y * c_p, V * s_y * c_p, -V * s_p])
    pos = pos + vel_const * t_const

    # Sub-phase 3: ramp up DESCENT_PITCH→0 over ramp_T seconds
    dp2 = -DESCENT_PITCH / ramp_T
    pos = np.array([
        pos[0] + V * c_y / dp2 * (np.sin(0.0) - np.sin(DESCENT_PITCH)),
        pos[1] + V * s_y / dp2 * (np.sin(0.0) - np.sin(DESCENT_PITCH)),
        pos[2] + V / dp2 * (np.cos(0.0) - np.cos(DESCENT_PITCH)),
    ])
    return pos


# ---------------------------------------------------------------------------
# Helper: which stage is active at time t?
# ---------------------------------------------------------------------------
def _stage_at(t: float) -> Stage:
    for s in STAGES:
        if s.t_start <= t < s.t_end:
            return s
    return STAGES[-1]   # clamp


# ---------------------------------------------------------------------------
# Core analytic trajectory function
# ---------------------------------------------------------------------------
def _trajectory_values(t: float) -> dict:
    """Return {position, velocity, acceleration, yaw, pitch, roll, yaw_rate, pitch_rate, roll_rate}."""

    # Clamp
    t = max(0.0, min(t, STAGES[-1].t_end))
    s = _stage_at(t)
    lt = t - s.t_start    # local time within stage

    pos = np.zeros(3)
    vel = np.zeros(3)
    acc = np.zeros(3)
    yaw, pitch, roll = 0.0, 0.0, 0.0
    yaw_rate, pitch_rate, roll_rate = 0.0, 0.0, 0.0

    # ------------------------------------------------------------------
    # Stage 1 — Stationary
    # ------------------------------------------------------------------
    if s.index == 0:
        # Everything zero, heading = North
        yaw = INITIAL_YAW

    # ------------------------------------------------------------------
    # Stage 2 — Acceleration
    # ------------------------------------------------------------------
    elif s.index == 1:
        yaw = INITIAL_YAW
        # surge acceleration along heading
        ax = SURGE_ACCEL * np.cos(yaw)
        ay = SURGE_ACCEL * np.sin(yaw)
        acc = np.array([ax, ay, 0.0])

        v0 = 0.0   # end of stage 0 is zero
        v_mag = v0 + SURGE_ACCEL * lt
        vx = v_mag * np.cos(yaw)
        vy = v_mag * np.sin(yaw)
        vel = np.array([vx, vy, 0.0])

        # integrate position analytically (v0 = 0)
        px = 0.5 * SURGE_ACCEL * lt ** 2 * np.cos(yaw)
        py = 0.5 * SURGE_ACCEL * lt ** 2 * np.sin(yaw)
        pos = np.array([px, py, 0.0])

    # ------------------------------------------------------------------
    # Stage 3 — Straight run (constant velocity, level)
    # ------------------------------------------------------------------
    elif s.index == 2:
        yaw = INITIAL_YAW
        vel = np.array([CRUISE_SPEED * np.cos(yaw),
                        CRUISE_SPEED * np.sin(yaw),
                        0.0])
        pos = np.array([
            0.5 * SURGE_ACCEL * (STAGES[1].duration) ** 2 * np.cos(yaw) + CRUISE_SPEED * lt * np.cos(yaw),
            0.5 * SURGE_ACCEL * (STAGES[1].duration) ** 2 * np.sin(yaw) + CRUISE_SPEED * lt * np.sin(yaw),
            0.0,
        ])

    # ------------------------------------------------------------------
    # Stage 4 — Turning (constant yaw rate)
    # ------------------------------------------------------------------
    elif s.index == 3:
        # Entry conditions from stage 3 end
        prev_pos = np.array([
            0.5 * SURGE_ACCEL * STAGES[1].duration ** 2 + CRUISE_SPEED * STAGES[2].duration,
            0.0,
            0.0,
        ])
        yaw_start = INITIAL_YAW
        yaw = yaw_start + TURN_DIRECTION * TURN_YAW_RATE * lt
        yaw_rate = TURN_DIRECTION * TURN_YAW_RATE

        # constant-speed arc
        # r = v / omega, angle swept = omega * lt
        # displacement = r * (sin(angle), 1 - cos(angle)) rotated by yaw_start
        omega = TURN_YAW_RATE
        r = CRUISE_SPEED / omega
        arc_n = r * np.sin(omega * lt)
        arc_e = r * (1.0 - np.cos(omega * lt)) * TURN_DIRECTION
        pos = prev_pos + np.array([arc_n, arc_e, 0.0])

        # velocity tangent to arc
        vel = np.array([CRUISE_SPEED * np.cos(yaw),
                        CRUISE_SPEED * np.sin(yaw),
                        0.0])

        # centripetal acceleration  a = d v / d t = -ω·r·sin, ω·r·cos = -V·ω·sin, +V·ω·cos
        acc = np.array([
            -CRUISE_SPEED * TURN_YAW_RATE * np.sin(yaw),
             CRUISE_SPEED * TURN_YAW_RATE * np.cos(yaw),
             0.0,
        ])

    # ------------------------------------------------------------------
    # Stage 5 — Descending (linear pitch ramps, constant body speed)
    # ------------------------------------------------------------------
    elif s.index == 4:
        omega = TURN_YAW_RATE
        r = CRUISE_SPEED / omega
        turn_end_pos = np.array([
            0.5 * SURGE_ACCEL * STAGES[1].duration ** 2 + CRUISE_SPEED * STAGES[2].duration + r * np.sin(omega * STAGES[3].duration),
            r * (1.0 - np.cos(omega * STAGES[3].duration)) * TURN_DIRECTION,
            0.0,
        ])
        yaw_at_turn_end = INITIAL_YAW + TURN_DIRECTION * TURN_YAW_RATE * STAGES[3].duration

        yaw = yaw_at_turn_end
        roll = 0.0
        V = DESCENT_SPEED
        ramp_T = PITCH_RAMP_TIME

        # Three sub-phases with LINEAR pitch profile (constant pitch_rate)
        # Phase 1: t∈[0, ramp_T], pitch 0→DESCENT_PITCH, pitch_rate = DESCENT_PITCH/ramp_T
        # Phase 2: t∈[ramp_T, dur-ramp_T], pitch = DESCENT_PITCH, pitch_rate = 0
        # Phase 3: t∈[dur-ramp_T, dur], pitch DESCENT_PITCH→0, pitch_rate = -DESCENT_PITCH/ramp_T
        dur = STAGES[4].duration
        if lt < ramp_T:
            # Phase 1: ramp down
            t_phase = lt
            pitch = DESCENT_PITCH * lt / ramp_T           # linear
            pitch_rate = DESCENT_PITCH / ramp_T            # constant
            p_start = 0.0
            dp = pitch_rate
            pos = turn_end_pos.copy()
            # ∫cos(p0 + dp*t) dt = (1/dp)(sin(p0 + dp*t) - sin(p0))
            c_y, s_y = np.cos(yaw), np.sin(yaw)
            sin_p = np.sin(p_start + dp * t_phase)
            cos_p = np.cos(p_start + dp * t_phase)
            sin_p0 = np.sin(p_start)
            cos_p0 = np.cos(p_start)
            pos = np.array([
                turn_end_pos[0] + V * c_y / dp * (sin_p - sin_p0),
                turn_end_pos[1] + V * s_y / dp * (sin_p - sin_p0),
                turn_end_pos[2] + V / dp * (cos_p - cos_p0),   # ∫-sin(pt)dt = (1/p)*cos(pt)
            ])
            vel = np.array([
                V * c_y * cos_p,
                V * s_y * cos_p,
                -V * sin_p,
            ])
        elif lt < dur - ramp_T:
            # Phase 2: constant pitch
            t_phase1 = ramp_T
            t_phase2 = lt - ramp_T
            pitch_rate_ramp = DESCENT_PITCH / ramp_T
            c_y, s_y = np.cos(yaw), np.sin(yaw)
            # Pos at end of phase 1
            p0_ramp = 0.0
            sin_end = np.sin(DESCENT_PITCH)
            cos_end = np.cos(DESCENT_PITCH)
            sin_0 = np.sin(p0_ramp)
            cos_0 = np.cos(p0_ramp)
            pos1 = np.array([
                turn_end_pos[0] + V * c_y / pitch_rate_ramp * (sin_end - sin_0),
                turn_end_pos[1] + V * s_y / pitch_rate_ramp * (sin_end - sin_0),
                turn_end_pos[2] + V / pitch_rate_ramp * (cos_end - cos_0),
            ])
            # Constant pitch → constant velocity → linear integration
            c_p, s_p = np.cos(DESCENT_PITCH), np.sin(DESCENT_PITCH)
            vel = np.array([V * c_y * c_p, V * s_y * c_p, -V * s_p])
            pos = pos1 + vel * t_phase2
            pitch = DESCENT_PITCH
            pitch_rate = 0.0
        else:
            # Phase 3: ramp up
            t_phase1 = ramp_T
            t_phase2 = dur - 2 * ramp_T
            t_phase3 = lt - (dur - ramp_T)
            pitch_rate_down = DESCENT_PITCH / ramp_T
            pitch_rate_up = -DESCENT_PITCH / ramp_T
            c_y, s_y = np.cos(yaw), np.sin(yaw)
            # Pos at end of ramp-down
            p0_ramp = 0.0
            sin_dp = np.sin(DESCENT_PITCH)
            cos_dp = np.cos(DESCENT_PITCH)
            sin_0 = np.sin(p0_ramp)
            cos_0 = np.cos(p0_ramp)
            pos1 = np.array([
                turn_end_pos[0] + V * c_y / pitch_rate_down * (sin_dp - sin_0),
                turn_end_pos[1] + V * s_y / pitch_rate_down * (sin_dp - sin_0),
                turn_end_pos[2] + V / pitch_rate_down * (cos_dp - cos_0),
            ])
            # Pos at end of constant phase
            c_p, s_p = np.cos(DESCENT_PITCH), np.sin(DESCENT_PITCH)
            vel_const = np.array([V * c_y * c_p, V * s_y * c_p, -V * s_p])
            pos2 = pos1 + vel_const * t_phase2
            # Ramp up: pitch goes from DESCENT_PITCH to 0, pitch_rate_up = -DESCENT_PITCH/ramp_T
            p0_up = DESCENT_PITCH
            dp_up = pitch_rate_up  # negative
            p_current = DESCENT_PITCH + dp_up * t_phase3
            sin_p_curr = np.sin(p_current)
            cos_p_curr = np.cos(p_current)
            sin_p0_up = np.sin(p0_up)
            cos_p0_up = np.cos(p0_up)
            pos = np.array([
                pos2[0] + V * c_y / dp_up * (sin_p_curr - sin_p0_up),
                pos2[1] + V * s_y / dp_up * (sin_p_curr - sin_p0_up),
                pos2[2] + V / dp_up * (cos_p_curr - cos_p0_up),
            ])
            vel = np.array([
                V * c_y * cos_p_curr,
                V * s_y * cos_p_curr,
                -V * sin_p_curr,
            ])
            pitch = p_current
            pitch_rate = dp_up

        # Acceleration: derivative of vel_ned
        if abs(pitch_rate) > 1e-9:
            c_y, s_y = np.cos(yaw), np.sin(yaw)
            c_p, s_p = np.cos(pitch), np.sin(pitch)
            acc = -pitch_rate * np.array([V * c_y * s_p, V * s_y * s_p, -V * c_p])
        else:
            acc = np.zeros(3)

        yaw_rate = 0.0
        roll_rate = 0.0

    # ------------------------------------------------------------------
    # Stage 6 — Continued (level at new depth, constant velocity)
    # ------------------------------------------------------------------
    else:   # s.index == 5
        yaw_at_turn_end = INITIAL_YAW + TURN_DIRECTION * TURN_YAW_RATE * STAGES[3].duration
        descend_end_pos = _descend_end_position()

        yaw = yaw_at_turn_end
        pitch = 0.0
        vel = np.array([FINAL_CRUISE_SPEED * np.cos(yaw),
                        FINAL_CRUISE_SPEED * np.sin(yaw),
                        0.0])
        pos = descend_end_pos + vel * lt
        acc = np.zeros(3)
        yaw_rate = pitch_rate = roll_rate = 0.0

    return dict(
        position=pos, velocity=vel, acceleration=acc,
        yaw=yaw, pitch=pitch, roll=roll,
        yaw_rate=yaw_rate, pitch_rate=pitch_rate, roll_rate=roll_rate,
    )


class Trajectory:
    """Analytic 120-second 6-stage trajectory."""

    duration: float = STAGES[-1].t_end

    def stages(self) -> list[Stage]:
        return list(STAGES)

    def state_at(self, t: float) -> VehicleState:
        """Return the exact ground-truth VehicleState at time t."""
        v = _trajectory_values(t)
        q = euler_to_quaternion(v["yaw"], v["pitch"], v["roll"])
        omega_body = np.array([v["roll_rate"], v["pitch_rate"], v["yaw_rate"]])
        return VehicleState(
            t=t,
            position=v["position"].copy(),
            velocity=v["velocity"].copy(),
            acceleration_ned=v["acceleration"].copy(),
            quaternion=q,
            angular_velocity=omega_body,
        )

    def sample(self, dt: float = 0.01) -> list[VehicleState]:
        """Return a dense list of states sampled every dt seconds."""
        states = []
        t = 0.0
        while t <= self.duration + 1e-9:
            states.append(self.state_at(t))
            t += dt
        return states

    # Convenience lambdas for plotting
    @property
    def position_fn(self) -> Callable[[float], np.ndarray]:
        return lambda t: _trajectory_values(t)["position"]

    @property
    def velocity_fn(self) -> Callable[[float], np.ndarray]:
        return lambda t: _trajectory_values(t)["velocity"]

    @property
    def heading_fn(self) -> Callable[[float], float]:
        return lambda t: _trajectory_values(t)["yaw"]
