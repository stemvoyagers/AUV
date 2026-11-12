"""Error-State Kalman Filter (ESKF) for AUV navigation.

State layout (Error-State, 15 × 1)
-----------------------------------
  [δp, δv, δθ, δb_a, δb_g]
   3    3    3    3     3

δθ is the small-angle error on orientation (rad), i.e.
    R_true ≈ R_nominal · (I + [δθ]×)
where [δθ]× is the skew-symmetric cross-product matrix.

Nominal state (17 × 1) lives inside ``NominalState`` and is propagated
directly from accelerometer + gyroscope readings.

Sensor updates
--------------
``ESKF.update_magnetometer(mag_body)``
    Corrects δθ using the known Earth-field vector.

``ESKF.update_pressure(depth_meas)``
    Corrects δp (depth axis) using pressure-derived depth.

``ESKF.update_accelerometer(accel_body)``
    Tilt aid.  Uses the *direction* of the measured specific force as a
    gravity reference.  This observes roll and pitch (the two DOF that a
    magnetometer alone cannot observe, since rotation about the magnetic
    field vector leaves the reading unchanged).

Propagation
-----------
``ESKF.propagate(dt, accel_body, gyro_body)``
    Advances both the nominal state and the covariance P by dt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.linalg import expm

from ..core.frames import (
    GRAVITY,
    ned_to_body_rotation,
    body_to_ned_rotation,
    quaternion_multiply,
    quaternion_normalize,
)


# ---------------------------------------------------------------------------
# Skew-symmetric cross-product matrix
# ---------------------------------------------------------------------------
def _skew(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).ravel()
    return np.array([
        [0.0,  -v[2],  v[1]],
        [v[2],  0.0,  -v[0]],
        [-v[1], v[0],  0.0],
    ])


# ---------------------------------------------------------------------------
# Nominal + Error state containers
# ---------------------------------------------------------------------------
@dataclass
class NominalState:
    t: float = 0.0
    p: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    q: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    b_a: np.ndarray = field(default_factory=lambda: np.zeros(3))
    b_g: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def depth(self) -> float:
        return float(self.p[2])

    @property
    def yaw_pitch_roll(self) -> tuple[float, float, float]:
        from ..core.frames import quaternion_to_euler
        return quaternion_to_euler(self.q)

    def copy(self) -> "NominalState":
        return NominalState(
            t=self.t,
            p=self.p.copy(),
            v=self.v.copy(),
            q=self.q.copy(),
            b_a=self.b_a.copy(),
            b_g=self.b_g.copy(),
        )


# Convenience type
ErrorState = np.ndarray   # 15-vector


# ---------------------------------------------------------------------------
# Core ESKF
# ---------------------------------------------------------------------------
class ESKF:
    """Simple but correct 15-state ESKF."""

    N = 15

    def __init__(self,
                 process_noise: Optional[dict] = None,
                 initial_cov: Optional[np.ndarray] = None,
                 gravity: float = GRAVITY):
        self.g = gravity

        # Process-noise strengths (PSD * dt, i.e. variance added per step)
        # Values reflect the physical sensor suite: a low-noise MEMS IMU with
        # near-constant biases (bias random walk is therefore tiny).
        qp = 1e-4       # position random walk        (m²/s)
        qv = 1e-2       # velocity random walk        (m²/s³)
        qth = 1e-6      # attitude random walk        (rad²/s)
        qba = 1e-8      # accel-bias random walk      (m²/s⁵)
        qbg = 1e-10     # gyro-bias random walk       (rad²/s³)

        if process_noise is not None:
            qp  = process_noise.get("qp",  qp)
            qv  = process_noise.get("qv",  qv)
            qth = process_noise.get("qth", qth)
            qba = process_noise.get("qba", qba)
            qbg = process_noise.get("qbg", qbg)

        self.Q_base = np.diag([qp, qp, qp,
                               qv, qv, qv,
                               qth, qth, qth,
                               qba, qba, qba,
                               qbg, qbg, qbg])

        self.nominal = NominalState()
        self.dx = np.zeros(self.N)

        if initial_cov is None:
            self.P = np.diag([10.0, 10.0, 10.0,         # pos uncertainty 10 m
                              5.0, 5.0, 5.0,            # vel uncertainty 5 m/s
                              np.deg2rad(30), np.deg2rad(30), np.deg2rad(30),
                              0.5, 0.5, 0.5,            # accel bias
                              0.1, 0.1, 0.1])           # gyro bias
        else:
            self.P = np.asarray(initial_cov, dtype=float).copy()

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        return self.nominal.p.copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.nominal.v.copy()

    @property
    def quaternion(self) -> np.ndarray:
        return self.nominal.q.copy()

    @property
    def yaw_pitch_roll(self) -> tuple[float, float, float]:
        from ..core.frames import quaternion_to_euler
        return quaternion_to_euler(self.nominal.q)

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------
    def propagate(self, dt: float, accel_body: np.ndarray, gyro_body: np.ndarray) -> None:
        """Advance the nominal state and the covariance by dt seconds.

        ``accel_body`` and ``gyro_body`` are the *raw* sensor readings; the
        filter subtracts its own bias estimates internally.
        """
        if dt <= 0:
            return

        a_body = np.asarray(accel_body, dtype=float) - self.nominal.b_a
        omega_body = np.asarray(gyro_body, dtype=float) - self.nominal.b_g

        R = body_to_ned_rotation(*self.yaw_pitch_roll)   # R_BN

        # ---- Nominal kinematics ----
        # accel_body = R_NB @ (a_ned - gravity_ned)
        #   gravity_ned = (0, 0, -g) in NED convention (down is +z, gravity pulls down)
        # →  a_ned = R_BN @ accel_body + gravity_ned  = R_BN @ accel_body + (0,0,-g)
        a_ned = R @ a_body + np.array([0.0, 0.0, -self.g])
        self.nominal.p = self.nominal.p + self.nominal.v * dt + 0.5 * a_ned * dt ** 2
        self.nominal.v = self.nominal.v + a_ned * dt

        # Quaternion integration: dq/dt = 0.5 * q ⊗ (0, omega)
        w, x, y, z = self.nominal.q
        p, qg, r = omega_body
        dq = 0.5 * np.array([
            -x * p - y * qg - z * r,
             w * p + y * r - z * qg,
             w * qg - x * r + z * p,
             w * r + x * qg - y * p,
        ])
        self.nominal.q = quaternion_normalize(self.nominal.q + dq * dt)
        self.nominal.t += dt

        # ---- Error-state Jacobians ----
        F = np.zeros((self.N, self.N))

        # δp_dot = δv
        F[0:3, 3:6] = np.eye(3)

        # δv_dot = -[R @ a_body]× · δθ - R · δb_a in NED frame
        F[3:6, 6:9] = -R @ _skew(a_body)
        F[3:6, 9:12] = -R

        # δθ_dot = -[ω_body]× · δθ - δb_g   (body-frame right perturbation convention)
        F[6:9, 6:9] = -_skew(omega_body)
        F[6:9, 12:15] = -np.eye(3)

        # bias drift: zero dynamics (1st-order Gauss-Markov with very long T)
        # δb_a_dot = 0, δb_g_dot = 0  →  F[9:12, 9:12] = I etc. (unused)

        # ---- Discretize using matrix exponential (unconditionally stable) ----
        # Old code: Phi = np.eye(self.N) + F * dt   ← Euler, eigenvalues can be > 1
        Phi = expm(F * dt)
        P_pred = Phi @ self.P @ Phi.T + self.Q_base * dt

        # Force symmetry
        self.P = 0.5 * (P_pred + P_pred.T)
        self.dx = Phi @ self.dx   # forward-propagate any accumulated error

    # ------------------------------------------------------------------
    # Joseph-form measurement update
    # ------------------------------------------------------------------
    def _joseph_update(self, z_innov: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        """Apply a Joseph-form update.

        z_innov : measured − predicted  (N × 1)
        H       : observation matrix     (N × 15)
        R       : measurement covariance (N × N)
        """
        P = self.P
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        # State correction
        self.dx = self.dx + K @ z_innov

        # Joseph-form (numerically stable)
        I = np.eye(self.N)
        IKH = I - K @ H
        self.P = IKH @ P @ IKH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def _inject_correction(self) -> None:
        """Fold the accumulated dx back into the nominal state and reset dx."""
        dx = self.dx
        if not np.any(np.abs(dx) > 1e-12):
            return

        # δp, δv direct
        self.nominal.p += dx[0:3]
        self.nominal.v += dx[3:6]

        # δθ → small-angle rotation correction (exact quaternion from rotvec)
        dtheta = dx[6:9]
        angle = float(np.linalg.norm(dtheta))
        if angle > 1e-12:
            axis = dtheta / angle
            half = 0.5 * angle
            dq = np.array([np.cos(half), *(np.sin(half) * axis)])
            self.nominal.q = quaternion_normalize(
                quaternion_multiply(self.nominal.q, dq)
            )

        # biases
        self.nominal.b_a += dx[9:12]
        self.nominal.b_g += dx[12:15]

        self.dx = np.zeros(self.N)

    # ------------------------------------------------------------------
    # Sensor-specific updates
    # ------------------------------------------------------------------
    def update_magnetometer(self, mag_body: np.ndarray,
                            earth_field_ned: Optional[np.ndarray] = None,
                            noise_r: float = 0.2) -> None:
        """Correct orientation using known Earth field measured in body frame."""
        if earth_field_ned is None:
            from ..sensors.magnetometer import Magnetometer
            earth_field_ned = Magnetometer.EARTH_FIELD_NED.copy()

        R_BN = body_to_ned_rotation(*self.yaw_pitch_roll)
        R_NB = R_BN.T

        mag_meas = np.asarray(mag_body, dtype=float).ravel()
        mag_pred = R_NB @ earth_field_ned

        z = mag_meas - mag_pred
        # z_innov = [R_NB @ earth_field]× @ δθ   (body-frame left perturbation)
        H_block = _skew(R_NB @ earth_field_ned)
        H = np.zeros((3, self.N))
        H[:, 6:9] = H_block
        R = np.eye(3) * (noise_r ** 2)

        self._joseph_update(z, H, R)
        self._inject_correction()

    def update_pressure(self, depth_meas: float, noise_r: float = 0.01) -> None:
        """Correct depth (p[2]) only.

        ``noise_r`` is the standard deviation of the pressure-derived depth
        measurement in metres.  The default 0.01 m is appropriate for a
        decent marine pressure sensor (σ_P ≈ 100 Pa ≈ 1 cm depth).
        For a sensor with σ_P Pa, use ``noise_r = σ_P / (ρ*g)``.
        """
        depth_pred = self.nominal.p[2]
        z = np.array([depth_meas - depth_pred])

        H = np.zeros((1, self.N))
        H[0, 2] = 1.0
        R = np.array([[noise_r ** 2]])

        self._joseph_update(z, H, R)
        self._inject_correction()

    def update_dvl(self, v_body_meas: np.ndarray, noise_r: float = 0.02) -> None:
        """Correct velocity using a DVL body-frame velocity measurement.

        Measurement model
        -----------------
            v_body = R_NB · v_ned

        Linearising with the body-frame right perturbation
        ``R_BN_true = R_BN_nom · exp([δθ]×)`` (so ``R_NB_true ≈ (I − [δθ]×)·R_NB``):

            v_body ≈ R_NB·v_ned + R_NB·δv + [R_NB·v_ned]× · δθ

        hence

            H = [0₃  R_NB  skew(R_NB·v_ned)  0₃  0₃]

        ``noise_r`` is the one-sigma velocity noise in m/s.  This update is what
        makes horizontal position observable; without it the position error
        grows as ½·b_a·t² and is unbounded.
        """
        R_NB = ned_to_body_rotation(*self.yaw_pitch_roll)
        v_body_pred = R_NB @ self.nominal.v

        z = np.asarray(v_body_meas, dtype=float).ravel() - v_body_pred

        H = np.zeros((3, self.N))
        H[:, 3:6] = R_NB
        H[:, 6:9] = _skew(v_body_pred)
        R = np.eye(3) * (noise_r ** 2)

        self._joseph_update(z, H, R)
        self._inject_correction()

    def update_accelerometer(self, accel_body: np.ndarray,
                             gyro_body: Optional[np.ndarray] = None,
                             noise_r: float = 0.5) -> None:
        """Tilt aid from the direction of the measured specific force.

        A stationary accelerometer measures ``f_body = R_NB @ [0, 0, g]``,
        so the *unit* direction of the reading is the body-frame direction
        of "down".  Comparing it against the predicted direction observes
        roll and pitch without caring about the magnitude of any linear
        acceleration (only its direction matters).

        During a constant-speed turn the specific force also contains the
        centripetal term ``ω × v_body``.  When ``gyro_body`` is supplied we
        subtract it from the prediction, so a coordinated turn no longer
        masquerades as a tilt error.

        ``noise_r`` is a one-sigma angular-equivalent noise in m/s²
        (``g * angle``); the default 0.5 m/s² ≈ 3° is deliberately loose so
        that genuine accelerations do not fight the gyro.
        """
        a = np.asarray(accel_body, dtype=float).ravel()
        norm = float(np.linalg.norm(a))
        if norm < 1e-9:
            return

        R_NB = ned_to_body_rotation(*self.yaw_pitch_roll)

        # Predicted specific force: gravity + (optional) centripetal term.
        f_pred = R_NB @ np.array([0.0, 0.0, self.g])
        if gyro_body is not None:
            omega = np.asarray(gyro_body, dtype=float).ravel() - self.nominal.b_g
            f_pred = f_pred + np.cross(omega, R_NB @ self.nominal.v)

        f_norm = float(np.linalg.norm(f_pred))
        if f_norm < 1e-9:
            return
        f_dir = f_pred / f_norm

        z = self.g * (a / norm - f_dir)

        # H = g · [f_dir]× for the body-frame attitude error.
        H = np.zeros((3, self.N))
        H[:, 6:9] = self.g * _skew(f_dir)
        R = np.eye(3) * (noise_r ** 2)

        self._joseph_update(z, H, R)
        self._inject_correction()

    # ------------------------------------------------------------------
    # Full run — integrate over a simulator history
    # ------------------------------------------------------------------
    def run(self, sim_history, dt_filter: float = 0.01) -> list[NominalState]:
        """Feed simulator sensor readings into the filter, return state history.

        ``dt_filter`` = nominal IMU propagation rate (Hz).  Sensor updates happen
        at their native rates whenever a reading is present in the log.

        Missing samples never abort the run: the filter keeps propagating with
        the last good IMU values and simply skips the corresponding update.
        This is what makes sensor dropouts a graceful degradation rather than a
        crash.
        """
        from ..simulator.simulator import LogEntry

        est: list[NominalState] = []
        last_prop_t = 0.0
        accel = None
        gyro = None

        for entry in sim_history:
            e: LogEntry = entry
            t = e.t

            new_accel = e.readings.get("accelerometer")
            new_gyro = e.readings.get("gyroscope")

            if new_accel is not None:
                accel = new_accel.data
            if new_gyro is not None:
                gyro = new_gyro.data

            # Propagate once per IMU cycle, using the most recent good samples.
            if (new_accel is not None or new_gyro is not None) \
                    and accel is not None and gyro is not None:
                dt = t - last_prop_t
                if dt > 0:
                    self.propagate(dt, accel, gyro)
                    last_prop_t = t

            # ---- Sensor updates (each at its own rate) ----
            if new_accel is not None:
                self.update_accelerometer(new_accel.data, gyro_body=gyro)

            if "magnetometer" in e.readings:
                self.update_magnetometer(e.readings["magnetometer"].data)

            if "pressure" in e.readings:
                from ..core.frames import pressure_to_depth
                depth = float(pressure_to_depth(e.readings["pressure"].data[0]))
                self.update_pressure(depth)

            if "dvl" in e.readings:
                self.update_dvl(e.readings["dvl"].data)

            est.append(self.nominal.copy())

        # Inject any leftover correction at the end
        self._inject_correction()
        return est
