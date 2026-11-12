from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .state import VehicleState, euler_to_quaternion

STATIONARY_DURATION = 10.0
ACCEL_DURATION = 10.0
NORTH_CRUISE_DURATION = 15.0
TURN_DURATION = 30.0
EAST_CRUISE_DURATION = 10.0
DESCENT_DURATION = 30.0
HOLD_DURATION = 15.0

CRUISE_SPEED = 0.8
TURN_ANGLE = np.pi / 2
DESCENT_DEPTH_CHANGE = 4.0

DEFAULT_INITIAL_DEPTH = 20.0
DEFAULT_INITIAL_YAW = 0.0

TOTAL_DURATION = (STATIONARY_DURATION + ACCEL_DURATION + NORTH_CRUISE_DURATION
                  + TURN_DURATION + EAST_CRUISE_DURATION + DESCENT_DURATION
                  + HOLD_DURATION)


@dataclass(frozen=True)
class Stage:
    index: int
    name: str
    t_start: float
    t_end: float

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


def build_stages() -> list[Stage]:
    bounds = [0.0,
              STATIONARY_DURATION,
              STATIONARY_DURATION + ACCEL_DURATION,
              STATIONARY_DURATION + ACCEL_DURATION + NORTH_CRUISE_DURATION,
              STATIONARY_DURATION + ACCEL_DURATION + NORTH_CRUISE_DURATION + TURN_DURATION,
              (STATIONARY_DURATION + ACCEL_DURATION + NORTH_CRUISE_DURATION + TURN_DURATION
               + EAST_CRUISE_DURATION),
              (STATIONARY_DURATION + ACCEL_DURATION + NORTH_CRUISE_DURATION + TURN_DURATION
               + EAST_CRUISE_DURATION + DESCENT_DURATION),
              TOTAL_DURATION]
    names = ["Stationary", "Acceleration", "North cruise", "Turning",
             "East cruise", "Descending", "Hold"]
    return [Stage(i, name, bounds[i], bounds[i + 1]) for i, name in enumerate(names)]


def _smootherstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return 6 * x ** 5 - 15 * x ** 4 + 10 * x ** 3


def _smootherstep_d1(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return 30 * x ** 2 * (1 - x) ** 2


def _smootherstep_d2(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return 60 * x * (2 * x - 1) * (x - 1)


def _smootherstep_integral(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x ** 6 - 3 * x ** 5 + 2.5 * x ** 4


class Trajectory:
    def __init__(self,
                 initial_depth: float = DEFAULT_INITIAL_DEPTH,
                 initial_yaw: float = DEFAULT_INITIAL_YAW,
                 cruise_speed: float = CRUISE_SPEED):
        self.initial_depth = float(initial_depth)
        self.initial_yaw = float(initial_yaw)
        self.cruise_speed = float(cruise_speed)
        if self.cruise_speed <= 0.0:
            raise ValueError("cruise_speed must be positive")

        self.stages_list = build_stages()
        self.duration = self.stages_list[-1].t_end

        self.turn_yaw_rate = TURN_ANGLE / TURN_DURATION
        self.turn_radius = self.cruise_speed / self.turn_yaw_rate
        self.yaw_after_turn = self.initial_yaw + TURN_ANGLE

        self._north_dir = np.array([np.cos(self.initial_yaw), np.sin(self.initial_yaw), 0.0])
        self._east_dir = np.array([np.cos(self.yaw_after_turn), np.sin(self.yaw_after_turn), 0.0])

        accel_distance = self.cruise_speed * ACCEL_DURATION * 0.5
        self._p_accel_end = np.array([0.0, 0.0, self.initial_depth]) + self._north_dir * accel_distance

        self._p_north_end = (self._p_accel_end
                             + self._north_dir * self.cruise_speed * NORTH_CRUISE_DURATION)

        dn = self.turn_radius * (np.sin(self.yaw_after_turn) - np.sin(self.initial_yaw))
        de = self.turn_radius * (np.cos(self.initial_yaw) - np.cos(self.yaw_after_turn))
        self._p_turn_end = self._p_north_end + np.array([dn, de, 0.0])

        self._p_east_end = (self._p_turn_end
                            + self._east_dir * self.cruise_speed * EAST_CRUISE_DURATION)

        self._p_descent_end = (self._p_east_end
                               + self._east_dir * self.cruise_speed * DESCENT_DURATION)
        self._p_descent_end[2] = self.initial_depth + DESCENT_DEPTH_CHANGE

    def stages(self) -> list[Stage]:
        return list(self.stages_list)

    @property
    def final_depth(self) -> float:
        return self.initial_depth + DESCENT_DEPTH_CHANGE

    def state_at(self, t: float) -> VehicleState:
        v = self._values(t)
        quaternion = euler_to_quaternion(v["yaw"], v["pitch"], v["roll"])
        angular_velocity = np.array([v["roll_rate"], v["pitch_rate"], v["yaw_rate"]])
        return VehicleState(
            t=float(t),
            position=v["position"].copy(),
            velocity=v["velocity"].copy(),
            acceleration_ned=v["acceleration"].copy(),
            quaternion=quaternion,
            angular_velocity=angular_velocity,
        )

    def sample(self, dt: float = 0.01) -> list[VehicleState]:
        if dt <= 0:
            raise ValueError("dt must be positive")
        states = []
        n = int(np.floor(self.duration / dt + 1e-9))
        for k in range(n + 1):
            states.append(self.state_at(k * dt))
        return states

    @property
    def position_fn(self) -> Callable[[float], np.ndarray]:
        return lambda t: self._values(t)["position"]

    @property
    def velocity_fn(self) -> Callable[[float], np.ndarray]:
        return lambda t: self._values(t)["velocity"]

    @property
    def heading_fn(self) -> Callable[[float], float]:
        return lambda t: self._values(t)["yaw"]

    def _stage_at(self, t: float) -> Stage:
        for i, s in enumerate(self.stages_list):
            if t < s.t_end:
                return s
        return self.stages_list[-1]

    def _values(self, t: float) -> dict:
        t = max(0.0, min(float(t), self.duration))
        stage = self._stage_at(t)
        lt = t - stage.t_start

        pos = np.zeros(3)
        vel = np.zeros(3)
        acc = np.zeros(3)
        yaw = self.initial_yaw
        pitch = 0.0
        roll = 0.0
        yaw_rate = 0.0
        pitch_rate = 0.0

        idx = stage.index
        V = self.cruise_speed

        if idx == 0:
            pos = np.array([0.0, 0.0, self.initial_depth])

        elif idx == 1:
            x = lt / ACCEL_DURATION
            pos = np.array([0.0, 0.0, self.initial_depth]) \
                + self._north_dir * (V * ACCEL_DURATION * _smootherstep_integral(x))
            vel = self._north_dir * (V * _smootherstep(x))
            acc = self._north_dir * (V * _smootherstep_d1(x) / ACCEL_DURATION)

        elif idx == 2:
            pos = self._p_accel_end + self._north_dir * (V * lt)
            vel = self._north_dir * V

        elif idx == 3:
            omega = self.turn_yaw_rate
            r = self.turn_radius
            yaw = self.initial_yaw + omega * lt
            dn = r * (np.sin(yaw) - np.sin(self.initial_yaw))
            de = r * (np.cos(self.initial_yaw) - np.cos(yaw))
            pos = self._p_north_end + np.array([dn, de, 0.0])
            vel = np.array([V * np.cos(yaw), V * np.sin(yaw), 0.0])
            acc = np.array([-V * omega * np.sin(yaw), V * omega * np.cos(yaw), 0.0])
            yaw_rate = omega

        elif idx == 4:
            yaw = self.yaw_after_turn
            pos = self._p_turn_end + self._east_dir * (V * lt)
            vel = self._east_dir * V

        elif idx == 5:
            yaw = self.yaw_after_turn
            x = lt / DESCENT_DURATION
            depth = self.initial_depth + DESCENT_DEPTH_CHANGE * _smootherstep(x)
            vz = DESCENT_DEPTH_CHANGE * _smootherstep_d1(x) / DESCENT_DURATION
            az = DESCENT_DEPTH_CHANGE * _smootherstep_d2(x) / DESCENT_DURATION ** 2

            pos = self._p_east_end + self._east_dir * (V * lt)
            pos[2] = depth
            vel = self._east_dir * V + np.array([0.0, 0.0, vz])
            acc = np.array([0.0, 0.0, az])

            pitch = float(np.arctan2(-vz, V))
            pitch_rate = float(-az * V / (V ** 2 + vz ** 2))

        else:
            yaw = self.yaw_after_turn
            pos = self._p_descent_end + self._east_dir * (V * lt)
            vel = self._east_dir * V

        return {
            "position": pos,
            "velocity": vel,
            "acceleration": acc,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "yaw_rate": yaw_rate,
            "pitch_rate": pitch_rate,
            "roll_rate": 0.0,
        }
