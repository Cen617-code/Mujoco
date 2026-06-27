"""First-pass body pitch balance controller for the wheeled biped."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import mujoco
import numpy as np

from scripts.pd_control import (
    JointControlMap,
    clip_targets_to_joint_limits,
    home_targets,
)


LEG_JOINTS = {
    "left_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "right_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
}
WHEEL_JOINTS = {"left_wheel_joint", "right_wheel_joint"}
WHEEL_ACTUATOR_SIGNS = {
    "left_wheel_joint": 1.0,
    "right_wheel_joint": -1.0,
}


@dataclass(frozen=True)
class BalanceConfig:
    pitch_target: float = 0.0
    pitch_rate_target: float = 0.0
    x_target: float | None = None
    x_velocity_target: float = 0.0
    kp_pitch: float = 35.0
    kd_pitch: float = 4.0
    kx: float = 0.0
    kv: float = 1.0
    leg_kp: float = 20.0
    leg_kd: float = 1.0


@dataclass(frozen=True)
class BalanceState:
    pitch: float
    pitch_rate: float
    x: float
    x_velocity: float
    wheel_torque: float


def quat_to_pitch(quat_wxyz) -> float:
    w, x, y, z = [float(value) for value in quat_wxyz]
    value = 2.0 * (w * y - z * x)
    return float(np.arcsin(np.clip(value, -1.0, 1.0)))


# The base free joint occupies qpos[0:7] as [x, y, z, qw, qx, qy, qz] and
# qvel[0:6] as [vx, vy, vz, wx, wy, wz] in this model. Positive pitch is a
# positive rotation about the base/world Y axis near the upright pose.
# base_pitch_rate() uses angular Y velocity (qvel[4]) as a near-upright,
# first-pass balance convention.
def base_pitch(data: mujoco.MjData) -> float:
    return quat_to_pitch(data.qpos[3:7])


def base_pitch_rate(data: mujoco.MjData) -> float:
    return float(data.qvel[4])


def default_balance_config() -> BalanceConfig:
    return BalanceConfig()


def _actuator_limits(model: mujoco.MjModel, entry: JointControlMap) -> tuple[float, float]:
    lower, upper = model.actuator_ctrlrange[entry.actuator_id]
    return float(lower), float(upper)


def compute_balance_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_map: list[JointControlMap],
    config: BalanceConfig | None = None,
    leg_targets: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, BalanceState]:
    config = config or default_balance_config()
    targets = home_targets(model, joint_map)
    if leg_targets:
        targets.update(leg_targets)
    targets = clip_targets_to_joint_limits(model, joint_map, targets)

    ctrl = np.zeros(model.nu, dtype=float)
    for entry in joint_map:
        if entry.joint_name not in LEG_JOINTS:
            continue
        q = float(data.qpos[entry.qposadr])
        qdot = float(data.qvel[entry.dofadr])
        tau = config.leg_kp * (targets[entry.joint_name] - q) - config.leg_kd * qdot
        lower, upper = _actuator_limits(model, entry)
        ctrl[entry.actuator_id] = np.clip(tau, lower, upper)

    pitch = base_pitch(data)
    pitch_rate = base_pitch_rate(data)
    x = float(data.qpos[0])
    x_velocity = float(data.qvel[0])
    x_target = x if config.x_target is None else float(config.x_target)
    tau_balance = (
        config.kp_pitch * (config.pitch_target - pitch)
        + config.kd_pitch * (config.pitch_rate_target - pitch_rate)
        + config.kx * (x_target - x)
        + config.kv * (config.x_velocity_target - x_velocity)
    )
    wheel_torque = 0.0
    for entry in joint_map:
        if entry.joint_name not in WHEEL_JOINTS:
            continue
        lower, upper = _actuator_limits(model, entry)
        # The left and right wheel joint axes are mirrored in world coordinates
        # (+Y for left, -Y for right near the home pose). Opposite actuator
        # signs therefore produce the same physical pitch-balancing wheel
        # torque direction.
        signed_tau = WHEEL_ACTUATOR_SIGNS[entry.joint_name] * tau_balance
        ctrl[entry.actuator_id] = np.clip(signed_tau, lower, upper)
        wheel_torque = max(wheel_torque, abs(float(ctrl[entry.actuator_id])))

    state = BalanceState(
        pitch=float(pitch),
        pitch_rate=float(pitch_rate),
        x=x,
        x_velocity=x_velocity,
        wheel_torque=wheel_torque,
    )
    return ctrl, state


def apply_balance_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_map: list[JointControlMap],
    config: BalanceConfig | None = None,
    leg_targets: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, BalanceState]:
    ctrl, state = compute_balance_control(model, data, joint_map, config, leg_targets)
    data.ctrl[:] = ctrl
    return ctrl, state
