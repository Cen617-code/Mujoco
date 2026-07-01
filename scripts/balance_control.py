"""First-pass body pitch balance controller for the wheeled biped.

当前控制器是“原地 pitch 平衡原型”：腿部关节用 PD 保持名义姿态，两个轮子
用差异化符号的力矩调机身俯仰。它还不是完整行走/速度/轨迹控制器。
"""

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
# IMU 名称和 converter 生成的 MJCF sensor 保持一致。
BASE_IMU_GYRO = "base_imu_gyro"
BASE_IMU_ACCEL = "base_imu_accel"
BASE_IMU_QUAT = "base_imu_quat"
DEFAULT_STANDING_HIP_PITCH = -0.15
DEFAULT_STANDING_KNEE = 0.35


@dataclass(frozen=True)
class BalanceConfig:
    """机身平衡控制参数。

    pitch/x 相关增益作用在轮子上；leg_kp/leg_kd 作用在 6 个腿部关节上。
    """

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
    """一次控制计算后用于记录/分析的机身状态摘要。"""

    pitch: float
    pitch_rate: float
    x: float
    x_velocity: float
    wheel_torque: float


def quat_to_pitch(quat_wxyz) -> float:
    """从 wxyz 四元数中提取绕 Y 轴的 pitch 角。"""
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


def sensor_slice(model: mujoco.MjModel, sensor_name: str) -> slice:
    """返回某个 MuJoCo sensor 在 data.sensordata 中的切片位置。"""
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sensor_id < 0:
        raise ValueError(f"Model is missing sensor {sensor_name!r}")
    start = int(model.sensor_adr[sensor_id])
    stop = start + int(model.sensor_dim[sensor_id])
    return slice(start, stop)


def has_base_imu(model: mujoco.MjModel) -> bool:
    """判断模型是否提供当前控制器需要的 IMU 姿态和角速度。"""
    return (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, BASE_IMU_GYRO) >= 0
        and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, BASE_IMU_QUAT) >= 0
    )


def base_imu_quat(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """读取 IMU 输出的 base/site 姿态四元数。"""
    values = data.sensordata[sensor_slice(model, BASE_IMU_QUAT)]
    if values.shape[0] != 4:
        raise ValueError("base_imu_quat must have dimension 4")
    return np.asarray(values, dtype=float)


def base_imu_gyro(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """读取 IMU 陀螺仪角速度。"""
    values = data.sensordata[sensor_slice(model, BASE_IMU_GYRO)]
    if values.shape[0] != 3:
        raise ValueError("base_imu_gyro must have dimension 3")
    return np.asarray(values, dtype=float)


def base_pitch_from_imu(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """优先用于闭环控制的 pitch 估计。"""
    return quat_to_pitch(base_imu_quat(model, data))


def base_pitch_rate_from_imu(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """读取 IMU gyro 的 Y 分量，匹配当前 near-upright pitch-rate 约定。"""
    return float(base_imu_gyro(model, data)[1])


def standing_leg_targets(
    hip_pitch: float = DEFAULT_STANDING_HIP_PITCH,
    knee: float = DEFAULT_STANDING_KNEE,
) -> dict[str, float]:
    """Return symmetric fixed leg targets for robust-standing attempts."""
    return {
        "left_hip_pitch_joint": float(hip_pitch),
        "right_hip_pitch_joint": float(hip_pitch),
        "left_knee_joint": float(knee),
        "right_knee_joint": float(knee),
    }


def default_balance_config() -> BalanceConfig:
    return BalanceConfig()


def default_standing_config() -> BalanceConfig:
    """Return the current best explicit robust-standing controller config."""
    return BalanceConfig(
        pitch_target=0.0,
        pitch_rate_target=0.0,
        x_target=None,
        x_velocity_target=0.0,
        kp_pitch=35.0,
        kd_pitch=4.0,
        kx=0.0,
        kv=1.0,
        leg_kp=20.0,
        leg_kd=1.0,
    )


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
    """计算腿部姿态保持力矩和左右轮平衡力矩。"""
    config = config or default_balance_config()
    targets = home_targets(model, joint_map)
    if leg_targets:
        targets.update(leg_targets)
    targets = clip_targets_to_joint_limits(model, joint_map, targets)

    ctrl = np.zeros(model.nu, dtype=float)
    for entry in joint_map:
        if entry.joint_name not in LEG_JOINTS:
            continue
        # 腿部只做关节姿态保持，不直接参与机身位置/速度闭环。
        q = float(data.qpos[entry.qposadr])
        qdot = float(data.qvel[entry.dofadr])
        tau = config.leg_kp * (targets[entry.joint_name] - q) - config.leg_kd * qdot
        lower, upper = _actuator_limits(model, entry)
        ctrl[entry.actuator_id] = np.clip(tau, lower, upper)

    if has_base_imu(model):
        # 接近真实机器人接口：有 IMU 时不直接从 freejoint 偷看姿态。
        pitch = base_pitch_from_imu(model, data)
        pitch_rate = base_pitch_rate_from_imu(model, data)
    else:
        pitch = base_pitch(data)
        pitch_rate = base_pitch_rate(data)
    x = float(data.qpos[0])
    x_velocity = float(data.qvel[0])
    x_target = x if config.x_target is None else float(config.x_target)
    tau_balance = (
        # 轮子控制量 = pitch PD + 可选的水平位置/速度反馈。
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
