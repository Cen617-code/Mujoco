"""PD joint-control helpers for the 8-DOF wheeled biped MuJoCo model.

这一层只关心“关节目标 -> actuator torque”的通用映射，不处理机身平衡逻辑。
平衡控制会复用这里的 joint/actuator 映射、限位裁剪和固定基座 weld 开关。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import mujoco
import numpy as np


CONTROLLED_JOINTS = [
    "left_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_wheel_joint",
    "right_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_wheel_joint",
]


@dataclass(frozen=True)
class JointControlMap:
    """一个受控关节在 MuJoCo model/data 中对应的索引集合。"""

    joint_name: str
    joint_id: int
    qposadr: int
    dofadr: int
    actuator_id: int


def _name(model: mujoco.MjModel, objtype, objid: int) -> str:
    return mujoco.mj_id2name(model, objtype, objid) or ""


def build_joint_map(model: mujoco.MjModel) -> list[JointControlMap]:
    """按 CONTROLLED_JOINTS 顺序建立关节到 actuator/qpos/qvel 的映射。"""
    entries: list[JointControlMap] = []
    for actuator_id, joint_name in enumerate(CONTROLLED_JOINTS):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Model is missing controlled joint {joint_name!r}")
        if actuator_id >= model.nu:
            raise ValueError(f"Model has no actuator slot {actuator_id} for {joint_name!r}")
        actuator_name = _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        expected_actuator = f"{joint_name}_motor"
        # 控制器默认 actuator 顺序和 CONTROLLED_JOINTS 完全一致；
        # 如果 XML 顺序被改，这里会立刻报错，避免力矩打到错误关节。
        if actuator_name != expected_actuator:
            raise ValueError(
                f"Expected actuator {expected_actuator!r} at slot {actuator_id}, "
                f"found {actuator_name!r}"
            )
        if model.actuator_trnid[actuator_id, 0] != joint_id:
            raise ValueError(f"Actuator {actuator_name!r} is not bound to joint {joint_name!r}")
        entries.append(
            JointControlMap(
                joint_name=joint_name,
                joint_id=joint_id,
                qposadr=int(model.jnt_qposadr[joint_id]),
                dofadr=int(model.jnt_dofadr[joint_id]),
                actuator_id=actuator_id,
            )
        )
    return entries


def home_targets(model: mujoco.MjModel, joint_map: list[JointControlMap]) -> dict[str, float]:
    """读取 MuJoCo 默认 qpos0 作为名义站立/零位目标。"""
    return {entry.joint_name: float(model.qpos0[entry.qposadr]) for entry in joint_map}


def clip_targets_to_joint_limits(
    model: mujoco.MjModel,
    joint_map: list[JointControlMap],
    targets: Mapping[str, float],
) -> dict[str, float]:
    """把目标角裁剪到关节限位；连续轮关节不裁剪。"""
    clipped: dict[str, float] = {}
    for entry in joint_map:
        value = float(targets.get(entry.joint_name, model.qpos0[entry.qposadr]))
        if model.jnt_limited[entry.joint_id]:
            lower, upper = model.jnt_range[entry.joint_id]
            value = float(np.clip(value, lower, upper))
        clipped[entry.joint_name] = value
    return clipped


def default_pd_gains(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    leg_wn: float = 8.0,
    wheel_wn: float = 12.0,
    zeta: float = 1.0,
) -> dict[str, tuple[float, float]]:
    """根据质量矩阵对角项生成一组保守的二阶 PD 增益。"""
    mujoco.mj_forward(model, data)
    mass_matrix = np.zeros((model.nv, model.nv), dtype=float)
    mujoco.mj_fullM(model, data, mass_matrix)
    gains: dict[str, tuple[float, float]] = {}
    for entry in build_joint_map(model):
        inertia = max(float(mass_matrix[entry.dofadr, entry.dofadr]), 1e-6)
        # 轮子响应可以稍快一些；腿部关节更保守，主要用于姿态保持。
        wn = wheel_wn if "wheel" in entry.joint_name else leg_wn
        kp = inertia * wn * wn
        kd = 2.0 * zeta * inertia * wn
        gains[entry.joint_name] = (float(kp), float(kd))
    return gains


def compute_pd_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_map: list[JointControlMap],
    targets: Mapping[str, float],
    gains: Mapping[str, tuple[float, float]],
) -> np.ndarray:
    """计算 8 个 actuator 的 PD 力矩，并按 ctrlrange 饱和。"""
    clipped_targets = clip_targets_to_joint_limits(model, joint_map, targets)
    ctrl = np.zeros(model.nu, dtype=float)
    for entry in joint_map:
        kp, kd = gains[entry.joint_name]
        q = float(data.qpos[entry.qposadr])
        qdot = float(data.qvel[entry.dofadr])
        tau = float(kp) * (clipped_targets[entry.joint_name] - q) - float(kd) * qdot
        lower, upper = model.actuator_ctrlrange[entry.actuator_id]
        ctrl[entry.actuator_id] = np.clip(tau, lower, upper)
    return ctrl


def apply_pd_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_map: list[JointControlMap],
    targets: Mapping[str, float],
    gains: Mapping[str, tuple[float, float]],
) -> np.ndarray:
    ctrl = compute_pd_control(model, data, joint_map, targets, gains)
    data.ctrl[:] = ctrl
    return ctrl


def set_base_weld_active(model: mujoco.MjModel, data: mujoco.MjData, active: bool) -> None:
    """打开/关闭 fixed_base_weld，用于在同一个模型中切换固定基座分析。"""
    equality_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "fixed_base_weld")
    if equality_id < 0:
        raise ValueError("Model is missing equality weld 'fixed_base_weld'")
    data.eq_active[equality_id] = 1 if active else 0
    mujoco.mj_forward(model, data)
