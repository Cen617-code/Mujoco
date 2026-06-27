from pathlib import Path

import mujoco
import numpy as np
import pytest

from scripts.convert_urdf_to_mjcf import convert_urdf
from scripts.pd_control import build_joint_map
from scripts.balance_control import (
    BalanceConfig,
    apply_balance_control,
    base_pitch,
    compute_balance_control,
    quat_to_pitch,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "8dof_URDF" / "urdf" / "robot.urdf"
MODEL_XML = ROOT / "8dof_URDF" / "mjcf" / "robot.xml"


@pytest.fixture(scope="session")
def model() -> mujoco.MjModel:
    convert_urdf(SOURCE, MODEL_XML)
    return mujoco.MjModel.from_xml_path(str(MODEL_XML))


def quat_y_rotation(angle: float) -> np.ndarray:
    return np.array([np.cos(angle / 2.0), 0.0, np.sin(angle / 2.0), 0.0])


def test_quat_to_pitch_extracts_small_y_axis_rotation():
    assert quat_to_pitch(quat_y_rotation(0.2)) == pytest.approx(0.2, abs=1e-9)
    assert quat_to_pitch(quat_y_rotation(-0.2)) == pytest.approx(-0.2, abs=1e-9)


def test_base_pitch_reads_free_joint_quaternion(model):
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[3:7] = quat_y_rotation(0.15)
    mujoco.mj_forward(model, data)
    assert base_pitch(data) == pytest.approx(0.15, abs=1e-9)


def test_balance_torque_direction_and_saturation(model):
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.qpos[3:7] = quat_y_rotation(0.2)
    mujoco.mj_forward(model, data)
    joint_map = build_joint_map(model)
    config = BalanceConfig(kp_pitch=100.0, kd_pitch=0.0, kx=0.0, kv=0.0)
    ctrl, state = compute_balance_control(model, data, joint_map, config)
    left_wheel = next(entry for entry in joint_map if entry.joint_name == "left_wheel_joint")
    right_wheel = next(entry for entry in joint_map if entry.joint_name == "right_wheel_joint")
    assert state.pitch == pytest.approx(0.2, abs=1e-9)
    assert ctrl[left_wheel.actuator_id] == pytest.approx(-10.0)
    assert ctrl[right_wheel.actuator_id] == pytest.approx(-10.0)


def test_leg_joints_receive_posture_pd_torques(model):
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    joint_map = build_joint_map(model)
    left_knee = next(entry for entry in joint_map if entry.joint_name == "left_knee_joint")
    data.qpos[left_knee.qposadr] = 0.05
    mujoco.mj_forward(model, data)
    config = BalanceConfig(leg_kp=20.0, leg_kd=0.0, kp_pitch=0.0, kd_pitch=0.0)
    ctrl, state = compute_balance_control(model, data, joint_map, config)
    assert state.pitch == pytest.approx(0.0, abs=1e-9)
    assert ctrl[left_knee.actuator_id] < 0.0


def test_apply_balance_control_writes_model_ctrl(model):
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    joint_map = build_joint_map(model)
    ctrl, state = apply_balance_control(model, data, joint_map, BalanceConfig())
    assert ctrl.shape == (model.nu,)
    assert np.allclose(data.ctrl, ctrl)
    assert np.isfinite(state.pitch)
