from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from scripts.convert_urdf_to_mjcf import convert_urdf


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "8dof_URDF" / "urdf" / "robot.urdf"
OUTPUT = ROOT / "8dof_URDF" / "mjcf" / "robot.xml"

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

EXPECTED_TORQUE_LIMITS = {
    "left_roll_joint": (-20.0, 20.0),
    "right_roll_joint": (-20.0, 20.0),
    "left_hip_pitch_joint": (-30.0, 30.0),
    "right_hip_pitch_joint": (-30.0, 30.0),
    "left_knee_joint": (-30.0, 30.0),
    "right_knee_joint": (-30.0, 30.0),
    "left_wheel_joint": (-10.0, 10.0),
    "right_wheel_joint": (-10.0, 10.0),
}


@pytest.fixture(scope="session")
def model() -> mujoco.MjModel:
    convert_urdf(SOURCE, OUTPUT)
    return mujoco.MjModel.from_xml_path(str(OUTPUT))


def name(model: mujoco.MjModel, objtype, objid: int) -> str:
    return mujoco.mj_id2name(model, objtype, objid) or ""


def test_hip_pitch_limits_are_overridden(model):
    for joint_name in ["left_hip_pitch_joint", "right_hip_pitch_joint"]:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        assert joint_id >= 0
        assert model.jnt_limited[joint_id]
        assert np.allclose(model.jnt_range[joint_id], [-1.22, 0.87])


def test_eight_named_torque_motors_are_bound_to_expected_joints(model):
    assert model.nu == 8
    actuator_names = [name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    assert actuator_names == [f"{joint}_motor" for joint in CONTROLLED_JOINTS]
    for actuator_id, joint_name in enumerate(CONTROLLED_JOINTS):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        assert joint_id >= 0
        assert model.actuator_trnid[actuator_id, 0] == joint_id
        assert np.allclose(model.actuator_ctrlrange[actuator_id], EXPECTED_TORQUE_LIMITS[joint_name])
        assert model.actuator_ctrllimited[actuator_id]


def test_base_weld_equality_exists_for_analysis(model):
    equality_names = [name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i) for i in range(model.neq)]
    assert "fixed_base_weld" in equality_names
    equality_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "fixed_base_weld")
    assert equality_id >= 0


def test_robot_xml_contains_motor_and_weld_elements(model):
    root = ET.parse(OUTPUT).getroot()
    motors = root.findall("actuator/motor")
    assert [motor.get("name") for motor in motors] == [f"{joint}_motor" for joint in CONTROLLED_JOINTS]
    welds = root.findall("equality/weld")
    assert any(weld.get("name") == "fixed_base_weld" for weld in welds)
