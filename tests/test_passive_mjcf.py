from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from scripts.convert_urdf_to_mjcf import convert_urdf

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "8dof_URDF" / "urdf" / "robot.urdf"
OUTPUT = ROOT / "8dof_URDF" / "mjcf" / "robot.xml"


@pytest.fixture(scope="session")
def model() -> mujoco.MjModel:
    convert_urdf(SOURCE, OUTPUT)
    return mujoco.MjModel.from_xml_path(str(OUTPUT))


def object_names(model, object_type, count):
    return [mujoco.mj_id2name(model, object_type, i) or "" for i in range(count)]


def mesh_geom_min_z(model, data, geom_id):
    assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
    mesh_id = model.geom_dataid[geom_id]
    assert mesh_id >= 0
    start = model.mesh_vertadr[mesh_id]
    count = model.mesh_vertnum[mesh_id]
    vertices = model.mesh_vert[start:start + count]
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    world_vertices = vertices @ rotation.T + data.geom_xpos[geom_id]
    return float(world_vertices[:, 2].min())


def test_structure_and_corrected_names(model):
    joint_names = object_names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    body_names = object_names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    mesh_names = object_names(model, mujoco.mjtObj.mjOBJ_MESH, model.nmesh)
    assert model.njnt == 9
    assert model.nq == 15
    assert model.nv == 14
    assert model.nu == 8
    assert list(model.jnt_type).count(mujoco.mjtJoint.mjJNT_FREE) == 1
    assert list(model.jnt_type).count(mujoco.mjtJoint.mjJNT_HINGE) == 8
    expected_hinges = {
        "left_roll_joint", "left_hip_pitch_joint", "left_knee_joint", "left_wheel_joint",
        "right_roll_joint", "right_hip_pitch_joint", "right_knee_joint", "right_wheel_joint",
    }
    assert set(joint_names) == {"base_freejoint", *expected_hinges}
    geom_names = object_names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)
    assert not any("yaw" in name for name in joint_names + body_names + geom_names + mesh_names)
    for name in ["left_wheel_joint", "right_wheel_joint"]:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert not model.jnt_limited[joint_id]


def test_home_keyframe_matches_default_pose(model):
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    assert key_id >= 0
    assert np.allclose(model.key_qpos[key_id], model.qpos0)
    assert np.allclose(model.key_qpos[key_id, 3:7], [1, 0, 0, 0])
    assert np.allclose(model.key_qpos[key_id, 7:], 0)


def test_roll_joint_world_axes_point_along_positive_x(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for name in ["left_roll_joint", "right_roll_joint"]:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert np.allclose(data.xaxis[joint_id], [1, 0, 0], atol=1e-5)


def test_contact_exclusions_are_exact(model):
    root = ET.parse(OUTPUT).getroot()
    actual = {
        frozenset((element.get("body1"), element.get("body2")))
        for element in root.findall("contact/exclude")
    }
    expected = {
        frozenset(("base_link", "left_roll_link")),
        frozenset(("left_roll_link", "left_hip_pitch_link")),
        frozenset(("left_hip_pitch_link", "left_knee_link")),
        frozenset(("left_knee_link", "left_wheel_link")),
        frozenset(("base_link", "right_roll_link")),
        frozenset(("right_roll_link", "right_hip_pitch_link")),
        frozenset(("right_hip_pitch_link", "right_knee_link")),
        frozenset(("right_knee_link", "right_wheel_link")),
        frozenset(("base_link", "left_hip_pitch_link")),
        frozenset(("base_link", "right_hip_pitch_link")),
    }
    assert actual == expected


def test_wheels_start_at_ground_without_other_penetration(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    wheel_names = ["left_wheel_collision", "right_wheel_collision"]
    for name in wheel_names:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        minimum_z = mesh_geom_min_z(model, data, geom_id)
        assert minimum_z >= -1e-7
        assert minimum_z <= 1e-3
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name == "floor" or name.endswith("_visual"):
            continue
        assert mesh_geom_min_z(model, data, geom_id) >= -1e-3


def test_initial_contacts_are_ground_only(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert data.ncon == 2
    contact_pairs = set()
    for index in range(data.ncon):
        contact = data.contact[index]
        assert floor in (contact.geom1, contact.geom2)
        other = contact.geom2 if contact.geom1 == floor else contact.geom1
        other_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other)
        assert other_name in {"left_wheel_collision", "right_wheel_collision"}
        contact_pairs.add(frozenset(("floor", other_name)))
        assert contact.dist >= -1e-7
    assert contact_pairs == {
        frozenset(("floor", "left_wheel_collision")),
        frozenset(("floor", "right_wheel_collision")),
    }
    assert np.isfinite(data.qacc).all()
    assert np.max(np.abs(data.qacc)) < 100.0


def test_two_second_passive_simulation_is_finite(model):
    data = mujoco.MjData(model)
    data.ctrl[:] = 0
    steps = int(2.0 / model.opt.timestep)
    peak_100ms = 0.0
    peak_2s = 0.0
    for step in range(steps):
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all()
        assert np.isfinite(data.qvel).all()
        velocity = float(np.max(np.abs(data.qvel)))
        peak_2s = max(peak_2s, velocity)
        if step < int(0.1 / model.opt.timestep):
            peak_100ms = max(peak_100ms, velocity)
    assert max(warning.number for warning in data.warning) == 0
    assert peak_100ms < 20.0
    assert peak_2s < 100.0
