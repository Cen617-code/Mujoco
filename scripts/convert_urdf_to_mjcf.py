"""Convert the project's eight-joint URDF to a native MuJoCo model.

这个脚本是当前项目的“模型生成入口”：所有手工修正过的 MuJoCo 设定
（free base、关节命名、力矩电机、IMU、初始落地高度等）都从这里生成到
``8dof_URDF/mjcf/robot.xml``，避免直接改 XML 后下次转换被覆盖。
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_LINKS = {
    "base_link",
    "left_yaw_link",
    "left_hip_pitch_link",
    "left_knee_link",
    "left_wheel_link",
    "right_yaw_link",
    "right_hip_pitch_link",
    "right_knee_link",
    "right_wheel_link",
}
EXPECTED_JOINTS = {
    "left_yaw_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_wheel_joint",
    "right_yaw_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_wheel_joint",
}
# MuJoCo actuator 的顺序会被 Python 控制器直接使用，因此这里固定顺序。
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

# 机械模型中 hip pitch 的可用范围；覆盖 URDF 原始 limit。
HIP_PITCH_RANGE = (-1.22, 0.87)

# 每个关节电机的力矩限幅，写入 MJCF 的 actuator ctrlrange。
TORQUE_LIMITS = {
    "left_roll_joint": (-20.0, 20.0),
    "right_roll_joint": (-20.0, 20.0),
    "left_hip_pitch_joint": (-30.0, 30.0),
    "right_hip_pitch_joint": (-30.0, 30.0),
    "left_knee_joint": (-30.0, 30.0),
    "right_knee_joint": (-30.0, 30.0),
    "left_wheel_joint": (-10.0, 10.0),
    "right_wheel_joint": (-10.0, 10.0),
}


def corrected_name(name: str) -> str:
    """把早期 URDF 中误写的 yaw 语义统一改成 roll。"""
    return name.replace("yaw", "roll")


def _numbers(value: str | None, default: str) -> list[float]:
    return [float(item) for item in (value or default).split()]


def _format(values) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _origin(element: ET.Element | None) -> tuple[list[float], list[float]]:
    """读取 URDF origin，并把 rpy 转为 MuJoCo 使用的 wxyz 四元数。"""
    if element is None:
        return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
    xyz = _numbers(element.get("xyz"), "0 0 0")
    rpy = _numbers(element.get("rpy"), "0 0 0")
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    quaternion = np.array(
        [
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr,
        ],
        dtype=float,
    )
    quaternion /= np.linalg.norm(quaternion)
    return xyz, quaternion.tolist()


def _required_child(element: ET.Element, path: str, context: str) -> ET.Element:
    child = element.find(path)
    if child is None:
        raise ValueError(f"Missing {path!r} in {context}")
    return child


def _write(tree: ET.ElementTree, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)


def _mesh_min_z(model: mujoco.MjModel, data: mujoco.MjData, geom_name: str) -> float:
    """计算某个 mesh geom 在当前姿态下的最低世界 z 坐标。"""
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id < 0:
        raise ValueError(f"Generated model is missing geom {geom_name!r}")
    mesh_id = model.geom_dataid[geom_id]
    if mesh_id < 0:
        raise ValueError(f"Geom {geom_name!r} is not a mesh")
    start = model.mesh_vertadr[mesh_id]
    count = model.mesh_vertnum[mesh_id]
    vertices = model.mesh_vert[start:start + count]
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    return float((vertices @ rotation.T + data.geom_xpos[geom_id])[:, 2].min())


def convert_urdf(source: Path, output: Path) -> Path:
    """Validate source, generate a native MJCF model, and return output."""
    source = Path(source).resolve()
    output = Path(output).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"URDF source does not exist: {source}")

    # 第一阶段只做拓扑和资源校验：如果 URDF 结构不符合预期，直接失败，
    # 不生成一个“看起来能加载但语义错了”的模型。
    urdf_root = ET.parse(source).getroot()
    links = {element.get("name"): element for element in urdf_root.findall("link")}
    joints = {element.get("name"): element for element in urdf_root.findall("joint")}
    missing_links = EXPECTED_LINKS - links.keys()
    missing_joints = EXPECTED_JOINTS - joints.keys()
    if missing_links or missing_joints:
        raise ValueError(
            f"Missing required links {sorted(missing_links)} or joints {sorted(missing_joints)}"
        )
    if set(links) != EXPECTED_LINKS or set(joints) != EXPECTED_JOINTS:
        raise ValueError("URDF must contain exactly the expected nine links and eight joints")

    children: dict[str, list[ET.Element]] = defaultdict(list)
    child_to_parent: dict[str, str] = {}
    for name, joint in joints.items():
        parent = _required_child(joint, "parent", f"joint {name}").get("link")
        child = _required_child(joint, "child", f"joint {name}").get("link")
        if parent not in links or child not in links:
            raise ValueError(f"Joint {name} references a missing parent or child link")
        if child in child_to_parent:
            raise ValueError(f"Link {child} has multiple parents")
        child_to_parent[child] = parent
        children[parent].append(joint)
    roots = set(links) - set(child_to_parent)
    if roots != {"base_link"}:
        raise ValueError(f"Expected only base_link as root, found {sorted(roots)}")
    visited: set[str] = set()

    def visit(link_name: str) -> None:
        if link_name in visited:
            raise ValueError(f"Cycle detected at link {link_name}")
        visited.add(link_name)
        for joint in children[link_name]:
            visit(_required_child(joint, "child", f"joint {joint.get('name')}").get("link"))

    visit("base_link")
    if visited != set(links):
        raise ValueError(f"Disconnected links: {sorted(set(links) - visited)}")

    # 本项目要求每个 link 使用同一个简化 STL 同时作为 visual/collision。
    mesh_files: dict[str, Path] = {}
    for link_name, link in links.items():
        inertial = link.find("inertial")
        if inertial is None or inertial.find("mass") is None or inertial.find("inertia") is None:
            raise ValueError(f"Link {link_name} is missing inertial data")
        visual_mesh = link.find("visual/geometry/mesh")
        collision_mesh = link.find("collision/geometry/mesh")
        if visual_mesh is None or collision_mesh is None:
            raise ValueError(f"Link {link_name} is missing visual or collision mesh data")
        visual_file = (source.parent / visual_mesh.get("filename")).resolve()
        collision_file = (source.parent / collision_mesh.get("filename")).resolve()
        if visual_file != collision_file:
            raise ValueError(f"Link {link_name} must use the same simplified visual/collision mesh")
        if not visual_file.is_file():
            raise FileNotFoundError(f"Mesh for {link_name} does not exist: {visual_file}")
        mesh_files[link_name] = visual_file

    # 第二阶段开始组装 MJCF。compiler.meshdir 相对于输出 XML 所在目录。
    model_root = ET.Element("mujoco", {"model": "passive_wheeled_biped"})
    ET.SubElement(
        model_root,
        "compiler",
        {"angle": "radian", "meshdir": "../meshes", "autolimits": "true", "balanceinertia": "true"},
    )
    ET.SubElement(
        model_root,
        "option",
        {"timestep": "0.001", "integrator": "implicitfast", "gravity": "0 0 -9.81"},
    )
    asset = ET.SubElement(model_root, "asset")
    for link_name in sorted(links):
        ET.SubElement(
            asset,
            "mesh",
            {"name": corrected_name(link_name), "file": mesh_files[link_name].name},
        )
    contact = ET.SubElement(model_root, "contact")
    exclusions = {
        tuple(sorted((corrected_name(parent), corrected_name(child))))
        for child, parent in child_to_parent.items()
    }
    exclusions.update(
        {
            tuple(sorted(("base_link", "left_hip_pitch_link"))),
            tuple(sorted(("base_link", "right_hip_pitch_link"))),
        }
    )
    for body1, body2 in sorted(exclusions):
        ET.SubElement(contact, "exclude", {"body1": body1, "body2": body2})

    worldbody = ET.SubElement(model_root, "worldbody")
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": "0 0 0.1",
            "friction": "1.0 0.005 0.0001",
        },
    )

    def add_link(link_name: str, parent_body: ET.Element, joint: ET.Element | None = None) -> ET.Element:
        """递归把 URDF link/joint 树翻译为 MuJoCo body/joint 树。"""
        attributes = {"name": corrected_name(link_name)}
        if joint is not None:
            xyz, quat = _origin(joint.find("origin"))
            attributes.update({"pos": _format(xyz), "quat": _format(quat)})
        body = ET.SubElement(parent_body, "body", attributes)
        if joint is None:
            # 根 body 是真实 free-base，不在默认模型里固定住。
            ET.SubElement(body, "freejoint", {"name": "base_freejoint"})
        else:
            joint_name = joint.get("name")
            joint_type = joint.get("type")
            if joint_type not in {"revolute", "continuous"}:
                raise ValueError(f"Unsupported joint type {joint_type!r} for {joint_name}")
            axis = _numbers(_required_child(joint, "axis", f"joint {joint_name}").get("xyz"), "1 0 0")
            joint_attributes = {
                "name": corrected_name(joint_name),
                "type": "hinge",
                "axis": _format(axis),
                "damping": "0.01" if joint_type == "continuous" else "0.1",
            }
            if joint_type == "revolute":
                limit = _required_child(joint, "limit", f"joint {joint_name}")
                if limit.get("lower") is None or limit.get("upper") is None:
                    raise ValueError(f"Revolute joint {joint_name} is missing its range")
                corrected_joint_name = corrected_name(joint_name)
                if corrected_joint_name in {"left_hip_pitch_joint", "right_hip_pitch_joint"}:
                    # hip pitch 的限位来自当前控制/机械约定，不沿用 URDF 旧值。
                    joint_attributes["range"] = _format(HIP_PITCH_RANGE)
                else:
                    joint_attributes["range"] = f"{limit.get('lower')} {limit.get('upper')}"
            ET.SubElement(body, "joint", joint_attributes)
        if link_name == "base_link":
            # 理想 IMU 安装点：site 本身不参与碰撞，只给 MuJoCo sensor 绑定。
            ET.SubElement(
                body,
                "site",
                {
                    "name": "base_imu_site",
                    "pos": "0 0 0.08",
                    "size": "0.015",
                    "rgba": "0 0.7 1 1",
                },
            )

        link = links[link_name]
        # MuJoCo 这里要求 inertia frame 不旋转；如果未来 URDF 导出旋转惯量，
        # 需要显式处理，而不是悄悄丢掉姿态信息。
        inertial = _required_child(link, "inertial", f"link {link_name}")
        inertial_xyz, inertial_quat = _origin(inertial.find("origin"))
        if not np.allclose(inertial_quat, [1.0, 0.0, 0.0, 0.0], atol=1e-10):
            raise ValueError(f"Link {link_name} has an unsupported rotated inertia frame")
        mass = _required_child(inertial, "mass", f"link {link_name}").get("value")
        tensor = _required_child(inertial, "inertia", f"link {link_name}")
        full_inertia = [
            tensor.get("ixx"),
            tensor.get("iyy"),
            tensor.get("izz"),
            tensor.get("ixy"),
            tensor.get("ixz"),
            tensor.get("iyz"),
        ]
        if mass is None or any(value is None for value in full_inertia):
            raise ValueError(f"Link {link_name} has incomplete inertial values")
        ET.SubElement(
            body,
            "inertial",
            {
                "pos": _format(inertial_xyz),
                "mass": mass,
                "fullinertia": " ".join(full_inertia),
            },
        )
        visual = _required_child(link, "visual", f"link {link_name}")
        visual_xyz, visual_quat = _origin(visual.find("origin"))
        color = visual.find("material/color")
        rgba = color.get("rgba") if color is not None and color.get("rgba") else "1 1 1 1"
        semantic_name = corrected_name(link_name)
        geom_stem = semantic_name.removesuffix("_link")
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{geom_stem}_visual",
                "type": "mesh",
                "mesh": semantic_name,
                "pos": _format(visual_xyz),
                "quat": _format(visual_quat),
                "rgba": rgba,
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
            },
        )
        collision = _required_child(link, "collision", f"link {link_name}")
        collision_xyz, collision_quat = _origin(collision.find("origin"))
        collision_attributes = {
                "name": f"{geom_stem}_collision",
                "type": "mesh",
                "mesh": semantic_name,
                "pos": _format(collision_xyz),
                "quat": _format(collision_quat),
                "rgba": "0 0 0 0",
                "contype": "1",
                "conaffinity": "1",
                "group": "3",
        }
        if link_name in {"left_wheel_link", "right_wheel_link"}:
            # 给轮子一点接触 margin，帮助初始地面接触更稳定。
            collision_attributes["margin"] = "0.001"
        ET.SubElement(body, "geom", collision_attributes)
        for child_joint in sorted(children[link_name], key=lambda element: element.get("name")):
            child_name = _required_child(
                child_joint, "child", f"joint {child_joint.get('name')}"
            ).get("link")
            add_link(child_name, body, child_joint)
        return body

    base_body = add_link("base_link", worldbody)
    actuator = ET.SubElement(model_root, "actuator")
    for joint_name in CONTROLLED_JOINTS:
        # 这里使用 torque motor；实际 PD/平衡控制在 Python 侧写 data.ctrl。
        lower, upper = TORQUE_LIMITS[joint_name]
        ET.SubElement(
            actuator,
            "motor",
            {
                "name": f"{joint_name}_motor",
                "joint": joint_name,
                "gear": "1",
                "ctrllimited": "true",
                "ctrlrange": _format([lower, upper]),
            },
        )
    equality = ET.SubElement(model_root, "equality")
    # 默认 inactive。分析脚本会临时打开它做固定基座阶跃响应。
    ET.SubElement(
        equality,
        "weld",
        {
            "name": "fixed_base_weld",
            "body1": "world",
            "body2": "base_link",
            "active": "false",
        },
    )
    sensor = ET.SubElement(model_root, "sensor")
    # IMU 输出顺序：gyro(3) + accel(3) + framequat(4)，总 sensordata 维度 10。
    ET.SubElement(sensor, "gyro", {"name": "base_imu_gyro", "site": "base_imu_site"})
    ET.SubElement(sensor, "accelerometer", {"name": "base_imu_accel", "site": "base_imu_site"})
    ET.SubElement(
        sensor,
        "framequat",
        {
            "name": "base_imu_quat",
            "objtype": "site",
            "objname": "base_imu_site",
        },
    )
    tree = ET.ElementTree(model_root)
    _write(tree, output)

    # 第三阶段：先加载一次临时模型，测轮子最低点，反推出 base_link 高度。
    provisional = mujoco.MjModel.from_xml_path(str(output))
    provisional_data = mujoco.MjData(provisional)
    mujoco.mj_forward(provisional, provisional_data)
    wheel_minima = [
        _mesh_min_z(provisional, provisional_data, "left_wheel_collision"),
        _mesh_min_z(provisional, provisional_data, "right_wheel_collision"),
    ]
    if abs(wheel_minima[0] - wheel_minima[1]) > 1e-3:
        raise ValueError(f"Wheel minima differ by more than 1 mm: {wheel_minima}")
    base_height = -float(min(wheel_minima))
    base_body.set("pos", _format([0.0, 0.0, base_height]))
    keyframe = ET.SubElement(model_root, "keyframe")
    ET.SubElement(
        keyframe,
        "key",
        {
            "name": "home",
            "qpos": _format([0.0, 0.0, base_height, 1.0, 0.0, 0.0, 0.0] + [0.0] * 8),
        },
    )
    _write(tree, output)

    # 最终自检：除轮子之外的碰撞体不能在初始姿态明显穿地。
    final_model = mujoco.MjModel.from_xml_path(str(output))
    final_data = mujoco.MjData(final_model)
    mujoco.mj_forward(final_model, final_data)
    for geom_id in range(final_model.ngeom):
        geom_name = mujoco.mj_id2name(final_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if geom_name == "floor" or geom_name.endswith("_visual") or "wheel_collision" in geom_name:
            continue
        if _mesh_min_z(final_model, final_data, geom_name) < -1e-3:
            raise ValueError(f"Collision geom {geom_name} starts more than 1 mm below ground")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "8dof_URDF" / "urdf" / "robot.urdf",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "8dof_URDF" / "mjcf" / "robot.xml",
    )
    args = parser.parse_args()
    print(convert_urdf(args.source, args.output))


if __name__ == "__main__":
    main()
