from pathlib import Path
import subprocess
import sys
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


from scripts.pd_control import (
    CONTROLLED_JOINTS as PD_CONTROLLED_JOINTS,
    build_joint_map,
    clip_targets_to_joint_limits,
    compute_pd_control,
    default_pd_gains,
    set_base_weld_active,
)


def test_pd_joint_order_matches_actuator_order(model):
    assert PD_CONTROLLED_JOINTS == CONTROLLED_JOINTS
    joint_map = build_joint_map(model)
    assert [entry.joint_name for entry in joint_map] == CONTROLLED_JOINTS
    assert [entry.actuator_id for entry in joint_map] == list(range(8))


def test_pd_target_clipping_and_torque_saturation(model):
    data = mujoco.MjData(model)
    joint_map = build_joint_map(model)
    raw_targets = {entry.joint_name: 100.0 for entry in joint_map}
    clipped = clip_targets_to_joint_limits(model, joint_map, raw_targets)
    assert clipped["left_hip_pitch_joint"] == pytest.approx(0.87)
    assert clipped["right_hip_pitch_joint"] == pytest.approx(0.87)
    assert clipped["left_wheel_joint"] == pytest.approx(100.0)
    gains = {entry.joint_name: (1_000.0, 0.0) for entry in joint_map}
    ctrl = compute_pd_control(model, data, joint_map, clipped, gains)
    assert ctrl.shape == (8,)
    assert np.all(ctrl <= model.actuator_ctrlrange[:, 1] + 1e-12)
    assert np.all(ctrl >= model.actuator_ctrlrange[:, 0] - 1e-12)
    assert ctrl[1] == pytest.approx(30.0)
    assert ctrl[5] == pytest.approx(30.0)
    assert ctrl[3] == pytest.approx(10.0)
    assert ctrl[7] == pytest.approx(10.0)


def test_default_pd_gains_are_positive_and_finite(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    gains = default_pd_gains(model, data)
    assert set(gains) == set(CONTROLLED_JOINTS)
    for kp, kd in gains.values():
        assert np.isfinite(kp)
        assert np.isfinite(kd)
        assert kp > 0
        assert kd > 0


def test_base_weld_can_be_toggled(model):
    data = mujoco.MjData(model)
    set_base_weld_active(model, data, True)
    equality_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "fixed_base_weld")
    assert data.eq_active[equality_id] == 1
    set_base_weld_active(model, data, False)
    assert data.eq_active[equality_id] == 0


from scripts.analyze_dynamics import (
    FreeBaseResult,
    StepMetric,
    StepResponseResult,
    run_fixed_base_step_response,
    run_free_base_posture_check,
    write_results,
)


def test_fixed_base_step_response_runs_finite(model):
    result = run_fixed_base_step_response(model, duration=0.25, step_size=0.05, wheel_step_size=0.1)
    assert len(result.metrics) == 8
    for metric in result.metrics:
        assert metric.joint_name in CONTROLLED_JOINTS
        assert np.isfinite(metric.final_position)
        assert np.isfinite(metric.steady_state_error)
        assert np.isfinite(metric.peak_torque)
        assert 0.0 <= metric.saturation_fraction <= 1.0
    assert result.warning_count == 0


def test_free_base_posture_check_runs_finite(model):
    result = run_free_base_posture_check(model, duration=0.25)
    assert result.steps > 0
    assert result.warning_count == 0
    assert np.isfinite(result.peak_abs_qvel)
    assert np.isfinite(result.peak_abs_ctrl)
    assert result.peak_abs_ctrl <= 30.0 + 1e-9


def test_write_results_uses_planned_artifact_names(tmp_path):
    step_result = StepResponseResult(
        duration=0.25,
        timestep=0.001,
        warning_count=0,
        metrics=[
            StepMetric(
                joint_name="left_roll_joint",
                target_position=0.05,
                final_position=0.04,
                steady_state_error=0.01,
                peak_torque=1.5,
                saturation_fraction=0.0,
            )
        ],
        traces={
            "left_roll_joint": {
                "time": [0.0],
                "position": [0.04],
                "torque": [1.5],
                "target": [0.05],
            }
        },
    )
    free_result = FreeBaseResult(
        duration=0.25,
        timestep=0.001,
        steps=250,
        warning_count=0,
        peak_abs_qvel=0.2,
        peak_abs_ctrl=1.5,
        final_base_height=0.3,
    )

    output_dir = write_results(step_result, free_result, tmp_path)

    assert output_dir == tmp_path
    assert (tmp_path / "step_response_metrics.csv").is_file()
    assert (tmp_path / "free_base_summary.csv").is_file()
    assert (tmp_path / "dynamics_report.md").is_file()
    assert not (tmp_path / "summary.json").exists()
    assert not (tmp_path / "step_metrics.csv").exists()
    assert not (tmp_path / "report.md").exists()
    assert "left_roll_joint" in (tmp_path / "step_response_metrics.csv").read_text()
    assert "peak_abs_qvel" in (tmp_path / "free_base_summary.csv").read_text()
    assert "Free-base posture check" in (tmp_path / "dynamics_report.md").read_text()


def test_analyze_dynamics_direct_script_cli_writes_planned_artifacts(tmp_path):
    script = ROOT / "scripts" / "analyze_dynamics.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--duration",
            "0.05",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "step_response_metrics.csv").is_file()
    assert (tmp_path / "free_base_summary.csv").is_file()
    report = (tmp_path / "dynamics_report.md").read_text()
    assert "Balance control is not implemented" in report
    assert "free-base falling is allowed" in report
