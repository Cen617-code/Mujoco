import csv
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

import scripts.analyze_balance as analyze_balance
from scripts.convert_urdf_to_mjcf import convert_urdf
from scripts.pd_control import build_joint_map
from scripts.balance_control import (
    BalanceConfig,
    apply_balance_control,
    base_pitch,
    base_pitch_rate,
    compute_balance_control,
    quat_to_pitch,
)
from scripts.analyze_balance import (
    BalanceSimulationResult,
    run_balance_simulation,
    write_balance_results,
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


def test_base_imu_pitch_matches_freejoint_pitch(model):
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.qpos[3:7] = quat_y_rotation(0.12)
    mujoco.mj_forward(model, data)
    from scripts.balance_control import base_pitch_from_imu, has_base_imu

    assert has_base_imu(model)
    assert base_pitch_from_imu(model, data) == pytest.approx(base_pitch(data), abs=1e-8)


def test_base_pitch_rate_uses_near_upright_free_joint_angular_y(model):
    data = mujoco.MjData(model)
    data.qvel[:] = 0.0
    data.qvel[4] = 0.3
    assert base_pitch_rate(data) == pytest.approx(0.3)


def test_base_imu_pitch_rate_matches_freejoint_pitch_rate(model):
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.qvel[4] = 0.23
    mujoco.mj_forward(model, data)
    from scripts.balance_control import base_pitch_rate_from_imu, has_base_imu

    assert has_base_imu(model)
    assert base_pitch_rate_from_imu(model, data) == pytest.approx(
        base_pitch_rate(data), abs=1e-8
    )


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
    assert ctrl[left_wheel.actuator_id] == pytest.approx(
        model.actuator_ctrlrange[left_wheel.actuator_id, 0]
    )
    assert ctrl[right_wheel.actuator_id] == pytest.approx(
        model.actuator_ctrlrange[right_wheel.actuator_id, 1]
    )


def test_balance_controller_prefers_imu_pitch(model, monkeypatch):
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.qpos[3:7] = quat_y_rotation(0.1)
    mujoco.mj_forward(model, data)
    joint_map = build_joint_map(model)

    import scripts.balance_control as balance_control

    calls = {"imu": 0}
    original = balance_control.base_pitch_from_imu

    def recording_pitch_from_imu(model_arg, data_arg):
        calls["imu"] += 1
        return original(model_arg, data_arg)

    monkeypatch.setattr(balance_control, "base_pitch_from_imu", recording_pitch_from_imu)
    compute_balance_control(model, data, joint_map, BalanceConfig())
    assert calls["imu"] == 1


def test_negative_pitch_saturates_wheel_torque_to_upper_limit(model):
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.qpos[3:7] = quat_y_rotation(-0.2)
    mujoco.mj_forward(model, data)
    joint_map = build_joint_map(model)
    config = BalanceConfig(kp_pitch=100.0, kd_pitch=0.0, kx=0.0, kv=0.0)
    ctrl, state = compute_balance_control(model, data, joint_map, config)
    left_wheel = next(entry for entry in joint_map if entry.joint_name == "left_wheel_joint")
    right_wheel = next(entry for entry in joint_map if entry.joint_name == "right_wheel_joint")
    assert state.pitch == pytest.approx(-0.2, abs=1e-9)
    assert ctrl[left_wheel.actuator_id] == pytest.approx(
        model.actuator_ctrlrange[left_wheel.actuator_id, 1]
    )
    assert ctrl[right_wheel.actuator_id] == pytest.approx(
        model.actuator_ctrlrange[right_wheel.actuator_id, 0]
    )


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


def test_balance_simulation_runs_finite(model):
    result = run_balance_simulation(model, duration=0.25)
    assert result.warning_count == 0
    assert result.finite
    assert result.steps > 0
    assert len(result.timeseries) == result.steps
    assert result.peak_abs_wheel_torque <= 10.0 + 1e-9
    assert np.isfinite(result.final_pitch)


def test_balance_simulation_defaults_none_x_target_to_initial_base_x(model, monkeypatch):
    captured_x_targets: list[float | None] = []
    original_apply_balance_control = analyze_balance.apply_balance_control

    def recording_apply_balance_control(
        model_arg,
        data,
        joint_map,
        config=None,
        leg_targets=None,
    ):
        captured_x_targets.append(None if config is None else config.x_target)
        return original_apply_balance_control(model_arg, data, joint_map, config, leg_targets)

    monkeypatch.setattr(
        analyze_balance,
        "apply_balance_control",
        recording_apply_balance_control,
    )
    result = run_balance_simulation(model, duration=0.01, config=BalanceConfig(kx=1.0))

    assert result.finite
    assert captured_x_targets
    assert all(
        x_target == pytest.approx(float(model.qpos0[0]))
        for x_target in captured_x_targets
    )


def test_balance_simulation_summary_matches_final_sample(model):
    result = run_balance_simulation(model, duration=0.25)
    final_sample = result.timeseries[-1]
    assert result.final_pitch == pytest.approx(final_sample["pitch"])
    assert result.final_base_height == pytest.approx(final_sample["base_height"])
    assert result.peak_abs_pitch == pytest.approx(
        max(abs(sample["pitch"]) for sample in result.timeseries)
    )
    assert result.peak_abs_pitch_rate == pytest.approx(
        max(abs(sample["pitch_rate"]) for sample in result.timeseries)
    )
    assert result.peak_abs_wheel_torque == pytest.approx(
        max(abs(sample["wheel_torque"]) for sample in result.timeseries)
    )


def test_write_balance_results_outputs_planned_files(model, tmp_path):
    result = run_balance_simulation(model, duration=0.05)
    write_balance_results(result, tmp_path)
    assert (tmp_path / "balance_summary.csv").is_file()
    assert (tmp_path / "balance_timeseries.csv").is_file()
    assert (tmp_path / "balance_report.md").is_file()
    report = (tmp_path / "balance_report.md").read_text(encoding="utf-8")
    assert "Balance Control Analysis" in report
    assert "not walking" in report
    with (tmp_path / "balance_timeseries.csv").open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == [
            "time",
            "pitch",
            "pitch_rate",
            "x",
            "x_velocity",
            "wheel_torque",
            "base_height",
        ]
        assert len(list(reader)) == result.steps


def test_write_balance_results_notes_wheel_torque_saturation(tmp_path):
    result = BalanceSimulationResult(
        duration=2.0,
        timestep=0.001,
        steps=2000,
        warning_count=0,
        finite=True,
        peak_abs_pitch=1.0,
        final_pitch=0.5,
        peak_abs_pitch_rate=5.0,
        peak_abs_wheel_torque=10.0,
        final_base_height=0.12,
        timeseries=[
            {
                "time": 0.001,
                "pitch": 0.1,
                "pitch_rate": 0.2,
                "x": 0.0,
                "x_velocity": 0.0,
                "wheel_torque": 10.0,
                "base_height": 0.12,
            }
        ],
    )
    write_balance_results(result, tmp_path)
    report = (tmp_path / "balance_report.md").read_text(encoding="utf-8")
    assert "reached the ±10 N·m wheel torque limit" in report
    assert "do not demonstrate robust standing or walking" in report


def test_run_balance_viewer_help_succeeds():
    script = ROOT / "scripts" / "run_balance_viewer.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0
    assert "--duration" in completed.stdout
