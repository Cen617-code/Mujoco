import csv
import json
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
    default_standing_config,
    quat_to_pitch,
    standing_leg_targets,
)
from scripts.analyze_balance import (
    BalanceSimulationResult,
    STANDING_FAILURE_SCORE,
    meets_standing_objective_values,
    run_balance_simulation,
    standing_score_values,
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


def test_compute_balance_control_uses_symmetric_standing_leg_targets(model):
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    joint_map = build_joint_map(model)

    config = BalanceConfig(leg_kp=10.0, leg_kd=0.0, kp_pitch=0.0, kd_pitch=0.0, kv=0.0)
    ctrl, _ = compute_balance_control(
        model,
        data,
        joint_map,
        config,
        leg_targets=standing_leg_targets(hip_pitch=-0.1, knee=0.2),
    )

    by_name = {entry.joint_name: entry for entry in joint_map}
    assert ctrl[by_name["left_hip_pitch_joint"].actuator_id] < 0.0
    assert ctrl[by_name["right_hip_pitch_joint"].actuator_id] < 0.0
    assert ctrl[by_name["left_knee_joint"].actuator_id] > 0.0
    assert ctrl[by_name["right_knee_joint"].actuator_id] > 0.0
    assert ctrl[by_name["left_wheel_joint"].actuator_id] == pytest.approx(0.0)
    assert ctrl[by_name["right_wheel_joint"].actuator_id] == pytest.approx(0.0)


def test_standing_leg_targets_are_symmetric_and_leg_only():
    targets = standing_leg_targets(hip_pitch=-0.15, knee=0.35)

    assert targets == {
        "left_hip_pitch_joint": -0.15,
        "right_hip_pitch_joint": -0.15,
        "left_knee_joint": 0.35,
        "right_knee_joint": 0.35,
    }
    assert standing_leg_targets() == targets
    assert not any("wheel" in joint_name for joint_name in targets)
    assert not any("roll" in joint_name for joint_name in targets)


def test_default_standing_config_is_explicit_and_does_not_change_generic_config():
    standing = default_standing_config()

    assert isinstance(standing, BalanceConfig)
    # Task 1 intentionally starts with the current generic numeric defaults;
    # future tuning may update this named standing preset independently.
    assert standing.pitch_target == pytest.approx(0.0)
    assert standing.pitch_rate_target == pytest.approx(0.0)
    assert standing.x_target is None
    assert standing.x_velocity_target == pytest.approx(0.0)
    assert standing.kp_pitch == pytest.approx(35.0)
    assert standing.kd_pitch == pytest.approx(4.0)
    assert standing.kx == pytest.approx(0.0)
    assert standing.kv == pytest.approx(1.0)
    assert standing.leg_kp == pytest.approx(20.0)
    assert standing.leg_kd == pytest.approx(1.0)


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


def test_standing_objective_values_accept_good_result():
    assert meets_standing_objective_values(
        warning_count=0,
        finite=True,
        final_abs_pitch=0.1,
        peak_abs_pitch=0.2,
        peak_abs_x_drift=0.05,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"warning_count": 1, "finite": True, "final_abs_pitch": 0.1, "peak_abs_pitch": 0.2, "peak_abs_x_drift": 0.05},
        {"warning_count": 0, "finite": False, "final_abs_pitch": 0.1, "peak_abs_pitch": 0.2, "peak_abs_x_drift": 0.05},
        {"warning_count": 0, "finite": True, "final_abs_pitch": 0.25, "peak_abs_pitch": 0.2, "peak_abs_x_drift": 0.05},
        {"warning_count": 0, "finite": True, "final_abs_pitch": 0.1, "peak_abs_pitch": 0.5, "peak_abs_x_drift": 0.05},
        {"warning_count": 0, "finite": True, "final_abs_pitch": 0.1, "peak_abs_pitch": 0.2, "peak_abs_x_drift": 0.3},
    ],
)
def test_standing_objective_values_reject_failures(kwargs):
    assert not meets_standing_objective_values(**kwargs)


def test_standing_score_values_penalizes_warning_and_nonfinite_results():
    good_score = standing_score_values(
        warning_count=0,
        finite=True,
        final_abs_pitch=0.1,
        peak_abs_pitch=0.2,
        peak_abs_x_drift=0.05,
        wheel_torque_saturation_fraction=0.0,
    )
    warning_score = standing_score_values(
        warning_count=1,
        finite=True,
        final_abs_pitch=0.1,
        peak_abs_pitch=0.2,
        peak_abs_x_drift=0.05,
        wheel_torque_saturation_fraction=0.0,
    )
    nonfinite_score = standing_score_values(
        warning_count=0,
        finite=False,
        final_abs_pitch=0.1,
        peak_abs_pitch=0.2,
        peak_abs_x_drift=0.05,
        wheel_torque_saturation_fraction=0.0,
    )

    assert good_score == pytest.approx(4.0 * 0.1 + 2.0 * 0.2 + 0.05)
    assert warning_score > 1_000.0
    assert nonfinite_score > 1_000.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"final_abs_pitch": float("nan")},
        {"peak_abs_pitch": float("inf")},
        {"peak_abs_x_drift": float("nan")},
        {"wheel_torque_saturation_fraction": float("inf")},
        {"warning_count": float("nan")},
    ],
)
def test_standing_score_values_returns_failure_for_nonfinite_inputs(overrides):
    kwargs = {
        "warning_count": 0,
        "finite": True,
        "final_abs_pitch": 0.1,
        "peak_abs_pitch": 0.2,
        "peak_abs_x_drift": 0.05,
        "wheel_torque_saturation_fraction": 0.0,
    }
    kwargs.update(overrides)

    assert standing_score_values(**kwargs) == STANDING_FAILURE_SCORE


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


def test_balance_simulation_uses_default_standing_targets_when_not_provided(model, monkeypatch):
    captured_leg_targets: list[dict[str, float] | None] = []
    original_apply_balance_control = analyze_balance.apply_balance_control

    def recording_apply_balance_control(
        model_arg,
        data,
        joint_map,
        config=None,
        leg_targets=None,
    ):
        captured_leg_targets.append(None if leg_targets is None else dict(leg_targets))
        return original_apply_balance_control(model_arg, data, joint_map, config, leg_targets)

    monkeypatch.setattr(
        analyze_balance,
        "apply_balance_control",
        recording_apply_balance_control,
    )

    result = run_balance_simulation(model, duration=0.01)

    assert result.finite
    assert captured_leg_targets
    assert all(targets == standing_leg_targets() for targets in captured_leg_targets)


def test_balance_simulation_passes_leg_targets_to_balance_control(model, monkeypatch):
    sentinel_targets = {"left_knee_joint": 0.12}
    captured_leg_targets = []
    original_apply_balance_control = analyze_balance.apply_balance_control

    def recording_apply_balance_control(
        model_arg,
        data,
        joint_map,
        config=None,
        leg_targets=None,
    ):
        captured_leg_targets.append(leg_targets)
        return original_apply_balance_control(
            model_arg,
            data,
            joint_map,
            config,
            leg_targets,
        )

    monkeypatch.setattr(
        analyze_balance,
        "apply_balance_control",
        recording_apply_balance_control,
    )
    result = run_balance_simulation(
        model,
        duration=0.01,
        leg_targets=sentinel_targets,
    )

    assert result.finite
    assert captured_leg_targets
    assert all(leg_targets is sentinel_targets for leg_targets in captured_leg_targets)


def test_balance_simulation_summary_matches_final_sample(model):
    result = run_balance_simulation(model, duration=0.25)
    final_sample = result.timeseries[-1]
    assert result.final_pitch == pytest.approx(final_sample["pitch"])
    assert result.final_base_height == pytest.approx(final_sample["base_height"])
    assert result.final_x == pytest.approx(final_sample["x"])
    assert result.x_drift == pytest.approx(result.final_x - result.initial_x)
    assert result.final_abs_pitch == pytest.approx(abs(result.final_pitch))
    assert result.peak_abs_pitch == pytest.approx(
        max(abs(sample["pitch"]) for sample in result.timeseries)
    )
    assert result.peak_abs_pitch_rate == pytest.approx(
        max(abs(sample["pitch_rate"]) for sample in result.timeseries)
    )
    assert result.peak_abs_wheel_torque == pytest.approx(
        max(abs(sample["wheel_torque"]) for sample in result.timeseries)
    )
    assert result.peak_abs_x_drift == pytest.approx(
        max(abs(sample["x_drift"]) for sample in result.timeseries)
    )
    assert 0.0 <= result.wheel_torque_saturation_fraction <= 1.0
    assert result.meets_standing_objective == meets_standing_objective_values(
        warning_count=result.warning_count,
        finite=result.finite,
        final_abs_pitch=result.final_abs_pitch,
        peak_abs_pitch=result.peak_abs_pitch,
        peak_abs_x_drift=result.peak_abs_x_drift,
    )
    assert result.standing_score == pytest.approx(
        standing_score_values(
            warning_count=result.warning_count,
            finite=result.finite,
            final_abs_pitch=result.final_abs_pitch,
            peak_abs_pitch=result.peak_abs_pitch,
            peak_abs_x_drift=result.peak_abs_x_drift,
            wheel_torque_saturation_fraction=result.wheel_torque_saturation_fraction,
        )
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
            "x_drift",
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
        initial_x=0.0,
        final_x=0.1,
        x_drift=0.1,
        peak_abs_x_drift=0.1,
        peak_abs_pitch=1.0,
        final_pitch=0.5,
        final_abs_pitch=0.5,
        peak_abs_pitch_rate=5.0,
        peak_abs_wheel_torque=10.0,
        wheel_torque_saturation_fraction=1.0,
        final_base_height=0.12,
        meets_standing_objective=False,
        standing_score=100.0,
        timeseries=[
            {
                "time": 0.001,
                "pitch": 0.1,
                "pitch_rate": 0.2,
                "x": 0.1,
                "x_velocity": 0.0,
                "x_drift": 0.1,
                "wheel_torque": 10.0,
                "base_height": 0.12,
            }
        ],
    )
    write_balance_results(result, tmp_path)
    report = (tmp_path / "balance_report.md").read_text(encoding="utf-8")
    assert "reached the ±10 N·m wheel torque limit" in report
    assert "do not demonstrate robust standing or walking" in report


def test_write_standing_tuning_results_outputs_planned_files(tmp_path):
    from scripts.tune_standing_balance import StandingCandidate, write_tuning_results

    rows = [
        {
            "rank": 1,
            "kp_pitch": 30.0,
            "kd_pitch": 4.0,
            "kx": 0.5,
            "kv": 1.0,
            "hip_pitch": -0.15,
            "knee": 0.35,
            "warning_count": 0,
            "finite": True,
            "final_abs_pitch": 0.1,
            "peak_abs_pitch": 0.2,
            "peak_abs_x_drift": 0.05,
            "wheel_torque_saturation_fraction": 0.0,
            "standing_score": 0.85,
            "meets_standing_objective": True,
        }
    ]
    best = StandingCandidate(
        kp_pitch=30.0,
        kd_pitch=4.0,
        kx=0.5,
        kv=1.0,
        hip_pitch=-0.15,
        knee=0.35,
    )

    output_dir = write_tuning_results(rows, best, tmp_path)

    assert output_dir == tmp_path
    assert (tmp_path / "standing_tuning_results.csv").is_file()
    assert (tmp_path / "standing_best_config.json").is_file()
    assert (tmp_path / "standing_tuning_report.md").is_file()
    best_config = json.loads((tmp_path / "standing_best_config.json").read_text(encoding="utf-8"))
    assert best_config["candidate"]["kp_pitch"] == pytest.approx(30.0)
    report = (tmp_path / "standing_tuning_report.md").read_text(encoding="utf-8")
    assert "Robust Standing Tuning" in report
    assert "Objective met: True" in report


def test_run_standing_tuning_smoke_returns_finite_candidate(model):
    from scripts.tune_standing_balance import StandingCandidate, run_tuning

    rows, best = run_tuning(
        model,
        duration=0.02,
        candidates=[
            StandingCandidate(
                kp_pitch=10.0,
                kd_pitch=1.0,
                kx=0.0,
                kv=0.5,
                hip_pitch=-0.15,
                knee=0.35,
            )
        ],
    )

    assert len(rows) == 1
    assert best is not None
    assert rows[0]["finite"]
    assert np.isfinite(rows[0]["standing_score"])


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
