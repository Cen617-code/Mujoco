"""Analyze first-pass free-base body balance control.

运行当前 pitch 平衡控制器，并把机身角度、轮子力矩、base 高度等写成 CSV/报告。
报告里的大 pitch 或力矩饱和并不算失败，而是告诉我们下一轮需要继续调控制器。
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import mujoco
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.balance_control import (
    BalanceConfig,
    BalanceState,
    apply_balance_control,
    base_pitch,
    base_pitch_rate,
)
from scripts.convert_urdf_to_mjcf import ROOT, convert_urdf
from scripts.pd_control import build_joint_map, set_base_weld_active


DEFAULT_SOURCE = ROOT / "8dof_URDF" / "urdf" / "robot.urdf"
DEFAULT_MODEL = ROOT / "8dof_URDF" / "mjcf" / "robot.xml"
DEFAULT_RESULTS = ROOT / "analysis" / "balance_results"
WHEEL_TORQUE_LIMIT_NM = 10.0
TORQUE_LIMIT_EPSILON = 1e-9
STANDING_FINAL_ABS_PITCH_LIMIT = 0.25
STANDING_PEAK_ABS_PITCH_LIMIT = 0.5
STANDING_PEAK_ABS_X_DRIFT_LIMIT = 0.3
STANDING_FAILURE_SCORE = 1_000_000.0


@dataclass(frozen=True)
class BalanceSimulationResult:
    """一次 free-base 平衡仿真的汇总指标和逐步时间序列。"""

    duration: float
    timestep: float
    steps: int
    warning_count: int
    finite: bool
    initial_x: float
    final_x: float
    x_drift: float
    peak_abs_x_drift: float
    peak_abs_pitch: float
    final_pitch: float
    final_abs_pitch: float
    peak_abs_pitch_rate: float
    peak_abs_wheel_torque: float
    wheel_torque_saturation_fraction: float
    final_base_height: float
    meets_standing_objective: bool
    standing_score: float
    timeseries: list[dict[str, float]]


def _warning_count(data: mujoco.MjData) -> int:
    return int(sum(int(warning.number) for warning in data.warning))


def _finite(data: mujoco.MjData, ctrl: np.ndarray) -> bool:
    return bool(
        np.isfinite(data.qpos).all()
        and np.isfinite(data.qvel).all()
        and np.isfinite(ctrl).all()
    )


def meets_standing_objective_values(
    *,
    warning_count: int,
    finite: bool,
    final_abs_pitch: float,
    peak_abs_pitch: float,
    peak_abs_x_drift: float,
) -> bool:
    return (
        int(warning_count) == 0
        and bool(finite)
        and float(final_abs_pitch) < STANDING_FINAL_ABS_PITCH_LIMIT
        and float(peak_abs_pitch) < STANDING_PEAK_ABS_PITCH_LIMIT
        and float(peak_abs_x_drift) < STANDING_PEAK_ABS_X_DRIFT_LIMIT
    )


def standing_score_values(
    *,
    warning_count: int,
    finite: bool,
    final_abs_pitch: float,
    peak_abs_pitch: float,
    peak_abs_x_drift: float,
    wheel_torque_saturation_fraction: float,
) -> float:
    values = [
        float(final_abs_pitch),
        float(peak_abs_pitch),
        float(peak_abs_x_drift),
        float(wheel_torque_saturation_fraction),
    ]
    try:
        warning_count_value = int(warning_count)
    except (TypeError, ValueError, OverflowError):
        return STANDING_FAILURE_SCORE
    if warning_count_value != 0 or not bool(finite) or not np.isfinite(values).all():
        return STANDING_FAILURE_SCORE
    score = (
        4.0 * float(final_abs_pitch)
        + 2.0 * float(peak_abs_pitch)
        + float(peak_abs_x_drift)
        + 0.5 * float(wheel_torque_saturation_fraction)
    )
    if float(peak_abs_pitch) >= 1.6 or float(final_abs_pitch) >= 1.2:
        score += 1_000.0
    return float(score)


def _sample_balance_state(data: mujoco.MjData, wheel_torque: float) -> BalanceState:
    """从 MuJoCo data 中抽取报告需要的机身状态。"""
    return BalanceState(
        pitch=base_pitch(data),
        pitch_rate=base_pitch_rate(data),
        x=float(data.qpos[0]),
        x_velocity=float(data.qvel[0]),
        wheel_torque=float(wheel_torque),
    )


def run_balance_simulation(
    model: mujoco.MjModel,
    duration: float = 2.0,
    config: BalanceConfig | None = None,
    leg_targets: Mapping[str, float] | None = None,
) -> BalanceSimulationResult:
    """以 free-base 模式运行平衡控制器并收集诊断数据。"""
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    set_base_weld_active(model, data, False)
    if config is None:
        # 默认把当前 x 位置作为目标，避免控制器一启动就试图回到全局 0。
        config = BalanceConfig(x_target=float(data.qpos[0]))
    elif config.x_target is None:
        config = replace(config, x_target=float(data.qpos[0]))
    initial_x = float(data.qpos[0])
    joint_map = build_joint_map(model)
    timestep = float(model.opt.timestep)
    steps = max(1, int(np.ceil(float(duration) / timestep)))
    finite = True
    timeseries: list[dict[str, float]] = []

    for _ in range(steps):
        # 控制在 step 前写入 data.ctrl；step 后采样新的状态作为 timeseries。
        ctrl, applied_state = apply_balance_control(model, data, joint_map, config, leg_targets)
        finite = finite and _finite(data, ctrl)
        mujoco.mj_step(model, data)
        finite = finite and _finite(data, ctrl)
        state = _sample_balance_state(data, applied_state.wheel_torque)
        x_drift = state.x - initial_x
        timeseries.append(
            {
                "time": float(data.time),
                "pitch": state.pitch,
                "pitch_rate": state.pitch_rate,
                "x": state.x,
                "x_velocity": state.x_velocity,
                "x_drift": x_drift,
                "wheel_torque": state.wheel_torque,
                "base_height": float(data.qpos[2]),
            }
        )

    final_sample = timeseries[-1]
    warning_count = _warning_count(data)
    final_pitch = float(final_sample["pitch"])
    final_abs_pitch = abs(final_pitch)
    peak_abs_pitch = max(abs(sample["pitch"]) for sample in timeseries)
    peak_abs_pitch_rate = max(abs(sample["pitch_rate"]) for sample in timeseries)
    peak_abs_wheel_torque = max(abs(sample["wheel_torque"]) for sample in timeseries)
    peak_abs_x_drift = max(abs(sample["x_drift"]) for sample in timeseries)
    saturated_samples = sum(
        sample["wheel_torque"] >= WHEEL_TORQUE_LIMIT_NM - TORQUE_LIMIT_EPSILON
        for sample in timeseries
    )
    wheel_torque_saturation_fraction = float(saturated_samples / len(timeseries))
    meets_objective = meets_standing_objective_values(
        warning_count=warning_count,
        finite=finite,
        final_abs_pitch=final_abs_pitch,
        peak_abs_pitch=peak_abs_pitch,
        peak_abs_x_drift=peak_abs_x_drift,
    )
    score = standing_score_values(
        warning_count=warning_count,
        finite=finite,
        final_abs_pitch=final_abs_pitch,
        peak_abs_pitch=peak_abs_pitch,
        peak_abs_x_drift=peak_abs_x_drift,
        wheel_torque_saturation_fraction=wheel_torque_saturation_fraction,
    )

    return BalanceSimulationResult(
        duration=float(duration),
        timestep=timestep,
        steps=steps,
        warning_count=warning_count,
        finite=finite,
        initial_x=initial_x,
        final_x=float(final_sample["x"]),
        x_drift=float(final_sample["x"] - initial_x),
        peak_abs_x_drift=peak_abs_x_drift,
        peak_abs_pitch=peak_abs_pitch,
        final_pitch=final_pitch,
        final_abs_pitch=final_abs_pitch,
        peak_abs_pitch_rate=peak_abs_pitch_rate,
        peak_abs_wheel_torque=peak_abs_wheel_torque,
        wheel_torque_saturation_fraction=wheel_torque_saturation_fraction,
        final_base_height=final_sample["base_height"],
        meets_standing_objective=meets_objective,
        standing_score=score,
        timeseries=timeseries,
    )


def write_balance_results(
    result: BalanceSimulationResult,
    output_dir: Path = DEFAULT_RESULTS,
) -> Path:
    """输出 summary CSV、timeseries CSV 和人可读 Markdown 报告。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_fields = [
        "duration",
        "timestep",
        "steps",
        "warning_count",
        "finite",
        "initial_x",
        "final_x",
        "x_drift",
        "peak_abs_x_drift",
        "peak_abs_pitch",
        "final_pitch",
        "final_abs_pitch",
        "peak_abs_pitch_rate",
        "peak_abs_wheel_torque",
        "wheel_torque_saturation_fraction",
        "final_base_height",
        "meets_standing_objective",
        "standing_score",
    ]
    with (output_dir / "balance_summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        summary = asdict(result)
        summary.pop("timeseries")
        writer.writerow(summary)

    timeseries_fields = [
        "time",
        "pitch",
        "pitch_rate",
        "x",
        "x_velocity",
        "x_drift",
        "wheel_torque",
        "base_height",
    ]
    with (output_dir / "balance_timeseries.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=timeseries_fields)
        writer.writeheader()
        writer.writerows(result.timeseries)

    lines = [
        "# Balance Control Analysis",
        "",
        f"- Duration: {result.duration:.6g} s",
        f"- Steps: {result.steps}",
        f"- MuJoCo warnings: {result.warning_count}",
        f"- Finite state: {result.finite}",
        f"- Standing objective met: {result.meets_standing_objective}",
        f"- Standing score: {result.standing_score:.6g}",
        f"- Peak |pitch|: {result.peak_abs_pitch:.6g} rad",
        f"- Final pitch: {result.final_pitch:.6g} rad",
        f"- Final |pitch|: {result.final_abs_pitch:.6g} rad",
        f"- Peak |pitch rate|: {result.peak_abs_pitch_rate:.6g} rad/s",
        f"- Peak |x drift|: {result.peak_abs_x_drift:.6g} m",
        f"- Final x drift: {result.x_drift:.6g} m",
        f"- Peak |wheel torque|: {result.peak_abs_wheel_torque:.6g} N·m",
        f"- Wheel torque saturation fraction: {result.wheel_torque_saturation_fraction:.3f}",
        f"- Final base height: {result.final_base_height:.6g} m",
        "",
        "This is a first-pass in-place balance prototype, not walking or trajectory tracking.",
    ]
    if not result.meets_standing_objective:
        lines.extend(
            [
                "",
                "This run does not meet the robust-standing v1 objective.",
            ]
        )
    if result.peak_abs_wheel_torque >= WHEEL_TORQUE_LIMIT_NM - TORQUE_LIMIT_EPSILON:
        # 轮子力矩打满时，明确提醒这不是“已经稳住”的证据。
        lines.extend(
            [
                "",
                "This run reached the ±10 N·m wheel torque limit; results are first-pass balance diagnostics "
                "and do not demonstrate robust standing or walking.",
            ]
        )
    lines.append("")
    (output_dir / "balance_report.md").write_text("\n".join(lines), encoding="utf-8")
    return output_dir


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--duration", type=float, default=2.0)
    args = parser.parse_args(argv)
    model_path = convert_urdf(args.source, args.model)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    result = run_balance_simulation(model, duration=args.duration)
    output_dir = write_balance_results(result, args.output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
