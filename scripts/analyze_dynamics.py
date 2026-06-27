"""Run basic finite-dynamics checks for the motorized MuJoCo model."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.convert_urdf_to_mjcf import ROOT, convert_urdf
from scripts.pd_control import (
    apply_pd_control,
    build_joint_map,
    default_pd_gains,
    home_targets,
    set_base_weld_active,
)


DEFAULT_SOURCE = ROOT / "8dof_URDF" / "urdf" / "robot.urdf"
DEFAULT_MODEL = ROOT / "8dof_URDF" / "mjcf" / "robot.xml"
DEFAULT_RESULTS = ROOT / "analysis" / "results"


@dataclass(frozen=True)
class StepMetric:
    joint_name: str
    target_position: float
    final_position: float
    steady_state_error: float
    peak_torque: float
    saturation_fraction: float


@dataclass(frozen=True)
class StepResponseResult:
    duration: float
    timestep: float
    warning_count: int
    metrics: list[StepMetric]
    traces: dict[str, dict[str, list[float]]]


@dataclass(frozen=True)
class FreeBaseResult:
    duration: float
    timestep: float
    steps: int
    warning_count: int
    peak_abs_qvel: float
    peak_abs_ctrl: float
    final_base_height: float


def _warning_count(data: mujoco.MjData) -> int:
    """Return the total number of MuJoCo warnings recorded on ``data``."""
    return int(sum(int(warning.number) for warning in data.warning))


def _require_finite(data: mujoco.MjData, context: str) -> None:
    if not np.all(np.isfinite(data.qpos)):
        raise FloatingPointError(f"Non-finite qpos during {context}")
    if not np.all(np.isfinite(data.qvel)):
        raise FloatingPointError(f"Non-finite qvel during {context}")
    if not np.all(np.isfinite(data.ctrl)):
        raise FloatingPointError(f"Non-finite ctrl during {context}")


def _reset_data(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def _metric_from_trace(
    joint_name: str,
    target_position: float,
    positions: Sequence[float],
    torques: Sequence[float],
    torque_limits: Sequence[float],
) -> StepMetric:
    position_values = np.asarray(positions, dtype=float)
    torque_values = np.asarray(torques, dtype=float)
    if position_values.size == 0 or torque_values.size == 0:
        raise ValueError(f"Trace for {joint_name!r} is empty")
    lower, upper = float(torque_limits[0]), float(torque_limits[1])
    saturated = (torque_values <= lower + 1e-12) | (torque_values >= upper - 1e-12)
    final_position = float(position_values[-1])
    return StepMetric(
        joint_name=joint_name,
        target_position=float(target_position),
        final_position=final_position,
        steady_state_error=float(target_position - final_position),
        peak_torque=float(np.max(np.abs(torque_values))),
        saturation_fraction=float(np.mean(saturated)),
    )


def run_fixed_base_step_response(
    model: mujoco.MjModel,
    duration: float = 1.0,
    step_size: float = 0.1,
    wheel_step_size: float = 0.25,
) -> StepResponseResult:
    """Step each controlled joint target with the base weld enabled."""
    joint_map = build_joint_map(model)
    metrics: list[StepMetric] = []
    traces: dict[str, dict[str, list[float]]] = {}
    warning_count = 0
    timestep = float(model.opt.timestep)
    steps = max(1, int(np.ceil(float(duration) / timestep)))

    for entry in joint_map:
        data = mujoco.MjData(model)
        _reset_data(model, data)
        set_base_weld_active(model, data, True)
        targets = home_targets(model, joint_map)
        increment = wheel_step_size if "wheel" in entry.joint_name else step_size
        targets[entry.joint_name] = targets[entry.joint_name] + float(increment)
        gains = default_pd_gains(model, data)

        times: list[float] = []
        positions: list[float] = []
        torques: list[float] = []
        for _ in range(steps):
            ctrl = apply_pd_control(model, data, joint_map, targets, gains)
            _require_finite(data, f"fixed-base control for {entry.joint_name}")
            times.append(float(data.time))
            positions.append(float(data.qpos[entry.qposadr]))
            torques.append(float(ctrl[entry.actuator_id]))
            mujoco.mj_step(model, data)
            _require_finite(data, f"fixed-base step for {entry.joint_name}")

        warning_count += _warning_count(data)
        traces[entry.joint_name] = {
            "time": times,
            "position": positions,
            "torque": torques,
            "target": [float(targets[entry.joint_name])] * len(times),
        }
        metrics.append(
            _metric_from_trace(
                entry.joint_name,
                float(targets[entry.joint_name]),
                positions,
                torques,
                model.actuator_ctrlrange[entry.actuator_id],
            )
        )

    return StepResponseResult(
        duration=float(duration),
        timestep=timestep,
        warning_count=warning_count,
        metrics=metrics,
        traces=traces,
    )


def run_free_base_posture_check(
    model: mujoco.MjModel,
    duration: float = 1.0,
) -> FreeBaseResult:
    """Hold the home posture with the base weld disabled and check finite motion."""
    data = mujoco.MjData(model)
    _reset_data(model, data)
    set_base_weld_active(model, data, False)
    joint_map = build_joint_map(model)
    targets = home_targets(model, joint_map)
    gains = default_pd_gains(model, data)
    timestep = float(model.opt.timestep)
    steps = max(1, int(np.ceil(float(duration) / timestep)))
    peak_abs_qvel = 0.0
    peak_abs_ctrl = 0.0

    for _ in range(steps):
        ctrl = apply_pd_control(model, data, joint_map, targets, gains)
        _require_finite(data, "free-base control")
        peak_abs_qvel = max(peak_abs_qvel, float(np.max(np.abs(data.qvel))))
        peak_abs_ctrl = max(peak_abs_ctrl, float(np.max(np.abs(ctrl))) if ctrl.size else 0.0)
        mujoco.mj_step(model, data)
        _require_finite(data, "free-base step")

    return FreeBaseResult(
        duration=float(duration),
        timestep=timestep,
        steps=steps,
        warning_count=_warning_count(data),
        peak_abs_qvel=peak_abs_qvel,
        peak_abs_ctrl=peak_abs_ctrl,
        final_base_height=float(data.qpos[2]),
    )


def write_results(
    step_result: StepResponseResult,
    free_result: FreeBaseResult,
    output_dir: Path = DEFAULT_RESULTS,
) -> Path:
    """Write planned CSV summaries, Markdown report, and optional plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_fields = [
        "joint_name",
        "target_position",
        "final_position",
        "steady_state_error",
        "peak_torque",
        "saturation_fraction",
    ]
    with (output_dir / "step_response_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=metric_fields)
        writer.writeheader()
        for metric in step_result.metrics:
            writer.writerow(asdict(metric))

    free_fields = [
        "duration",
        "timestep",
        "steps",
        "warning_count",
        "peak_abs_qvel",
        "peak_abs_ctrl",
        "final_base_height",
    ]
    with (output_dir / "free_base_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=free_fields)
        writer.writeheader()
        writer.writerow(asdict(free_result))

    _write_report(output_dir / "dynamics_report.md", step_result, free_result)
    _write_plot_if_available(output_dir / "step_response.png", step_result)
    return output_dir


def _write_report(path: Path, step_result: StepResponseResult, free_result: FreeBaseResult) -> None:
    lines = [
        "# Motor Dynamics Analysis",
        "",
        "## Fixed-base step response",
        "",
        f"- Duration: {step_result.duration:.6g} s",
        f"- Timestep: {step_result.timestep:.6g} s",
        f"- MuJoCo warnings: {step_result.warning_count}",
        "",
        "| Joint | Target | Final | Error | Peak torque | Saturation |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metric in step_result.metrics:
        lines.append(
            "| "
            f"{metric.joint_name} | "
            f"{metric.target_position:.6g} | "
            f"{metric.final_position:.6g} | "
            f"{metric.steady_state_error:.6g} | "
            f"{metric.peak_torque:.6g} | "
            f"{metric.saturation_fraction:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Free-base posture check",
            "",
            f"- Duration: {free_result.duration:.6g} s",
            f"- Steps: {free_result.steps}",
            f"- MuJoCo warnings: {free_result.warning_count}",
            f"- Peak |qvel|: {free_result.peak_abs_qvel:.6g}",
            f"- Peak |ctrl|: {free_result.peak_abs_ctrl:.6g}",
            f"- Final base height: {free_result.final_base_height:.6g} m",
            (
                "- Interpretation: Balance control is not implemented; "
                "free-base falling is allowed when the simulation remains finite "
                "and MuJoCo reports no warnings."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_plot_if_available(path: Path, step_result: StepResponseResult) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    figure, axis = plt.subplots(figsize=(10, 6))
    for joint_name, trace in step_result.traces.items():
        axis.plot(trace["time"], trace["position"], label=joint_name)
    axis.set_title("Fixed-base joint step responses")
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Joint position [rad]")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize="small", ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--duration", type=float, default=1.0)
    args = parser.parse_args(argv)

    model_path = convert_urdf(args.source, args.model)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    step_result = run_fixed_base_step_response(model, duration=args.duration)
    free_result = run_free_base_posture_check(model, duration=args.duration)
    output_dir = write_results(step_result, free_result, args.output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
