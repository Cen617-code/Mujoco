"""Analyze first-pass free-base body balance control."""

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

from scripts.balance_control import BalanceConfig, apply_balance_control
from scripts.convert_urdf_to_mjcf import ROOT, convert_urdf
from scripts.pd_control import build_joint_map, set_base_weld_active


DEFAULT_SOURCE = ROOT / "8dof_URDF" / "urdf" / "robot.urdf"
DEFAULT_MODEL = ROOT / "8dof_URDF" / "mjcf" / "robot.xml"
DEFAULT_RESULTS = ROOT / "analysis" / "balance_results"


@dataclass(frozen=True)
class BalanceSimulationResult:
    duration: float
    timestep: float
    steps: int
    warning_count: int
    finite: bool
    peak_abs_pitch: float
    final_pitch: float
    peak_abs_pitch_rate: float
    peak_abs_wheel_torque: float
    final_base_height: float
    timeseries: list[dict[str, float]]


def _warning_count(data: mujoco.MjData) -> int:
    return int(sum(int(warning.number) for warning in data.warning))


def _finite(data: mujoco.MjData, ctrl: np.ndarray) -> bool:
    return bool(
        np.isfinite(data.qpos).all()
        and np.isfinite(data.qvel).all()
        and np.isfinite(ctrl).all()
    )


def run_balance_simulation(
    model: mujoco.MjModel,
    duration: float = 2.0,
    config: BalanceConfig | None = None,
) -> BalanceSimulationResult:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    set_base_weld_active(model, data, False)
    joint_map = build_joint_map(model)
    timestep = float(model.opt.timestep)
    steps = max(1, int(np.ceil(float(duration) / timestep)))
    finite = True
    timeseries: list[dict[str, float]] = []
    peak_abs_pitch = 0.0
    peak_abs_pitch_rate = 0.0
    peak_abs_wheel_torque = 0.0
    final_pitch = 0.0

    for _ in range(steps):
        ctrl, state = apply_balance_control(model, data, joint_map, config)
        finite = finite and _finite(data, ctrl)
        timeseries.append(
            {
                "time": float(data.time),
                "pitch": state.pitch,
                "pitch_rate": state.pitch_rate,
                "x": state.x,
                "x_velocity": state.x_velocity,
                "wheel_torque": state.wheel_torque,
                "base_height": float(data.qpos[2]),
            }
        )
        peak_abs_pitch = max(peak_abs_pitch, abs(state.pitch))
        peak_abs_pitch_rate = max(peak_abs_pitch_rate, abs(state.pitch_rate))
        peak_abs_wheel_torque = max(peak_abs_wheel_torque, abs(state.wheel_torque))
        final_pitch = state.pitch
        mujoco.mj_step(model, data)
        finite = finite and _finite(data, ctrl)

    return BalanceSimulationResult(
        duration=float(duration),
        timestep=timestep,
        steps=steps,
        warning_count=_warning_count(data),
        finite=finite,
        peak_abs_pitch=peak_abs_pitch,
        final_pitch=final_pitch,
        peak_abs_pitch_rate=peak_abs_pitch_rate,
        peak_abs_wheel_torque=peak_abs_wheel_torque,
        final_base_height=float(data.qpos[2]),
        timeseries=timeseries,
    )


def write_balance_results(
    result: BalanceSimulationResult,
    output_dir: Path = DEFAULT_RESULTS,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_fields = [
        "duration",
        "timestep",
        "steps",
        "warning_count",
        "finite",
        "peak_abs_pitch",
        "final_pitch",
        "peak_abs_pitch_rate",
        "peak_abs_wheel_torque",
        "final_base_height",
    ]
    with (output_dir / "balance_summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        summary = asdict(result)
        summary.pop("timeseries")
        writer.writerow(summary)

    timeseries_fields = ["time", "pitch", "pitch_rate", "x", "x_velocity", "wheel_torque", "base_height"]
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
        f"- Peak |pitch|: {result.peak_abs_pitch:.6g} rad",
        f"- Final pitch: {result.final_pitch:.6g} rad",
        f"- Peak |pitch rate|: {result.peak_abs_pitch_rate:.6g} rad/s",
        f"- Peak |wheel torque|: {result.peak_abs_wheel_torque:.6g} N·m",
        f"- Final base height: {result.final_base_height:.6g} m",
        "",
        "This is a first-pass in-place balance prototype, not walking or trajectory tracking.",
        "",
    ]
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
