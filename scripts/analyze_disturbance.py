"""Analyze push-disturbance rejection for the standing wheeled biped.

The disturbance v1 benchmark applies a short horizontal force to ``base_link``
while the existing standing controller is running.  It is intentionally narrow:
front/back pushes only, no side pushes, walking, or random disturbances yet.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_balance import (
    ALLOWED_STANDING_GROUND_CONTACT_GEOMS,
    DEFAULT_MODEL,
    DEFAULT_SOURCE,
    TORQUE_LIMIT_EPSILON,
    WHEEL_TORQUE_LIMIT_NM,
)
from scripts.balance_control import (
    apply_balance_control,
    base_pitch,
    base_pitch_rate,
    default_standing_config,
    standing_leg_targets,
)
from scripts.convert_urdf_to_mjcf import ROOT, convert_urdf
from scripts.pd_control import build_joint_map, set_base_weld_active


DEFAULT_RESULTS = ROOT / "analysis" / "disturbance_results"
DEFAULT_PUSH_FORCES = (-50.0, 50.0)
DEFAULT_PUSH_START = 1.0
DEFAULT_PUSH_DURATION = 0.1
DEFAULT_DURATION = 6.0
DISTURBANCE_FINAL_ABS_PITCH_LIMIT = 0.18
DISTURBANCE_PEAK_ABS_PITCH_LIMIT = 0.45
DISTURBANCE_PEAK_ABS_X_DRIFT_LIMIT = 0.5
DISTURBANCE_WHEEL_SATURATION_LIMIT = 0.2


@dataclass(frozen=True)
class DisturbanceSimulationResult:
    """Summary metrics and timeseries for one push-disturbance scenario."""

    push_force: float
    push_start: float
    push_duration: float
    push_impulse: float
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
    non_wheel_ground_contact_count: int
    non_wheel_ground_contact_geoms: str
    final_base_height: float
    meets_disturbance_objective: bool
    disturbance_score: float
    timeseries: list[dict[str, float]]


def _warning_count(data: mujoco.MjData) -> int:
    return int(sum(int(warning.number) for warning in data.warning))


def _ground_contact_geom_names(model: mujoco.MjModel, data: mujoco.MjData) -> set[str]:
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    names: set[str] = set()
    for index in range(data.ncon):
        contact = data.contact[index]
        if floor not in (contact.geom1, contact.geom2):
            continue
        other = contact.geom2 if contact.geom1 == floor else contact.geom1
        names.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or "")
    return names


def _non_wheel_ground_contact_names(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> set[str]:
    return _ground_contact_geom_names(model, data) - ALLOWED_STANDING_GROUND_CONTACT_GEOMS


def _finite(data: mujoco.MjData, ctrl: np.ndarray) -> bool:
    return bool(
        np.isfinite(data.qpos).all()
        and np.isfinite(data.qvel).all()
        and np.isfinite(ctrl).all()
    )


def meets_disturbance_objective_values(
    *,
    warning_count: int,
    finite: bool,
    non_wheel_ground_contact_count: int,
    final_abs_pitch: float,
    peak_abs_pitch: float,
    peak_abs_x_drift: float,
    wheel_torque_saturation_fraction: float,
) -> bool:
    """Return whether one disturbance run meets the v1 recovery limits."""
    return (
        int(warning_count) == 0
        and bool(finite)
        and int(non_wheel_ground_contact_count) == 0
        and float(final_abs_pitch) < DISTURBANCE_FINAL_ABS_PITCH_LIMIT
        and float(peak_abs_pitch) < DISTURBANCE_PEAK_ABS_PITCH_LIMIT
        and float(peak_abs_x_drift) < DISTURBANCE_PEAK_ABS_X_DRIFT_LIMIT
        and float(wheel_torque_saturation_fraction) < DISTURBANCE_WHEEL_SATURATION_LIMIT
    )


def disturbance_score_values(
    *,
    warning_count: int,
    finite: bool,
    non_wheel_ground_contact_count: int,
    final_abs_pitch: float,
    peak_abs_pitch: float,
    peak_abs_x_drift: float,
    wheel_torque_saturation_fraction: float,
) -> float:
    """Lower-is-better scalar used only to rank comparable disturbance runs."""
    values = [
        float(final_abs_pitch),
        float(peak_abs_pitch),
        float(peak_abs_x_drift),
        float(wheel_torque_saturation_fraction),
    ]
    try:
        warning_count_value = int(warning_count)
        contact_count_value = int(non_wheel_ground_contact_count)
    except (TypeError, ValueError, OverflowError):
        return 1_000_000.0
    if (
        warning_count_value != 0
        or contact_count_value != 0
        or not bool(finite)
        or not np.isfinite(values).all()
    ):
        return 1_000_000.0
    return float(
        4.0 * float(final_abs_pitch)
        + 2.0 * float(peak_abs_pitch)
        + float(peak_abs_x_drift)
        + 0.5 * float(wheel_torque_saturation_fraction)
    )


def run_disturbance_simulation(
    model: mujoco.MjModel,
    *,
    push_force: float,
    push_start: float = DEFAULT_PUSH_START,
    push_duration: float = DEFAULT_PUSH_DURATION,
    duration: float = DEFAULT_DURATION,
) -> DisturbanceSimulationResult:
    """Run one standing-control simulation with a single base push."""
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.xfrc_applied[:] = 0.0
    set_base_weld_active(model, data, False)

    joint_map = build_joint_map(model)
    config = default_standing_config()
    if config.x_target is None:
        config = replace(config, x_target=float(data.qpos[0]))
    leg_targets = standing_leg_targets()
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if base_body_id < 0:
        raise ValueError("Model is missing body 'base_link'")

    timestep = float(model.opt.timestep)
    steps = max(1, int(np.ceil(float(duration) / timestep)))
    initial_x = float(data.qpos[0])
    finite = True
    non_wheel_ground_contact_count = 0
    non_wheel_ground_contact_names: set[str] = set()
    timeseries: list[dict[str, float]] = []

    for _ in range(steps):
        data.xfrc_applied[:] = 0.0
        applied_force = 0.0
        if float(push_start) <= float(data.time) < float(push_start + push_duration):
            applied_force = float(push_force)
            data.xfrc_applied[base_body_id, 0] = applied_force

        ctrl, applied_state = apply_balance_control(model, data, joint_map, config, leg_targets)
        finite = finite and _finite(data, ctrl)
        mujoco.mj_step(model, data)
        finite = finite and _finite(data, ctrl)

        bad_contacts = _non_wheel_ground_contact_names(model, data)
        non_wheel_ground_contact_count += len(bad_contacts)
        non_wheel_ground_contact_names.update(bad_contacts)

        pitch = base_pitch(data)
        pitch_rate = base_pitch_rate(data)
        x = float(data.qpos[0])
        x_drift = x - initial_x
        timeseries.append(
            {
                "time": float(data.time),
                "push_force": float(push_force),
                "applied_force": applied_force,
                "pitch": float(pitch),
                "pitch_rate": float(pitch_rate),
                "x": x,
                "x_velocity": float(data.qvel[0]),
                "x_drift": x_drift,
                "wheel_torque": float(applied_state.wheel_torque),
                "base_height": float(data.qpos[2]),
            }
        )

    final_sample = timeseries[-1]
    warning_count = _warning_count(data)
    final_pitch = float(final_sample["pitch"])
    final_abs_pitch = abs(final_pitch)
    peak_abs_pitch = max(abs(sample["pitch"]) for sample in timeseries)
    peak_abs_pitch_rate = max(abs(sample["pitch_rate"]) for sample in timeseries)
    peak_abs_x_drift = max(abs(sample["x_drift"]) for sample in timeseries)
    peak_abs_wheel_torque = max(abs(sample["wheel_torque"]) for sample in timeseries)
    saturated_samples = sum(
        sample["wheel_torque"] >= WHEEL_TORQUE_LIMIT_NM - TORQUE_LIMIT_EPSILON
        for sample in timeseries
    )
    wheel_torque_saturation_fraction = float(saturated_samples / len(timeseries))
    meets_objective = meets_disturbance_objective_values(
        warning_count=warning_count,
        finite=finite,
        non_wheel_ground_contact_count=non_wheel_ground_contact_count,
        final_abs_pitch=final_abs_pitch,
        peak_abs_pitch=peak_abs_pitch,
        peak_abs_x_drift=peak_abs_x_drift,
        wheel_torque_saturation_fraction=wheel_torque_saturation_fraction,
    )
    score = disturbance_score_values(
        warning_count=warning_count,
        finite=finite,
        non_wheel_ground_contact_count=non_wheel_ground_contact_count,
        final_abs_pitch=final_abs_pitch,
        peak_abs_pitch=peak_abs_pitch,
        peak_abs_x_drift=peak_abs_x_drift,
        wheel_torque_saturation_fraction=wheel_torque_saturation_fraction,
    )

    return DisturbanceSimulationResult(
        push_force=float(push_force),
        push_start=float(push_start),
        push_duration=float(push_duration),
        push_impulse=float(push_force * push_duration),
        duration=float(duration),
        timestep=timestep,
        steps=steps,
        warning_count=warning_count,
        finite=bool(finite),
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
        non_wheel_ground_contact_count=non_wheel_ground_contact_count,
        non_wheel_ground_contact_geoms=";".join(sorted(non_wheel_ground_contact_names)),
        final_base_height=float(final_sample["base_height"]),
        meets_disturbance_objective=meets_objective,
        disturbance_score=score,
        timeseries=timeseries,
    )


def run_default_disturbance_suite(
    model: mujoco.MjModel,
    *,
    push_forces: Sequence[float] = DEFAULT_PUSH_FORCES,
    push_start: float = DEFAULT_PUSH_START,
    push_duration: float = DEFAULT_PUSH_DURATION,
    duration: float = DEFAULT_DURATION,
) -> list[DisturbanceSimulationResult]:
    """Run the deterministic v1 front/back push suite."""
    return [
        run_disturbance_simulation(
            model,
            push_force=float(push_force),
            push_start=push_start,
            push_duration=push_duration,
            duration=duration,
        )
        for push_force in push_forces
    ]


def write_disturbance_results(
    results: Sequence[DisturbanceSimulationResult],
    output_dir: Path = DEFAULT_RESULTS,
) -> Path:
    """Write summary CSV, timeseries CSV, and a Markdown report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(results)
    summary_fields = [
        "push_force",
        "push_start",
        "push_duration",
        "push_impulse",
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
        "non_wheel_ground_contact_count",
        "non_wheel_ground_contact_geoms",
        "final_base_height",
        "meets_disturbance_objective",
        "disturbance_score",
    ]
    with (output_dir / "disturbance_summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        for result in rows:
            summary = asdict(result)
            summary.pop("timeseries")
            writer.writerow(summary)

    timeseries_fields = [
        "scenario",
        "time",
        "push_force",
        "applied_force",
        "pitch",
        "pitch_rate",
        "x",
        "x_velocity",
        "x_drift",
        "wheel_torque",
        "base_height",
    ]
    with (output_dir / "disturbance_timeseries.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=timeseries_fields)
        writer.writeheader()
        for result in rows:
            scenario = f"{result.push_force:+.6g}N"
            for sample in result.timeseries:
                row = {"scenario": scenario}
                row.update(sample)
                writer.writerow(row)

    default_force_text = ", ".join(f"{force:+.6g} N" for force in DEFAULT_PUSH_FORCES)
    lines = [
        "# Disturbance Rejection Analysis",
        "",
        f"- Default push forces: {default_force_text}",
        f"- Push start: {DEFAULT_PUSH_START:.6g} s",
        f"- Push duration: {DEFAULT_PUSH_DURATION:.6g} s",
        "",
        "| Push force | Objective | Peak |pitch| rad | Final |pitch| rad | Peak |x drift| m | Final x drift m | Bad contacts | Wheel saturation |",
        "| ---: | :---: | ---: | ---: | ---: | ---: | :--- | ---: |",
    ]
    for result in rows:
        lines.append(
            "| "
            f"{result.push_force:.6g} | "
            f"{result.meets_disturbance_objective} | "
            f"{result.peak_abs_pitch:.6g} | "
            f"{result.final_abs_pitch:.6g} | "
            f"{result.peak_abs_x_drift:.6g} | "
            f"{result.x_drift:.6g} | "
            f"{result.non_wheel_ground_contact_geoms or 'none'} | "
            f"{result.wheel_torque_saturation_fraction:.3f} |"
        )
    lines.extend(
        [
            "",
            "Disturbance v1 is a deterministic front/back push benchmark. It does not cover side pushes, random disturbances, walking, or uneven terrain.",
        ]
    )
    if any(not result.meets_disturbance_objective for result in rows):
        lines.extend(
            [
                "",
                "At least one run does not meet the disturbance rejection v1 objective.",
            ]
        )
    lines.append("")
    (output_dir / "disturbance_report.md").write_text("\n".join(lines), encoding="utf-8")
    return output_dir


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--push-forces", type=float, nargs="*", default=None)
    parser.add_argument("--push-start", type=float, default=DEFAULT_PUSH_START)
    parser.add_argument("--push-duration", type=float, default=DEFAULT_PUSH_DURATION)
    args = parser.parse_args(argv)

    model_path = convert_urdf(args.source, args.model)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    push_forces = DEFAULT_PUSH_FORCES if args.push_forces is None else tuple(args.push_forces)
    results = run_default_disturbance_suite(
        model,
        push_forces=push_forces,
        push_start=args.push_start,
        push_duration=args.push_duration,
        duration=args.duration,
    )
    output_dir = write_disturbance_results(results, args.output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
