"""Analyze wheel-speed walking control for the wheeled biped."""

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

from scripts.analyze_balance import (
    ALLOWED_STANDING_GROUND_CONTACT_GEOMS,
    DEFAULT_MODEL,
    DEFAULT_SOURCE,
    TORQUE_LIMIT_EPSILON,
    WHEEL_TORQUE_LIMIT_NM,
)
from scripts.balance_control import base_pitch, base_pitch_rate, standing_leg_targets
from scripts.convert_urdf_to_mjcf import ROOT, convert_urdf
from scripts.pd_control import build_joint_map, set_base_weld_active
from scripts.walking_control import (
    DEFAULT_FORWARD_VELOCITY,
    DEFAULT_RAMP_TIME,
    WalkingConfig,
    apply_walking_control,
    compute_forward_distance,
    compute_forward_velocity,
)


DEFAULT_RESULTS = ROOT / "analysis" / "walking_results"
DEFAULT_DURATION = 8.0
DEFAULT_VELOCITY_WINDOW = 2.0
WALKING_PEAK_ABS_PITCH_LIMIT = 0.3
WALKING_MIN_FORWARD_DISTANCE = 1.0
WALKING_AVERAGE_VELOCITY_TOLERANCE = 0.08
WALKING_WHEEL_SATURATION_LIMIT = 0.2
WALKING_FAILURE_SCORE = 1_000_000.0


@dataclass(frozen=True)
class WalkingSimulationResult:
    """Summary metrics and timeseries for one walking run."""

    forward_velocity_target: float
    ramp_time: float
    duration: float
    timestep: float
    steps: int
    velocity_window: float
    warning_count: int
    finite: bool
    initial_x: float
    final_x: float
    x_displacement: float
    forward_distance: float
    final_forward_velocity: float
    average_forward_velocity_last_window: float
    average_forward_velocity_error: float
    peak_abs_pitch: float
    final_pitch: float
    final_abs_pitch: float
    peak_abs_pitch_rate: float
    peak_abs_wheel_torque: float
    wheel_torque_saturation_fraction: float
    non_wheel_ground_contact_count: int
    non_wheel_ground_contact_geoms: str
    final_base_height: float
    meets_walking_objective: bool
    walking_score: float
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


def meets_walking_objective_values(
    *,
    warning_count: int,
    finite: bool,
    non_wheel_ground_contact_count: int,
    peak_abs_pitch: float,
    forward_distance: float,
    average_forward_velocity_error: float,
    wheel_torque_saturation_fraction: float,
) -> bool:
    """Return whether a walking run meets the v1 objective."""
    return (
        int(warning_count) == 0
        and bool(finite)
        and int(non_wheel_ground_contact_count) == 0
        and float(peak_abs_pitch) < WALKING_PEAK_ABS_PITCH_LIMIT
        and float(forward_distance) > WALKING_MIN_FORWARD_DISTANCE
        and abs(float(average_forward_velocity_error)) < WALKING_AVERAGE_VELOCITY_TOLERANCE
        and float(wheel_torque_saturation_fraction) < WALKING_WHEEL_SATURATION_LIMIT
    )


def walking_score_values(
    *,
    warning_count: int,
    finite: bool,
    non_wheel_ground_contact_count: int,
    peak_abs_pitch: float,
    forward_distance: float,
    average_forward_velocity_error: float,
    wheel_torque_saturation_fraction: float,
) -> float:
    """Lower-is-better scalar for comparing walking runs."""
    values = [
        float(peak_abs_pitch),
        float(forward_distance),
        float(average_forward_velocity_error),
        float(wheel_torque_saturation_fraction),
    ]
    try:
        warning_count_value = int(warning_count)
        contact_count_value = int(non_wheel_ground_contact_count)
    except (TypeError, ValueError, OverflowError):
        return WALKING_FAILURE_SCORE
    if (
        warning_count_value != 0
        or contact_count_value != 0
        or not bool(finite)
        or not np.isfinite(values).all()
    ):
        return WALKING_FAILURE_SCORE
    distance_shortfall = max(0.0, WALKING_MIN_FORWARD_DISTANCE - float(forward_distance))
    return float(
        3.0 * float(peak_abs_pitch)
        + 5.0 * abs(float(average_forward_velocity_error))
        + 2.0 * distance_shortfall
        + 0.5 * float(wheel_torque_saturation_fraction)
    )


def run_walking_simulation(
    model: mujoco.MjModel,
    *,
    forward_velocity: float = DEFAULT_FORWARD_VELOCITY,
    ramp_time: float = DEFAULT_RAMP_TIME,
    duration: float = DEFAULT_DURATION,
    velocity_window: float = DEFAULT_VELOCITY_WINDOW,
) -> WalkingSimulationResult:
    """Run one deterministic wheel-speed walking simulation."""
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    set_base_weld_active(model, data, False)

    joint_map = build_joint_map(model)
    leg_targets = standing_leg_targets()
    config = WalkingConfig(forward_velocity=forward_velocity, ramp_time=ramp_time)
    timestep = float(model.opt.timestep)
    steps = max(1, int(np.ceil(float(duration) / timestep)))
    initial_x = float(data.qpos[0])
    finite = True
    non_wheel_ground_contact_count = 0
    non_wheel_ground_contact_names: set[str] = set()
    timeseries: list[dict[str, float]] = []

    for _ in range(steps):
        ctrl, walking_state = apply_walking_control(
            model,
            data,
            joint_map,
            config=config,
            leg_targets=leg_targets,
            initial_x=initial_x,
        )
        finite = finite and _finite(data, ctrl)
        mujoco.mj_step(model, data)
        finite = finite and _finite(data, ctrl)

        bad_contacts = _non_wheel_ground_contact_names(model, data)
        non_wheel_ground_contact_count += len(bad_contacts)
        non_wheel_ground_contact_names.update(bad_contacts)

        pitch = base_pitch(data)
        pitch_rate = base_pitch_rate(data)
        forward_distance = compute_forward_distance(data, initial_x)
        forward_velocity_value = compute_forward_velocity(data)
        timeseries.append(
            {
                "time": float(data.time),
                "forward_velocity_target": float(walking_state.forward_velocity_target),
                "balance_x_velocity_target": float(walking_state.balance_x_velocity_target),
                "forward_velocity": forward_velocity_value,
                "forward_distance": forward_distance,
                "x": float(data.qpos[0]),
                "x_velocity": float(data.qvel[0]),
                "pitch": float(pitch),
                "pitch_rate": float(pitch_rate),
                "wheel_torque": float(walking_state.balance_state.wheel_torque),
                "base_height": float(data.qpos[2]),
            }
        )

    final_sample = timeseries[-1]
    warning_count = _warning_count(data)
    window_start = max(0.0, float(duration) - float(velocity_window))
    window_samples = [
        sample for sample in timeseries if float(sample["time"]) >= window_start
    ] or timeseries
    average_forward_velocity = float(
        np.mean([sample["forward_velocity"] for sample in window_samples])
    )
    average_velocity_error = average_forward_velocity - float(forward_velocity)
    final_pitch = float(final_sample["pitch"])
    final_abs_pitch = abs(final_pitch)
    peak_abs_pitch = max(abs(sample["pitch"]) for sample in timeseries)
    peak_abs_pitch_rate = max(abs(sample["pitch_rate"]) for sample in timeseries)
    peak_abs_wheel_torque = max(abs(sample["wheel_torque"]) for sample in timeseries)
    saturated_samples = sum(
        sample["wheel_torque"] >= WHEEL_TORQUE_LIMIT_NM - TORQUE_LIMIT_EPSILON
        for sample in timeseries
    )
    wheel_torque_saturation_fraction = float(saturated_samples / len(timeseries))
    forward_distance = float(final_sample["forward_distance"])
    meets_objective = meets_walking_objective_values(
        warning_count=warning_count,
        finite=finite,
        non_wheel_ground_contact_count=non_wheel_ground_contact_count,
        peak_abs_pitch=peak_abs_pitch,
        forward_distance=forward_distance,
        average_forward_velocity_error=average_velocity_error,
        wheel_torque_saturation_fraction=wheel_torque_saturation_fraction,
    )
    score = walking_score_values(
        warning_count=warning_count,
        finite=finite,
        non_wheel_ground_contact_count=non_wheel_ground_contact_count,
        peak_abs_pitch=peak_abs_pitch,
        forward_distance=forward_distance,
        average_forward_velocity_error=average_velocity_error,
        wheel_torque_saturation_fraction=wheel_torque_saturation_fraction,
    )

    return WalkingSimulationResult(
        forward_velocity_target=float(forward_velocity),
        ramp_time=float(ramp_time),
        duration=float(duration),
        timestep=timestep,
        steps=steps,
        velocity_window=float(velocity_window),
        warning_count=warning_count,
        finite=bool(finite),
        initial_x=initial_x,
        final_x=float(final_sample["x"]),
        x_displacement=float(final_sample["x"] - initial_x),
        forward_distance=forward_distance,
        final_forward_velocity=float(final_sample["forward_velocity"]),
        average_forward_velocity_last_window=average_forward_velocity,
        average_forward_velocity_error=average_velocity_error,
        peak_abs_pitch=peak_abs_pitch,
        final_pitch=final_pitch,
        final_abs_pitch=final_abs_pitch,
        peak_abs_pitch_rate=peak_abs_pitch_rate,
        peak_abs_wheel_torque=peak_abs_wheel_torque,
        wheel_torque_saturation_fraction=wheel_torque_saturation_fraction,
        non_wheel_ground_contact_count=non_wheel_ground_contact_count,
        non_wheel_ground_contact_geoms=";".join(sorted(non_wheel_ground_contact_names)),
        final_base_height=float(final_sample["base_height"]),
        meets_walking_objective=meets_objective,
        walking_score=score,
        timeseries=timeseries,
    )


def write_walking_results(
    result: WalkingSimulationResult,
    output_dir: Path = DEFAULT_RESULTS,
) -> Path:
    """Write summary CSV, timeseries CSV, and a Markdown walking report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_fields = [
        "forward_velocity_target",
        "ramp_time",
        "duration",
        "timestep",
        "steps",
        "velocity_window",
        "warning_count",
        "finite",
        "initial_x",
        "final_x",
        "x_displacement",
        "forward_distance",
        "final_forward_velocity",
        "average_forward_velocity_last_window",
        "average_forward_velocity_error",
        "peak_abs_pitch",
        "final_pitch",
        "final_abs_pitch",
        "peak_abs_pitch_rate",
        "peak_abs_wheel_torque",
        "wheel_torque_saturation_fraction",
        "non_wheel_ground_contact_count",
        "non_wheel_ground_contact_geoms",
        "final_base_height",
        "meets_walking_objective",
        "walking_score",
    ]
    with (output_dir / "walking_summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        summary = asdict(result)
        summary.pop("timeseries")
        writer.writerow(summary)

    timeseries_fields = [
        "time",
        "forward_velocity_target",
        "balance_x_velocity_target",
        "forward_velocity",
        "forward_distance",
        "x",
        "x_velocity",
        "pitch",
        "pitch_rate",
        "wheel_torque",
        "base_height",
    ]
    with (output_dir / "walking_timeseries.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=timeseries_fields)
        writer.writeheader()
        writer.writerows(result.timeseries)

    lines = [
        "# Walking Control Analysis",
        "",
        "Walking v1 keeps the standing leg pose and uses wheel-speed control.",
        "The current convention is: positive forward velocity maps to world -X.",
        "",
        f"- Target forward velocity: {result.forward_velocity_target:.6g} m/s",
        f"- Ramp time: {result.ramp_time:.6g} s",
        f"- Duration: {result.duration:.6g} s",
        f"- Velocity averaging window: {result.velocity_window:.6g} s",
        f"- MuJoCo warnings: {result.warning_count}",
        f"- Finite state: {result.finite}",
        f"- Walking objective met: {result.meets_walking_objective}",
        f"- Walking score: {result.walking_score:.6g}",
        f"- Forward distance: {result.forward_distance:.6g} m",
        f"- World X displacement: {result.x_displacement:.6g} m",
        f"- Final forward velocity: {result.final_forward_velocity:.6g} m/s",
        f"- Average forward velocity last window: {result.average_forward_velocity_last_window:.6g} m/s",
        f"- Average forward velocity error: {result.average_forward_velocity_error:.6g} m/s",
        f"- Peak |pitch|: {result.peak_abs_pitch:.6g} rad",
        f"- Final pitch: {result.final_pitch:.6g} rad",
        f"- Peak |pitch rate|: {result.peak_abs_pitch_rate:.6g} rad/s",
        f"- Peak |wheel torque|: {result.peak_abs_wheel_torque:.6g} N·m",
        f"- Wheel torque saturation fraction: {result.wheel_torque_saturation_fraction:.3f}",
        f"- Non-wheel ground contact count: {result.non_wheel_ground_contact_count}",
        f"- Non-wheel ground contact geoms: {result.non_wheel_ground_contact_geoms or 'none'}",
        f"- Final base height: {result.final_base_height:.6g} m",
        "",
        "This is wheel-speed walking v1, not legged gait generation or turning.",
    ]
    if not result.meets_walking_objective:
        lines.extend(
            [
                "",
                "This run does not meet the walking v1 objective.",
            ]
        )
    lines.append("")
    (output_dir / "walking_report.md").write_text("\n".join(lines), encoding="utf-8")
    return output_dir


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--velocity", type=float, default=DEFAULT_FORWARD_VELOCITY)
    parser.add_argument("--ramp-time", type=float, default=DEFAULT_RAMP_TIME)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--velocity-window", type=float, default=DEFAULT_VELOCITY_WINDOW)
    args = parser.parse_args(argv)

    model_path = convert_urdf(args.source, args.model)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    result = run_walking_simulation(
        model,
        forward_velocity=args.velocity,
        ramp_time=args.ramp_time,
        duration=args.duration,
        velocity_window=args.velocity_window,
    )
    output_dir = write_walking_results(result, args.output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
