"""Tune fixed-posture wheeled standing balance with deterministic grid search."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

import mujoco

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_balance import (
    DEFAULT_MODEL,
    DEFAULT_SOURCE,
    BalanceSimulationResult,
    run_balance_simulation,
)
from scripts.balance_control import BalanceConfig, standing_leg_targets
from scripts.convert_urdf_to_mjcf import ROOT, convert_urdf


DEFAULT_RESULTS = ROOT / "analysis" / "standing_tuning"


@dataclass(frozen=True)
class StandingCandidate:
    kp_pitch: float
    kd_pitch: float
    kx: float
    kv: float
    hip_pitch: float
    knee: float
    leg_kp: float = 60.0
    leg_kd: float = 4.0

    def config(self) -> BalanceConfig:
        return BalanceConfig(
            pitch_target=0.0,
            pitch_rate_target=0.0,
            x_target=None,
            x_velocity_target=0.0,
            kp_pitch=float(self.kp_pitch),
            kd_pitch=float(self.kd_pitch),
            kx=float(self.kx),
            kv=float(self.kv),
            leg_kp=float(self.leg_kp),
            leg_kd=float(self.leg_kd),
        )

    def leg_targets(self) -> dict[str, float]:
        return standing_leg_targets(hip_pitch=float(self.hip_pitch), knee=float(self.knee))


def candidate_grid() -> list[StandingCandidate]:
    return [
        StandingCandidate(
            kp_pitch=kp_pitch,
            kd_pitch=kd_pitch,
            kx=kx,
            kv=kv,
            hip_pitch=hip_pitch,
            knee=knee,
            leg_kp=leg_kp,
            leg_kd=leg_kd,
        )
        for kp_pitch, kd_pitch, kx, kv, hip_pitch, knee, leg_kp, leg_kd in product(
            [15.0, 20.0, 35.0],
            [4.0, 6.0],
            [0.0, 16.0, 20.0],
            [10.0, 12.0],
            [0.0, -0.1, -0.2],
            [0.1, 0.18, 0.25],
            [60.0],
            [4.0],
        )
    ]


def _row_from_result(
    rank: int,
    candidate: StandingCandidate,
    result: BalanceSimulationResult,
) -> dict[str, float | int | bool | str]:
    row: dict[str, float | int | bool | str] = {
        "rank": int(rank),
        **asdict(candidate),
        "warning_count": int(result.warning_count),
        "finite": bool(result.finite),
        "non_wheel_ground_contact_count": int(result.non_wheel_ground_contact_count),
        "non_wheel_ground_contact_geoms": result.non_wheel_ground_contact_geoms,
        "final_abs_pitch": float(result.final_abs_pitch),
        "peak_abs_pitch": float(result.peak_abs_pitch),
        "peak_abs_x_drift": float(result.peak_abs_x_drift),
        "wheel_torque_saturation_fraction": float(result.wheel_torque_saturation_fraction),
        "standing_score": float(result.standing_score),
        "meets_standing_objective": bool(result.meets_standing_objective),
    }
    return row


def run_tuning(
    model: mujoco.MjModel,
    duration: float = 2.0,
    candidates: Iterable[StandingCandidate] | None = None,
) -> tuple[list[dict[str, float | int | bool | str]], StandingCandidate | None]:
    candidate_list = list(candidate_grid() if candidates is None else candidates)
    scored: list[tuple[StandingCandidate, BalanceSimulationResult]] = []
    for candidate in candidate_list:
        result = run_balance_simulation(
            model,
            duration=duration,
            config=candidate.config(),
            leg_targets=candidate.leg_targets(),
        )
        scored.append((candidate, result))

    scored.sort(key=lambda item: item[1].standing_score)
    rows = [
        _row_from_result(rank=index + 1, candidate=candidate, result=result)
        for index, (candidate, result) in enumerate(scored)
    ]
    best = scored[0][0] if scored else None
    return rows, best


def write_tuning_results(
    rows: list[dict[str, float | int | bool | str]],
    best: StandingCandidate | None,
    output_dir: Path = DEFAULT_RESULTS,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "kp_pitch",
        "kd_pitch",
        "kx",
        "kv",
        "hip_pitch",
        "knee",
        "leg_kp",
        "leg_kd",
        "warning_count",
        "finite",
        "non_wheel_ground_contact_count",
        "non_wheel_ground_contact_geoms",
        "final_abs_pitch",
        "peak_abs_pitch",
        "peak_abs_x_drift",
        "wheel_torque_saturation_fraction",
        "standing_score",
        "meets_standing_objective",
    ]
    with (output_dir / "standing_tuning_results.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best_row = rows[0] if rows else None
    best_payload = {
        "candidate": asdict(best) if best is not None else None,
        "result": best_row,
    }
    (output_dir / "standing_best_config.json").write_text(
        json.dumps(best_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    objective_met = bool(best_row["meets_standing_objective"]) if best_row else False
    lines = [
        "# Robust Standing Tuning",
        "",
        f"- Candidates evaluated: {len(rows)}",
        f"- Objective met: {objective_met}",
    ]
    if best_row:
        lines.extend(
            [
                f"- Best score: {float(best_row['standing_score']):.6g}",
                f"- Best non-wheel ground contact count: {int(best_row.get('non_wheel_ground_contact_count', 0))}",
                f"- Best non-wheel ground contact geoms: {best_row.get('non_wheel_ground_contact_geoms') or 'none'}",
                f"- Best final |pitch|: {float(best_row['final_abs_pitch']):.6g} rad",
                f"- Best peak |pitch|: {float(best_row['peak_abs_pitch']):.6g} rad",
                f"- Best peak |x drift|: {float(best_row['peak_abs_x_drift']):.6g} m",
                f"- Best wheel saturation fraction: {float(best_row['wheel_torque_saturation_fraction']):.3f}",
                "",
                "Best candidate:",
                "",
                "```json",
                json.dumps(best_payload["candidate"], indent=2, sort_keys=True),
                "```",
            ]
        )
    if not objective_met:
        lines.extend(
            [
                "",
                "The best candidate does not yet meet robust-standing v1 acceptance criteria.",
            ]
        )
    lines.append("")
    (output_dir / "standing_tuning_report.md").write_text("\n".join(lines), encoding="utf-8")
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
    rows, best = run_tuning(model, duration=args.duration)
    output_dir = write_tuning_results(rows, best, args.output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
