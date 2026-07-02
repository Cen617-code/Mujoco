"""Run the MuJoCo viewer with wheel-speed walking control."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

import mujoco
import mujoco.viewer
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_balance import DEFAULT_MODEL, DEFAULT_SOURCE
from scripts.convert_urdf_to_mjcf import convert_urdf
from scripts.pd_control import build_joint_map, set_base_weld_active
from scripts.walking_control import (
    DEFAULT_FORWARD_VELOCITY,
    DEFAULT_RAMP_TIME,
    WalkingConfig,
    apply_walking_control,
)


def run_viewer(
    model_path: Path,
    duration: float | None = None,
    *,
    velocity: float = DEFAULT_FORWARD_VELOCITY,
    ramp_time: float = DEFAULT_RAMP_TIME,
) -> None:
    """Load the model and run walking control in a passive viewer loop."""
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    set_base_weld_active(model, data, False)

    joint_map = build_joint_map(model)
    config = WalkingConfig(forward_velocity=velocity, ramp_time=ramp_time)
    initial_x = float(data.qpos[0])
    start = time.time()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            apply_walking_control(
                model,
                data,
                joint_map,
                config=config,
                initial_x=initial_x,
            )
            mujoco.mj_step(model, data)
            if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
                raise FloatingPointError("Non-finite state in walking viewer")
            viewer.sync()
            if duration is not None and time.time() - start >= duration:
                break
            time.sleep(float(model.opt.timestep))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--no-regenerate", action="store_true")
    parser.add_argument("--velocity", type=float, default=DEFAULT_FORWARD_VELOCITY)
    parser.add_argument("--ramp-time", type=float, default=DEFAULT_RAMP_TIME)
    args = parser.parse_args(argv)

    model_path = args.model if args.no_regenerate else convert_urdf(args.source, args.model)
    run_viewer(
        model_path,
        duration=args.duration,
        velocity=args.velocity,
        ramp_time=args.ramp_time,
    )


if __name__ == "__main__":
    main()
