"""Run the MuJoCo viewer with the Python body balance controller.

直接用 ``python -m mujoco.viewer --mjcf=...`` 只能看被动 XML。
这个脚本在 viewer 循环里每步调用 Python 平衡控制器，所以能看到受控仿真。
"""

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
from scripts.analyze_disturbance import DEFAULT_PUSH_DURATION, DEFAULT_PUSH_START
from scripts.balance_control import (
    apply_balance_control,
    default_standing_config,
    standing_leg_targets,
)
from scripts.convert_urdf_to_mjcf import convert_urdf
from scripts.pd_control import build_joint_map, set_base_weld_active


def run_viewer(
    model_path: Path,
    duration: float | None = None,
    *,
    push_force: float = 0.0,
    push_start: float = DEFAULT_PUSH_START,
    push_duration: float = DEFAULT_PUSH_DURATION,
) -> None:
    """加载模型，关闭固定基座 weld，并在 viewer 循环中实时写入控制力矩。"""
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.xfrc_applied[:] = 0.0
    set_base_weld_active(model, data, False)
    joint_map = build_joint_map(model)
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if base_body_id < 0:
        raise ValueError("Model is missing body 'base_link'")
    config = default_standing_config()
    if config.x_target is None:
        from dataclasses import replace

        config = replace(config, x_target=float(data.qpos[0]))
    leg_targets = standing_leg_targets()
    start = time.time()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # launch_passive 不会自动推进仿真；这里手动控制、step、sync。
            data.xfrc_applied[:] = 0.0
            if push_start <= float(data.time) < push_start + push_duration:
                data.xfrc_applied[base_body_id, 0] = float(push_force)
            apply_balance_control(model, data, joint_map, config, leg_targets)
            mujoco.mj_step(model, data)
            if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
                raise FloatingPointError("Non-finite state in balance viewer")
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
    parser.add_argument("--push-force", type=float, default=0.0)
    parser.add_argument("--push-start", type=float, default=DEFAULT_PUSH_START)
    parser.add_argument("--push-duration", type=float, default=DEFAULT_PUSH_DURATION)
    args = parser.parse_args(argv)
    model_path = args.model if args.no_regenerate else convert_urdf(args.source, args.model)
    run_viewer(
        model_path,
        duration=args.duration,
        push_force=args.push_force,
        push_start=args.push_start,
        push_duration=args.push_duration,
    )


if __name__ == "__main__":
    main()
