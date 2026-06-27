# Body Balance Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-pass free-base body pitch balance controller that uses wheel torque for upright stabilization while leg joints hold posture with PD.

**Architecture:** Keep the MJCF unchanged and implement closed-loop control in Python. `scripts/balance_control.py` owns state extraction and torque computation, `scripts/analyze_balance.py` owns automated simulation/reporting, and `scripts/run_balance_viewer.py` owns interactive visualization with the same controller.

**Tech Stack:** Python 3.10, MuJoCo Python API, NumPy, pytest, GitHub-flavored Markdown, existing `.venv`.

---

## File Structure

- Create `scripts/balance_control.py`
  - `BalanceConfig`
  - `BalanceState`
  - `quat_to_pitch`
  - `base_pitch`
  - `base_pitch_rate`
  - `default_balance_config`
  - `compute_balance_control`
  - `apply_balance_control`
- Create `scripts/analyze_balance.py`
  - free-base balance simulation
  - CSV/Markdown output under `analysis/balance_results/`
  - direct script and `python -m` compatible CLI
- Create `scripts/run_balance_viewer.py`
  - launches MuJoCo viewer and applies the Python balance controller each step
- Create `tests/test_balance_control.py`
  - unit and short integration tests for orientation, torque direction, saturation, finite simulation, and analysis artifacts
- Modify `README.md`
  - add balance-analysis and controlled-viewer commands
- Generate `analysis/balance_results/`
  - `balance_summary.csv`
  - `balance_timeseries.csv`
  - `balance_report.md`

---

### Task 1: Balance controller core

**Files:**
- Create: `scripts/balance_control.py`
- Create: `tests/test_balance_control.py`

- [ ] **Step 1: Write failing tests for pitch extraction and wheel torque**

Create `tests/test_balance_control.py` with:

```python
from pathlib import Path

import mujoco
import numpy as np
import pytest

from scripts.convert_urdf_to_mjcf import convert_urdf
from scripts.pd_control import build_joint_map
from scripts.balance_control import (
    BalanceConfig,
    apply_balance_control,
    base_pitch,
    compute_balance_control,
    quat_to_pitch,
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
    assert ctrl[left_wheel.actuator_id] == pytest.approx(-10.0)
    assert ctrl[right_wheel.actuator_id] == pytest.approx(-10.0)


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
```

- [ ] **Step 2: Run tests and verify import failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py::test_quat_to_pitch_extracts_small_y_axis_rotation -v
```

Expected: fail with `ModuleNotFoundError: No module named 'scripts.balance_control'`.

- [ ] **Step 3: Implement `scripts/balance_control.py`**

Create `scripts/balance_control.py`:

```python
"""First-pass body pitch balance controller for the wheeled biped."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import mujoco
import numpy as np

from scripts.pd_control import (
    JointControlMap,
    clip_targets_to_joint_limits,
    home_targets,
)


LEG_JOINTS = {
    "left_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "right_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
}
WHEEL_JOINTS = {"left_wheel_joint", "right_wheel_joint"}


@dataclass(frozen=True)
class BalanceConfig:
    pitch_target: float = 0.0
    pitch_rate_target: float = 0.0
    x_target: float | None = None
    x_velocity_target: float = 0.0
    kp_pitch: float = 35.0
    kd_pitch: float = 4.0
    kx: float = 0.0
    kv: float = 1.0
    leg_kp: float = 20.0
    leg_kd: float = 1.0


@dataclass(frozen=True)
class BalanceState:
    pitch: float
    pitch_rate: float
    x: float
    x_velocity: float
    wheel_torque: float


def quat_to_pitch(quat_wxyz) -> float:
    w, x, y, z = [float(value) for value in quat_wxyz]
    value = 2.0 * (w * y - z * x)
    return float(np.arcsin(np.clip(value, -1.0, 1.0)))


def base_pitch(data: mujoco.MjData) -> float:
    return quat_to_pitch(data.qpos[3:7])


def base_pitch_rate(data: mujoco.MjData) -> float:
    return float(data.qvel[4])


def default_balance_config() -> BalanceConfig:
    return BalanceConfig()


def _actuator_limits(model: mujoco.MjModel, entry: JointControlMap) -> tuple[float, float]:
    lower, upper = model.actuator_ctrlrange[entry.actuator_id]
    return float(lower), float(upper)


def compute_balance_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_map: list[JointControlMap],
    config: BalanceConfig | None = None,
    leg_targets: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, BalanceState]:
    config = config or default_balance_config()
    targets = home_targets(model, joint_map)
    if leg_targets:
        targets.update(leg_targets)
    targets = clip_targets_to_joint_limits(model, joint_map, targets)

    ctrl = np.zeros(model.nu, dtype=float)
    for entry in joint_map:
        if entry.joint_name not in LEG_JOINTS:
            continue
        q = float(data.qpos[entry.qposadr])
        qdot = float(data.qvel[entry.dofadr])
        tau = config.leg_kp * (targets[entry.joint_name] - q) - config.leg_kd * qdot
        lower, upper = _actuator_limits(model, entry)
        ctrl[entry.actuator_id] = np.clip(tau, lower, upper)

    pitch = base_pitch(data)
    pitch_rate = base_pitch_rate(data)
    x = float(data.qpos[0])
    x_velocity = float(data.qvel[0])
    x_target = x if config.x_target is None else float(config.x_target)
    tau_balance = (
        config.kp_pitch * (config.pitch_target - pitch)
        + config.kd_pitch * (config.pitch_rate_target - pitch_rate)
        + config.kx * (x_target - x)
        + config.kv * (config.x_velocity_target - x_velocity)
    )
    wheel_torque = 0.0
    for entry in joint_map:
        if entry.joint_name not in WHEEL_JOINTS:
            continue
        lower, upper = _actuator_limits(model, entry)
        ctrl[entry.actuator_id] = np.clip(tau_balance, lower, upper)
        wheel_torque = float(ctrl[entry.actuator_id])

    state = BalanceState(
        pitch=float(pitch),
        pitch_rate=float(pitch_rate),
        x=x,
        x_velocity=x_velocity,
        wheel_torque=wheel_torque,
    )
    return ctrl, state


def apply_balance_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_map: list[JointControlMap],
    config: BalanceConfig | None = None,
    leg_targets: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, BalanceState]:
    ctrl, state = compute_balance_control(model, data, joint_map, config, leg_targets)
    data.ctrl[:] = ctrl
    return ctrl, state
```

- [ ] **Step 4: Run Task 1 tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py -v
```

Expected: all tests in `tests/test_balance_control.py` pass.

- [ ] **Step 5: Commit**

Run:

```powershell
git add scripts\balance_control.py tests\test_balance_control.py
git commit -m "feat: add body balance controller"
```

---

### Task 2: Balance analysis script

**Files:**
- Create: `scripts/analyze_balance.py`
- Modify: `tests/test_balance_control.py`

- [ ] **Step 1: Add failing analysis tests**

Append to `tests/test_balance_control.py`:

```python
from scripts.analyze_balance import run_balance_simulation, write_balance_results


def test_balance_simulation_runs_finite(model):
    result = run_balance_simulation(model, duration=0.25)
    assert result.warning_count == 0
    assert result.finite
    assert result.steps > 0
    assert result.peak_abs_wheel_torque <= 10.0 + 1e-9
    assert np.isfinite(result.final_pitch)


def test_write_balance_results_outputs_planned_files(model, tmp_path):
    result = run_balance_simulation(model, duration=0.05)
    write_balance_results(result, tmp_path)
    assert (tmp_path / "balance_summary.csv").is_file()
    assert (tmp_path / "balance_timeseries.csv").is_file()
    assert (tmp_path / "balance_report.md").is_file()
    report = (tmp_path / "balance_report.md").read_text(encoding="utf-8")
    assert "Balance Control Analysis" in report
    assert "not walking" in report
```

- [ ] **Step 2: Run tests and verify import failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py::test_balance_simulation_runs_finite -v
```

Expected: fail with `ModuleNotFoundError: No module named 'scripts.analyze_balance'`.

- [ ] **Step 3: Implement `scripts/analyze_balance.py`**

Create `scripts/analyze_balance.py`:

```python
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
```

- [ ] **Step 4: Run analysis tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py::test_balance_simulation_runs_finite tests\test_balance_control.py::test_write_balance_results_outputs_planned_files -v
```

Expected: both tests pass. If warning count is nonzero, inspect and fix the cause rather than relaxing the assertion.

- [ ] **Step 5: Commit**

Run:

```powershell
git add scripts\analyze_balance.py tests\test_balance_control.py
git commit -m "feat: add balance analysis"
```

---

### Task 3: Controlled MuJoCo viewer

**Files:**
- Create: `scripts/run_balance_viewer.py`
- Modify: `tests/test_balance_control.py`

- [ ] **Step 1: Add a smoke test for viewer script import and CLI help**

Append to `tests/test_balance_control.py`:

```python
import subprocess
import sys


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
```

- [ ] **Step 2: Run test and verify script missing failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py::test_run_balance_viewer_help_succeeds -v
```

Expected: fail because `scripts/run_balance_viewer.py` does not exist.

- [ ] **Step 3: Implement `scripts/run_balance_viewer.py`**

Create `scripts/run_balance_viewer.py`:

```python
"""Run the MuJoCo viewer with the Python body balance controller."""

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
from scripts.balance_control import BalanceConfig, apply_balance_control
from scripts.convert_urdf_to_mjcf import convert_urdf
from scripts.pd_control import build_joint_map, set_base_weld_active


def run_viewer(model_path: Path, duration: float | None = None) -> None:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    set_base_weld_active(model, data, False)
    joint_map = build_joint_map(model)
    config = BalanceConfig()
    start = time.time()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            apply_balance_control(model, data, joint_map, config)
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
    args = parser.parse_args(argv)
    model_path = args.model if args.no_regenerate else convert_urdf(args.source, args.model)
    run_viewer(model_path, duration=args.duration)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run viewer help test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py::test_run_balance_viewer_help_succeeds -v
```

Expected: test passes. Do not open the GUI during automated tests.

- [ ] **Step 5: Commit**

Run:

```powershell
git add scripts\run_balance_viewer.py tests\test_balance_control.py
git commit -m "feat: add balance viewer runner"
```

---

### Task 4: README and balance artifacts

**Files:**
- Modify: `README.md`
- Generate: `analysis/balance_results/balance_summary.csv`
- Generate: `analysis/balance_results/balance_timeseries.csv`
- Generate: `analysis/balance_results/balance_report.md`

- [ ] **Step 1: Run balance analysis**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_balance.py --duration 2.0
```

Expected: exits 0 and prints `D:\Workspace\Mujoco\analysis\balance_results`.

- [ ] **Step 2: Update README**

Add this section after the existing dynamics-analysis section in `README.md`:

```markdown
## 机身平衡控制

第一版机身平衡控制使用腿部 PD 保持站立姿态，并用左右轮同向力矩调节机身 pitch。它是原地平衡原型，不是行走控制器。

运行平衡分析：

```powershell
.\.venv\Scripts\python.exe scripts\analyze_balance.py --duration 2.0
```

查看受控仿真：

```powershell
.\.venv\Scripts\python.exe scripts\run_balance_viewer.py
```

输出目录：

```text
analysis\balance_results\
```
```

- [ ] **Step 3: Verify README and artifacts**

Run:

```powershell
Test-Path analysis\balance_results\balance_summary.csv
Test-Path analysis\balance_results\balance_timeseries.csv
Test-Path analysis\balance_results\balance_report.md
Select-String -Path README.md -Pattern 'analyze_balance.py','run_balance_viewer.py','机身平衡控制'
```

Expected: all three `Test-Path` outputs are `True`, and all README patterns are found.

- [ ] **Step 4: Commit**

Run:

```powershell
git add README.md analysis\balance_results
git commit -m "docs: add balance control results"
```

---

### Task 5: Final verification

**Files:**
- No new files expected unless verification regenerates committed balance artifacts.

- [ ] **Step 1: Run full test suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

Expected: all tests pass.

- [ ] **Step 2: Run balance CLI smoke verification**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_balance.py --duration 0.5 --output-dir analysis\balance_results_verify
```

Expected: exits 0 and writes `balance_summary.csv`, `balance_timeseries.csv`, and `balance_report.md`. Remove `analysis\balance_results_verify` after inspection.

- [ ] **Step 3: Inspect balance summary**

Run:

```powershell
@'
import csv
from pathlib import Path
summary = list(csv.DictReader(Path("analysis/balance_results/balance_summary.csv").open(encoding="utf-8")))[0]
print("warning_count=", summary["warning_count"])
print("finite=", summary["finite"])
print("peak_abs_wheel_torque=", summary["peak_abs_wheel_torque"])
'@ | .\.venv\Scripts\python.exe -
```

Expected:

- `warning_count= 0`
- `finite= True`
- `peak_abs_wheel_torque` is at most `10.0`

- [ ] **Step 4: Check git status**

Run:

```powershell
git status --short
```

Expected: no output.

- [ ] **Step 5: Commit final regenerated artifacts if needed**

If Task 5 changed tracked artifacts, run:

```powershell
git add analysis\balance_results README.md
git commit -m "test: verify balance control"
```

If no files changed, do not create an empty commit.

---

## Self-Review

- Spec coverage:
  - Balance controller import/tests: Task 1.
  - Wheel torque pitch control and leg PD: Task 1.
  - Free-base short simulation and outputs: Task 2.
  - Controlled viewer: Task 3.
  - README update and artifacts: Task 4.
  - Final verification: Task 5.
- Placeholder scan:
  - No unfinished placeholder markers.
  - Each code-producing task includes concrete code and commands.
- Type consistency:
  - `BalanceConfig`, `BalanceState`, `BalanceSimulationResult`, `run_balance_simulation`, and `write_balance_results` are named consistently across tests and implementation.
