# Robust Standing Control v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build robust standing control v1 by adding configurable standing posture, objective metrics, deterministic parameter tuning, and default standing script integration for the MuJoCo wheeled biped.

**Architecture:** Keep the existing IMU-based pitch controller and leg PD controller. Add standing posture helpers in `scripts/balance_control.py`, shared standing objective metrics in `scripts/analyze_balance.py`, and a new deterministic sweep script `scripts/tune_standing_balance.py`. The implementation must report whether the v1 objective is met; it must not claim robust standing if tuning fails under current actuator limits.

**Tech Stack:** Python 3.10, MuJoCo Python API, NumPy, pytest, CSV/JSON/Markdown artifacts, PowerShell commands using `.venv\Scripts\python.exe`.

---

## Current Workspace Warning

At plan creation time, the worktree contains unrelated pending edits:

```text
 D 8dof_URDF/meshes/base_link_mujoco.STL
 M scripts/analyze_balance.py
 M scripts/analyze_dynamics.py
 M scripts/balance_control.py
 M scripts/convert_urdf_to_mjcf.py
 M scripts/pd_control.py
 M scripts/run_balance_viewer.py
```

Those script modifications are Chinese comment additions from prior work. Do not revert them. Do not include the deleted STL in robust-standing commits unless the user explicitly asks. Before each commit, inspect `git diff --cached --stat` and ensure only files for the current task are staged.

## File Structure

- Modify `scripts/balance_control.py`
  - Add `standing_leg_targets()`.
  - Add `default_standing_config()`.
  - Keep `BalanceConfig()` generic.
- Modify `scripts/analyze_balance.py`
  - Add standing objective constants and scoring helpers.
  - Extend `BalanceSimulationResult`.
  - Add optional `leg_targets` support to `run_balance_simulation()`.
  - Write new metrics to CSV and Markdown.
- Create `scripts/tune_standing_balance.py`
  - Deterministic grid search over pitch gains, x gains, and hip/knee fixed posture.
  - Write CSV, JSON, and Markdown tuning artifacts.
- Modify `scripts/run_balance_viewer.py`
  - Use `default_standing_config()` and `standing_leg_targets()` for controlled viewer.
- Modify `README.md`
  - Document robust-standing tuning and updated metrics.
  - Update stale test count if still present.
- Modify `tests/test_balance_control.py`
  - Add tests for standing posture helpers, new metrics, objective scoring, and tuning outputs.
- Generate/update `analysis/balance_results/*`
  - Regenerate with `scripts/analyze_balance.py --duration 2.0`.
- Generate/update `analysis/standing_tuning/*`
  - Generate with `scripts/tune_standing_balance.py`.

---

### Task 1: Add standing posture helpers

**Files:**
- Modify: `scripts/balance_control.py`
- Modify: `tests/test_balance_control.py`

- [ ] **Step 1: Add failing tests for symmetric standing targets and default standing config**

Append these imports to the existing `from scripts.balance_control import (...)` block in `tests/test_balance_control.py`:

```python
    default_standing_config,
    standing_leg_targets,
```

Append these tests near the existing leg-control tests:

```python
def test_standing_leg_targets_are_symmetric_and_leg_only():
    targets = standing_leg_targets(hip_pitch=-0.15, knee=0.35)

    assert targets == {
        "left_hip_pitch_joint": -0.15,
        "right_hip_pitch_joint": -0.15,
        "left_knee_joint": 0.35,
        "right_knee_joint": 0.35,
    }
    assert not any("wheel" in joint_name for joint_name in targets)
    assert not any("roll" in joint_name for joint_name in targets)


def test_default_standing_config_is_explicit_and_does_not_change_generic_config():
    generic = BalanceConfig()
    standing = default_standing_config()

    assert standing != generic
    assert standing.pitch_target == pytest.approx(0.0)
    assert standing.pitch_rate_target == pytest.approx(0.0)
    assert standing.kp_pitch > 0.0
    assert standing.kd_pitch >= 0.0
    assert standing.leg_kp > 0.0
    assert standing.leg_kd >= 0.0
```

- [ ] **Step 2: Run tests and verify helper imports fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py::test_standing_leg_targets_are_symmetric_and_leg_only tests\test_balance_control.py::test_default_standing_config_is_explicit_and_does_not_change_generic_config -v
```

Expected: fail with import errors for `standing_leg_targets` and `default_standing_config`.

- [ ] **Step 3: Implement standing helpers**

In `scripts/balance_control.py`, add constants after the IMU sensor constants:

```python
DEFAULT_STANDING_HIP_PITCH = -0.15
DEFAULT_STANDING_KNEE = 0.35
```

Replace the existing `default_balance_config()` section with this expanded block while keeping `default_balance_config()` behavior unchanged:

```python
def standing_leg_targets(
    hip_pitch: float = DEFAULT_STANDING_HIP_PITCH,
    knee: float = DEFAULT_STANDING_KNEE,
) -> dict[str, float]:
    """Return symmetric fixed leg targets for robust-standing attempts."""
    return {
        "left_hip_pitch_joint": float(hip_pitch),
        "right_hip_pitch_joint": float(hip_pitch),
        "left_knee_joint": float(knee),
        "right_knee_joint": float(knee),
    }


def default_balance_config() -> BalanceConfig:
    return BalanceConfig()


def default_standing_config() -> BalanceConfig:
    """Return the current best explicit robust-standing controller config."""
    return BalanceConfig(
        pitch_target=0.0,
        pitch_rate_target=0.0,
        x_target=None,
        x_velocity_target=0.0,
        kp_pitch=35.0,
        kd_pitch=4.0,
        kx=0.0,
        kv=1.0,
        leg_kp=20.0,
        leg_kd=1.0,
    )
```

This initial `default_standing_config()` is a named entry point. Later tasks may update the gains after tuning.

- [ ] **Step 4: Run the new helper tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py::test_standing_leg_targets_are_symmetric_and_leg_only tests\test_balance_control.py::test_default_standing_config_is_explicit_and_does_not_change_generic_config -v
```

Expected: both tests pass.

- [ ] **Step 5: Add a control-path test for standing leg targets**

Append this test after `test_leg_joints_receive_posture_pd_torques`:

```python
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
```

- [ ] **Step 6: Run the control-path test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py::test_compute_balance_control_uses_symmetric_standing_leg_targets -v
```

Expected: pass.

- [ ] **Step 7: Run full balance-control tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py -v
```

Expected: all tests in `tests/test_balance_control.py` pass.

- [ ] **Step 8: Commit Task 1**

Stage only the task files:

```powershell
git add scripts\balance_control.py tests\test_balance_control.py
git diff --cached --stat
git commit -m "feat: add standing posture helpers"
```

Expected staged files:

```text
scripts/balance_control.py
tests/test_balance_control.py
```

---

### Task 2: Add robust-standing metrics and objective scoring

**Files:**
- Modify: `scripts/analyze_balance.py`
- Modify: `tests/test_balance_control.py`

- [ ] **Step 1: Add failing tests for objective value helpers**

Add these names to the existing `from scripts.analyze_balance import (...)` block in `tests/test_balance_control.py`:

```python
    meets_standing_objective_values,
    standing_score_values,
```

Append these tests near the balance simulation tests:

```python
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
```

- [ ] **Step 2: Run tests and verify helper imports fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py::test_standing_objective_values_accept_good_result tests\test_balance_control.py::test_standing_score_values_penalizes_warning_and_nonfinite_results -v
```

Expected: fail with missing imports for `meets_standing_objective_values` and `standing_score_values`.

- [ ] **Step 3: Implement objective constants and helper functions**

In `scripts/analyze_balance.py`, add this import:

```python
from typing import Mapping, Sequence
```

Replace the current `from typing import Sequence` line with the line above.

After `TORQUE_LIMIT_EPSILON = 1e-9`, add:

```python
STANDING_FINAL_ABS_PITCH_LIMIT = 0.25
STANDING_PEAK_ABS_PITCH_LIMIT = 0.5
STANDING_PEAK_ABS_X_DRIFT_LIMIT = 0.3
STANDING_FAILURE_SCORE = 1_000_000.0
```

After `_finite()`, add:

```python
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
    if int(warning_count) != 0 or not bool(finite) or not np.isfinite(values).all():
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
```

- [ ] **Step 4: Run objective helper tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py::test_standing_objective_values_accept_good_result tests\test_balance_control.py::test_standing_objective_values_reject_failures tests\test_balance_control.py::test_standing_score_values_penalizes_warning_and_nonfinite_results -v
```

Expected: all objective helper tests pass.

- [ ] **Step 5: Add failing tests for new simulation result fields**

Modify `test_balance_simulation_summary_matches_final_sample` to assert the new fields. Replace the test body with:

```python
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
```

Modify `test_write_balance_results_outputs_planned_files` so the expected timeseries fields are:

```python
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
```

Modify `test_write_balance_results_notes_wheel_torque_saturation` constructor to include the new dataclass fields:

```python
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
```

- [ ] **Step 6: Run tests and verify field failures**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py::test_balance_simulation_summary_matches_final_sample tests\test_balance_control.py::test_write_balance_results_outputs_planned_files tests\test_balance_control.py::test_write_balance_results_notes_wheel_torque_saturation -v
```

Expected: fail because `BalanceSimulationResult` and `timeseries` do not yet include the new fields.

- [ ] **Step 7: Extend `BalanceSimulationResult` and `run_balance_simulation()`**

In `scripts/analyze_balance.py`, replace the `BalanceSimulationResult` dataclass fields with:

```python
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
```

Change the `run_balance_simulation()` signature to:

```python
def run_balance_simulation(
    model: mujoco.MjModel,
    duration: float = 2.0,
    config: BalanceConfig | None = None,
    leg_targets: Mapping[str, float] | None = None,
) -> BalanceSimulationResult:
```

Inside `run_balance_simulation()`, after setting `config`, add:

```python
    initial_x = float(data.qpos[0])
```

Replace the call to `apply_balance_control` with:

```python
        ctrl, applied_state = apply_balance_control(model, data, joint_map, config, leg_targets)
```

Replace the timeseries append dictionary with:

```python
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
```

Replace the return block with:

```python
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
```

- [ ] **Step 8: Update result writers**

In `write_balance_results()`, replace `summary_fields` with:

```python
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
```

Replace `timeseries_fields` with:

```python
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
```

In the Markdown `lines`, add these entries after `Finite state`:

```python
        f"- Standing objective met: {result.meets_standing_objective}",
        f"- Standing score: {result.standing_score:.6g}",
```

Add these entries near the pitch/x metrics:

```python
        f"- Final |pitch|: {result.final_abs_pitch:.6g} rad",
        f"- Peak |x drift|: {result.peak_abs_x_drift:.6g} m",
        f"- Final x drift: {result.x_drift:.6g} m",
        f"- Wheel torque saturation fraction: {result.wheel_torque_saturation_fraction:.3f}",
```

Before the existing torque-limit warning block, add:

```python
    if not result.meets_standing_objective:
        lines.extend(
            [
                "",
                "This run does not meet the robust-standing v1 objective.",
            ]
        )
```

- [ ] **Step 9: Run focused metric and writer tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py::test_balance_simulation_summary_matches_final_sample tests\test_balance_control.py::test_write_balance_results_outputs_planned_files tests\test_balance_control.py::test_write_balance_results_notes_wheel_torque_saturation -v
```

Expected: pass.

- [ ] **Step 10: Run full balance-control tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py -v
```

Expected: all tests in `tests/test_balance_control.py` pass.

- [ ] **Step 11: Commit Task 2**

Stage only the task files:

```powershell
git add scripts\analyze_balance.py tests\test_balance_control.py
git diff --cached --stat
git commit -m "feat: add standing objective metrics"
```

Expected staged files:

```text
scripts/analyze_balance.py
tests/test_balance_control.py
```

---

### Task 3: Add deterministic standing tuning script

**Files:**
- Create: `scripts/tune_standing_balance.py`
- Modify: `tests/test_balance_control.py`

- [ ] **Step 1: Add failing tests for tuning outputs and smoke run**

Add this import section near the top of `tests/test_balance_control.py`:

```python
import json
```

Append these tests near the other analysis output tests:

```python
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
```

- [ ] **Step 2: Run tests and verify module import fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py::test_write_standing_tuning_results_outputs_planned_files tests\test_balance_control.py::test_run_standing_tuning_smoke_returns_finite_candidate -v
```

Expected: fail with `ModuleNotFoundError: No module named 'scripts.tune_standing_balance'`.

- [ ] **Step 3: Create `scripts/tune_standing_balance.py`**

Create the file with this content:

```python
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
            leg_kp=20.0,
            leg_kd=1.0,
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
        )
        for kp_pitch, kd_pitch, kx, kv, hip_pitch, knee in product(
            [20.0, 35.0, 50.0],
            [2.0, 4.0, 6.0],
            [0.0, 1.0],
            [0.5, 1.0],
            [-0.2, 0.0, 0.2],
            [0.0, 0.35],
        )
    ]


def _row_from_result(
    rank: int,
    candidate: StandingCandidate,
    result: BalanceSimulationResult,
) -> dict[str, float | int | bool]:
    row: dict[str, float | int | bool] = {
        "rank": int(rank),
        **asdict(candidate),
        "warning_count": int(result.warning_count),
        "finite": bool(result.finite),
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
) -> tuple[list[dict[str, float | int | bool]], StandingCandidate | None]:
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
    rows: list[dict[str, float | int | bool]],
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
        "warning_count",
        "finite",
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
```

- [ ] **Step 4: Run tuning tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py::test_write_standing_tuning_results_outputs_planned_files tests\test_balance_control.py::test_run_standing_tuning_smoke_returns_finite_candidate -v
```

Expected: both tuning tests pass.

- [ ] **Step 5: Run tuning CLI smoke with a short duration**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\tune_standing_balance.py --duration 0.02 --output-dir analysis\standing_tuning_smoke
Get-ChildItem analysis\standing_tuning_smoke
Remove-Item analysis\standing_tuning_smoke -Recurse -Force
```

Expected: CLI exits 0, writes `standing_tuning_results.csv`, `standing_best_config.json`, and `standing_tuning_report.md`, then the temporary directory is removed.

- [ ] **Step 6: Run full balance-control tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py -v
```

Expected: all tests in `tests/test_balance_control.py` pass.

- [ ] **Step 7: Commit Task 3**

Stage only the task files:

```powershell
git add scripts\tune_standing_balance.py tests\test_balance_control.py
git diff --cached --stat
git commit -m "feat: add standing balance tuning"
```

Expected staged files:

```text
scripts/tune_standing_balance.py
tests/test_balance_control.py
```

---

### Task 4: Integrate standing defaults into analysis and viewer

**Files:**
- Modify: `scripts/analyze_balance.py`
- Modify: `scripts/run_balance_viewer.py`
- Modify: `tests/test_balance_control.py`

- [ ] **Step 1: Add failing test that analysis uses standing defaults**

Add `standing_leg_targets` to the `from scripts.balance_control import (...)` block if not already imported.

Append this test near `test_balance_simulation_defaults_none_x_target_to_initial_base_x`:

```python
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
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py::test_balance_simulation_uses_default_standing_targets_when_not_provided -v
```

Expected: fail because `run_balance_simulation()` still passes `leg_targets=None`.

- [ ] **Step 3: Update `analyze_balance.py` imports and defaults**

In `scripts/analyze_balance.py`, add these imports from `scripts.balance_control`:

```python
    default_standing_config,
    standing_leg_targets,
```

In `run_balance_simulation()`, replace the config default block:

```python
    if config is None:
        # 默认把当前 x 位置作为目标，避免控制器一启动就试图回到全局 0。
        config = BalanceConfig(x_target=float(data.qpos[0]))
    elif config.x_target is None:
        config = replace(config, x_target=float(data.qpos[0]))
```

with:

```python
    if config is None:
        # 默认使用当前最好的 standing 入口；仍把当前 x 位置作为目标。
        config = default_standing_config()
    if config.x_target is None:
        config = replace(config, x_target=float(data.qpos[0]))
    if leg_targets is None:
        leg_targets = standing_leg_targets()
```

- [ ] **Step 4: Run default standing-target test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py::test_balance_simulation_uses_default_standing_targets_when_not_provided -v
```

Expected: pass.

- [ ] **Step 5: Update viewer to use standing defaults**

In `scripts/run_balance_viewer.py`, change the import:

```python
from scripts.balance_control import BalanceConfig, apply_balance_control
```

to:

```python
from scripts.balance_control import apply_balance_control, default_standing_config, standing_leg_targets
```

Replace:

```python
    config = BalanceConfig(x_target=float(data.qpos[0]))
```

with:

```python
    config = default_standing_config()
    if config.x_target is None:
        config = type(config)(
            pitch_target=config.pitch_target,
            pitch_rate_target=config.pitch_rate_target,
            x_target=float(data.qpos[0]),
            x_velocity_target=config.x_velocity_target,
            kp_pitch=config.kp_pitch,
            kd_pitch=config.kd_pitch,
            kx=config.kx,
            kv=config.kv,
            leg_kp=config.leg_kp,
            leg_kd=config.leg_kd,
        )
    leg_targets = standing_leg_targets()
```

Replace the viewer loop control call:

```python
            apply_balance_control(model, data, joint_map, config)
```

with:

```python
            apply_balance_control(model, data, joint_map, config, leg_targets)
```

- [ ] **Step 6: Run viewer help test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py::test_run_balance_viewer_help_succeeds -v
```

Expected: pass.

- [ ] **Step 7: Run full balance-control tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py -v
```

Expected: all tests in `tests/test_balance_control.py` pass.

- [ ] **Step 8: Commit Task 4**

Stage only the task files:

```powershell
git add scripts\analyze_balance.py scripts\run_balance_viewer.py tests\test_balance_control.py
git diff --cached --stat
git commit -m "feat: use standing defaults in balance analysis"
```

Expected staged files:

```text
scripts/analyze_balance.py
scripts/run_balance_viewer.py
tests/test_balance_control.py
```

---

### Task 5: Run tuning, update artifacts, and document status

**Files:**
- Modify: `README.md`
- Generate/modify: `analysis/balance_results/balance_summary.csv`
- Generate/modify: `analysis/balance_results/balance_timeseries.csv`
- Generate/modify: `analysis/balance_results/balance_report.md`
- Generate: `analysis/standing_tuning/standing_tuning_results.csv`
- Generate: `analysis/standing_tuning/standing_best_config.json`
- Generate: `analysis/standing_tuning/standing_tuning_report.md`
- Possibly modify: `scripts/balance_control.py`

- [ ] **Step 1: Run full-duration standing tuning**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\tune_standing_balance.py --duration 2.0
```

Expected: exits 0 and writes:

```text
analysis\standing_tuning\standing_tuning_results.csv
analysis\standing_tuning\standing_best_config.json
analysis\standing_tuning\standing_tuning_report.md
```

- [ ] **Step 2: Inspect best configuration**

Run:

```powershell
Get-Content -Raw analysis\standing_tuning\standing_best_config.json
Get-Content -Raw analysis\standing_tuning\standing_tuning_report.md
```

Expected: JSON contains a `candidate` object and `result` object. Markdown states `Objective met: True` or `Objective met: False`.

- [ ] **Step 3: If objective is met, update `default_standing_config()` and standing posture constants**

If `standing_best_config.json` has `"meets_standing_objective": true`, update `scripts/balance_control.py`:

```python
DEFAULT_STANDING_HIP_PITCH = <best candidate hip_pitch>
DEFAULT_STANDING_KNEE = <best candidate knee>
```

and update `default_standing_config()` values:

```python
def default_standing_config() -> BalanceConfig:
    """Return the current best explicit robust-standing controller config."""
    return BalanceConfig(
        pitch_target=0.0,
        pitch_rate_target=0.0,
        x_target=None,
        x_velocity_target=0.0,
        kp_pitch=<best candidate kp_pitch>,
        kd_pitch=<best candidate kd_pitch>,
        kx=<best candidate kx>,
        kv=<best candidate kv>,
        leg_kp=20.0,
        leg_kd=1.0,
    )
```

Use numeric literals from the JSON. Do not round to fewer than three significant digits.

If the objective is not met, do not update defaults to imply success. Keep the explicit standing entry point and preserve the tuning artifacts showing the best failed candidate.

- [ ] **Step 4: Run standing analysis with current defaults**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_balance.py --duration 2.0
```

Expected: exits 0 and updates `analysis\balance_results\*` with new standing metrics.

- [ ] **Step 5: Add default-standing integration test only if objective is met**

If the full-duration analysis report now satisfies:

```text
warning_count == 0
finite == True
final_abs_pitch < 0.25
peak_abs_pitch < 0.5
peak_abs_x_drift < 0.3
```

append this test to `tests/test_balance_control.py`:

```python
def test_default_standing_config_meets_two_second_objective(model):
    result = run_balance_simulation(model, duration=2.0)

    assert result.warning_count == 0
    assert result.finite
    assert result.final_abs_pitch < 0.25
    assert result.peak_abs_pitch < 0.5
    assert result.peak_abs_x_drift < 0.3
    assert result.meets_standing_objective
```

If the objective is not met, do not add this test. Instead, rely on tuning/report artifacts and final handoff to state that robust standing v1 remains unmet under current sweep.

- [ ] **Step 6: Update README**

In `README.md`, update the current status and balance section to mention:

Add this Markdown section:

    ## 稳健站立 v1

    当前控制路线采用 IMU pitch/pitch-rate + 轮子力矩控制，并允许 hip/knee 使用一组固定站姿目标。参数扫描脚本：

    ```powershell
    .\.venv\Scripts\python.exe scripts\tune_standing_balance.py --duration 2.0
    ```

    输出目录：

    ```text
    analysis\standing_tuning\
    ```

    稳健站立 v1 的目标是 2 秒 free-base 仿真中 `final_abs_pitch < 0.25 rad`、`peak_abs_pitch < 0.5 rad`、`peak_abs_x_drift < 0.3 m`，同时无 MuJoCo warning 且状态有限。

Also update the stale test count if README still says:

```text
19 passed
```

Change it to the current full-test count after running the full suite in Task 6.

- [ ] **Step 7: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py -v
```

Expected: all balance-control tests pass.

- [ ] **Step 8: Commit Task 5**

If `scripts/balance_control.py` was updated with a successful best candidate, stage it. If not, omit it.

Run:

```powershell
git add README.md analysis\balance_results analysis\standing_tuning tests\test_balance_control.py
git add scripts\balance_control.py
git diff --cached --stat
git commit -m "docs: record robust standing tuning results"
```

If `scripts/balance_control.py` has no changes, `git add scripts\balance_control.py` is harmless. Expected staged files include README, analysis artifacts, tuning artifacts, and possibly `scripts/balance_control.py` plus the optional integration test.

---

### Task 6: Final verification

**Files:**
- No new files expected unless verification regenerates tracked artifacts.

- [ ] **Step 1: Run full test suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

Expected: all tests pass.

- [ ] **Step 2: Run direct standing diagnostics**

Run:

```powershell
@'
from pathlib import Path
import mujoco
from scripts.analyze_balance import run_balance_simulation
from scripts.convert_urdf_to_mjcf import convert_urdf

model_path = convert_urdf(Path("8dof_URDF/urdf/robot.urdf"), Path("8dof_URDF/mjcf/robot.xml"))
model = mujoco.MjModel.from_xml_path(str(model_path))
result = run_balance_simulation(model, duration=2.0)
print("warning_count=", result.warning_count)
print("finite=", result.finite)
print("final_abs_pitch=", result.final_abs_pitch)
print("peak_abs_pitch=", result.peak_abs_pitch)
print("peak_abs_x_drift=", result.peak_abs_x_drift)
print("wheel_torque_saturation_fraction=", result.wheel_torque_saturation_fraction)
print("meets_standing_objective=", result.meets_standing_objective)
print("standing_score=", result.standing_score)
'@ | .\.venv\Scripts\python.exe -
```

Expected: command exits 0 and prints finite numeric diagnostics. If `meets_standing_objective=False`, final handoff must state that v1 diagnostics are implemented but the current tuned/default controller still misses the target.

- [ ] **Step 3: Run CLI smoke commands**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_balance.py --duration 0.1 --output-dir analysis\standing_verify_balance
.\.venv\Scripts\python.exe scripts\tune_standing_balance.py --duration 0.02 --output-dir analysis\standing_verify_tuning
Test-Path analysis\standing_verify_balance\balance_report.md
Test-Path analysis\standing_verify_tuning\standing_tuning_report.md
Remove-Item analysis\standing_verify_balance -Recurse -Force
Remove-Item analysis\standing_verify_tuning -Recurse -Force
```

Expected: both `Test-Path` calls print `True`, and temporary directories are removed.

- [ ] **Step 4: Inspect git status**

Run:

```powershell
git status --short --untracked-files=all
```

Expected: robust-standing task files should be committed or intentionally staged for a final verification commit. The pre-existing unrelated deleted STL and Chinese comment changes may still appear if the user has not asked to handle them.

- [ ] **Step 5: Commit verification artifacts only if tracked outputs changed**

If `analysis/balance_results/*`, `analysis/standing_tuning/*`, `README.md`, or `scripts/balance_control.py` changed during final verification and those changes belong to robust-standing work, run:

```powershell
git add README.md analysis\balance_results analysis\standing_tuning scripts\balance_control.py tests\test_balance_control.py
git diff --cached --stat
git commit -m "test: verify robust standing diagnostics"
```

If no robust-standing tracked files changed, do not create an empty commit.

---

## Self-Review

Spec coverage:

- Configurable fixed hip/knee standing posture: Task 1.
- Explicit default standing config: Task 1 and Task 5.
- New x drift, saturation, objective, and score metrics: Task 2.
- Deterministic tuning script and artifacts: Task 3 and Task 5.
- Analysis/viewer integration: Task 4.
- README and generated artifacts: Task 5.
- Final full tests and diagnostics: Task 6.

Scope control:

- No LQR, MPC, walking, actuator limit changes, or model geometry changes are included.
- If tuning does not meet the target, the plan records diagnostics honestly rather than forcing a success claim.

Type consistency:

- `standing_leg_targets()` returns `dict[str, float]`.
- `run_balance_simulation()` accepts `leg_targets: Mapping[str, float] | None`.
- `StandingCandidate.config()` returns `BalanceConfig`.
- Tuning rows are dictionaries with stable CSV/JSON keys.
