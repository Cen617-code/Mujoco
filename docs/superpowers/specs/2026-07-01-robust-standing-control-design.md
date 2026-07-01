# Robust Standing Control v1 Design

## Goal

Build a first robust-standing control iteration for the 8-DOF wheeled biped using the existing MuJoCo model, IMU feedback, torque motors, and Python controller stack.

The first target is not walking. It is a 2-second free-base standing run from the current nominal initial pose with finite dynamics, no MuJoCo warnings, reduced pitch, and bounded horizontal drift.

## Current Baseline

The current balance controller is a first-pass pitch prototype:

- Legs hold the nominal `qpos0` pose with PD torques.
- Wheels apply pitch-balancing torque from IMU/freejoint pitch feedback.
- Left and right wheel actuator signs are mirrored to account for mirrored wheel joint axes.
- The controller runs finite and warning-free, but the robot still falls far from upright.

Current 2-second balance analysis:

```text
warning_count = 0
finite = True
peak_abs_pitch ≈ 1.55 rad
final_pitch ≈ 1.34 rad
peak_abs_wheel_torque = 10 N·m
```

This means the model/control stack is numerically usable, but the controller does not yet satisfy robust standing.

## Acceptance Criteria

Robust standing v1 succeeds when a 2-second free-base simulation from `model.qpos0` satisfies:

```text
warning_count == 0
finite == True
final_abs_pitch < 0.25 rad
peak_abs_pitch < 0.5 rad
peak_abs_x_drift < 0.3 m
```

Wheel torque saturation is not a hard failure for v1, but it must be measured and reported as a diagnostic:

```text
wheel_torque_saturation_fraction
```

The controller may move the wheels and base in x, but it should keep drift under the 0.3 m target.

## Chosen Approach

Use approach A: enhance the existing PD-style balance controller with configurable standing posture, richer diagnostics, and parameter sweep.

Do not implement in v1:

- LQR.
- MPC.
- Walking or trajectory tracking.
- Dynamic leg feedback based on pitch.
- Contact-mode optimization.

Do implement in v1:

- A configurable fixed standing posture for hip/knee joints.
- Improved balance simulation metrics.
- A tuning script that scans pitch gains, x gains, and fixed hip/knee standing targets.
- A default standing configuration selected from the tuning result if it satisfies the acceptance criteria.

## Control Architecture

The control structure remains simple and explicit:

```text
base_imu_quat + base_imu_gyro
        ↓
pitch, pitch_rate
        ↓
wheel pitch/x feedback
        ↓
left/right wheel torque
```

The legs use fixed posture PD:

```text
standing leg target
        ↓
leg joint PD
        ↓
roll/hip/knee torque
```

The legs are not a dynamic balance controller in v1. They only maintain a configurable support pose.

## Standing Posture

Add a named standing posture helper rather than burying values inside ad-hoc test code.

The first configurable posture only changes symmetric hip/knee pitch targets:

```text
left_hip_pitch_joint
right_hip_pitch_joint
left_knee_joint
right_knee_joint
```

Roll joints remain at their home target unless explicitly overridden by the caller. Wheel joints are not posture targets.

The posture helper should produce a dictionary compatible with the existing `leg_targets` argument to `compute_balance_control()`.

Example interface:

```python
def standing_leg_targets(
    hip_pitch: float = 0.0,
    knee: float = 0.0,
) -> dict[str, float]:
    ...
```

The implementation plan may choose exact default values after running parameter sweep, but the helper must keep left/right targets symmetric.

## Standing Configuration

Keep `BalanceConfig()` as the generic controller configuration. Add a separate explicit standing entry point:

```python
def default_standing_config() -> BalanceConfig:
    ...
```

This avoids silently changing the meaning of existing tests or scripts that use plain `BalanceConfig()`.

`default_standing_config()` should return the tuned v1 values once tuning identifies an acceptable set. Until then, it can return the best diagnostic candidate and the analysis report must honestly state whether it meets the objective.

## Balance Metrics

Extend `BalanceSimulationResult` with:

```text
initial_x
final_x
x_drift
peak_abs_x_drift
final_abs_pitch
wheel_torque_saturation_fraction
meets_standing_objective
standing_score
```

Definitions:

- `initial_x`: base x at reset, before the first control step.
- `final_x`: base x at the final sample.
- `x_drift`: `final_x - initial_x`.
- `peak_abs_x_drift`: maximum `abs(x - initial_x)` over the simulation.
- `final_abs_pitch`: `abs(final_pitch)`.
- `wheel_torque_saturation_fraction`: fraction of samples where `abs(wheel_torque)` is within epsilon of the ±10 N·m wheel torque limit.
- `meets_standing_objective`: boolean computed from the v1 acceptance criteria.
- `standing_score`: scalar score used to rank tuning candidates.

Initial scoring formula:

```text
standing_score =
    4.0 * final_abs_pitch
  + 2.0 * peak_abs_pitch
  + 1.0 * peak_abs_x_drift
  + 0.5 * wheel_torque_saturation_fraction
```

If a simulation has warnings, non-finite state, or severe pitch divergence, it should receive a high penalty score.

## Tuning Script

Add:

```text
scripts/tune_standing_balance.py
```

The script should perform a small deterministic grid search over:

```text
kp_pitch
kd_pitch
kx
kv
hip_pitch target
knee target
```

Each candidate runs a 2-second free-base simulation using the same model and analysis pipeline as `analyze_balance.py`.

Output directory:

```text
analysis/standing_tuning/
```

Output files:

```text
standing_tuning_results.csv
standing_best_config.json
standing_tuning_report.md
```

The CSV should contain one row per candidate with all metrics needed to compare runs. The JSON should contain the best candidate's controller gains and standing posture. The Markdown report should state whether the best candidate satisfies the robust-standing v1 acceptance criteria.

## Script Integration

`analyze_balance.py` should be able to evaluate the tuned/default standing configuration without duplicating simulation logic.

Preferred direction:

- Keep `run_balance_simulation()` as the core simulation function.
- Add optional `leg_targets` support to analysis if needed.
- Use shared scoring/objective helpers so `analyze_balance.py`, tests, and `tune_standing_balance.py` agree on the same definition.

`run_balance_viewer.py` should use the explicit standing configuration once it exists, so opening the controlled viewer shows the best current robust-standing attempt.

## Testing Strategy

Add tests before implementation where possible.

Required test coverage:

1. `standing_leg_targets()` returns symmetric hip/knee targets and does not include wheel targets.
2. `compute_balance_control()` applies supplied standing leg targets to leg PD torques.
3. Balance simulation results include the new x-drift, saturation, objective, and score fields.
4. Objective evaluation returns `True` for a constructed good result and `False` for results that violate pitch, drift, warning, or finite-state requirements.
5. Tuning output writer creates the planned CSV, JSON, and Markdown files.
6. A short tuning smoke test runs a small grid and returns at least one finite candidate.
7. If a tuned/default standing configuration satisfies the 2-second acceptance criteria, add a 2-second integration test for it.

If the sweep cannot find a satisfying candidate under the current actuator limits, the implementation must not pretend success. It should still commit the diagnostic framework and report that robust standing v1 remains unmet, with the best candidate metrics visible.

## Risks

### Torque saturation

The wheel motors are limited to ±10 N·m. If the best candidates saturate most of the time, the current actuator limit may be insufficient for this posture/model.

### Standing posture sensitivity

The nominal `qpos0` may put the center of mass in an unfavorable position. Fixed hip/knee posture scanning is included specifically to test this.

### Heuristic controller limits

This v1 remains a heuristic PD/state-feedback style controller. It may improve standing but does not provide formal stability guarantees.

### Existing dirty worktree

At design time, the workspace includes unrelated pending edits:

- Chinese comments in main scripts.
- Deleted `8dof_URDF/meshes/base_link_mujoco.STL`.

Implementation should avoid accidentally including unrelated changes unless the user explicitly asks to clean or commit them.

## Non-Goals

- No walking gait.
- No LQR/MPC in this iteration.
- No model geometry changes.
- No actuator limit changes unless separately approved.
- No dynamic leg balance response in v1.

## Expected User Workflow After Implementation

Run tuning:

```powershell
.\.venv\Scripts\python.exe scripts\tune_standing_balance.py
```

Run standing analysis:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_balance.py --duration 2.0
```

View controlled simulation:

```powershell
.\.venv\Scripts\python.exe scripts\run_balance_viewer.py
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

