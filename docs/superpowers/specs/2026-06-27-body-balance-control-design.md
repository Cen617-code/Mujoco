# Body Balance Control Design

Date: 2026-06-27  
Project: `D:\Workspace\Mujoco`

## Goal

Add a first-pass body balance controller for the 8-DOF wheeled biped MuJoCo model. The controller should keep the robot near upright from the existing standing pose by using wheel torque to regulate body pitch while the leg joints hold a nominal posture with PD control.

This is the next step after the current motorized model and finite-dynamics validation:

- MJCF has a free base.
- 8 torque motors exist.
- Hip pitch limits are `[-1.22, 0.87]`.
- Python PD joint control exists.
- Fixed-base step response and free-base finite dynamics analysis exist.

## Non-goals

This phase will not implement walking, trajectory planning, MPC, whole-body QP control, terrain handling, external push recovery, velocity command tracking, or a formally linearized LQR controller.

The target is a practical first closed-loop balance prototype that is easy to inspect, test, and tune.

## Recommended Approach

Use a simple wheel-based body pitch stabilizer plus existing joint PD posture control.

The controller has two layers:

1. Leg posture layer
   - Use existing PD joint control for six leg joints:
     - `left_roll_joint`
     - `left_hip_pitch_joint`
     - `left_knee_joint`
     - `right_roll_joint`
     - `right_hip_pitch_joint`
     - `right_knee_joint`
   - Keep these joints near a nominal standing target, initially the current home pose.

2. Wheel balance layer
   - Use left/right wheel motors to regulate base pitch and optionally horizontal base motion.
   - Apply equal wheel torque to both wheels for pitch stabilization.
   - Keep wheel torque clipped by the existing wheel actuator limits of ±10 N·m.

## State Signals

The controller should read state directly from MuJoCo:

- Base quaternion from `data.qpos[3:7]`.
- Base linear velocity from free-joint qvel.
- Base angular velocity from free-joint qvel.
- Joint positions and velocities through existing joint map helpers.

The first version will compute body pitch from the base quaternion and pitch rate from base angular velocity. The implementation must document the chosen axis convention and include tests for small pitch rotations so sign mistakes are caught early.

## Control Law

Initial wheel balance torque:

```text
tau_balance =
    Kp_pitch * (pitch_target - pitch)
  + Kd_pitch * (pitch_rate_target - pitch_rate)
  + Kx       * (x_target - x)
  + Kv       * (x_velocity_target - x_velocity)
```

Default targets:

```text
pitch_target = 0
pitch_rate_target = 0
x_target = initial_base_x
x_velocity_target = 0
```

The same `tau_balance` is sent to both wheel motors, then clipped by each wheel actuator `ctrlrange`.

Leg joint torques come from existing PD control. Wheel entries from the existing PD controller will be replaced by the balance torque in this mode.

## Initial Gains

Use conservative tunable defaults:

- `Kp_pitch`: moderate positive value
- `Kd_pitch`: damping value
- `Kx`: small value, initially optional
- `Kv`: damping value, initially optional

The exact default values will be chosen during implementation by short simulation tests. They must not rely on exceeding motor torque limits.

## Scripts

Expected new scripts:

- `scripts/balance_control.py`
  - body orientation helpers
  - balance-controller dataclass/config
  - combined leg-PD + wheel-balance torque computation

- `scripts/analyze_balance.py`
  - runs free-base balance simulations
  - records pitch, base height, wheel torque, warnings, and finite-state checks
  - writes CSV and Markdown summary to `analysis/balance_results/`

- `scripts/run_balance_viewer.py`
  - launches a MuJoCo viewer with the Python balance controller running in the loop
  - allows the user to visually inspect closed-loop behavior

## Tests

Tests should cover:

1. Quaternion-to-pitch extraction for small positive and negative rotations.
2. Balance torque direction for positive/negative pitch error.
3. Wheel torque saturation at ±10 N·m.
4. Leg joints continue to receive PD posture torques.
5. Free-base balance simulation runs for a short duration with finite qpos/qvel/control and zero MuJoCo warnings.
6. Analysis output files are generated with planned names and useful metrics.

## Analysis Outputs

Write balance results to:

```text
analysis/balance_results/
```

Planned files:

- `balance_summary.csv`
- `balance_timeseries.csv`
- `balance_report.md`

Minimum metrics:

- duration
- warning count
- peak absolute pitch
- final pitch
- peak absolute pitch rate
- peak absolute wheel torque
- final base height
- whether state remained finite

## Viewer Behavior

The existing command:

```powershell
.\.venv\Scripts\python.exe -m mujoco.viewer --mjcf=8dof_URDF\mjcf\robot.xml
```

loads the model only and does not run the Python controller. The new viewer script should be the recommended command for seeing balance control:

```powershell
.\.venv\Scripts\python.exe scripts\run_balance_viewer.py
```

## Acceptance Criteria

This phase is complete when:

1. The balance controller can be imported and tested.
2. A short free-base controlled simulation runs without NaN/Inf and with MuJoCo warning count 0.
3. Wheel torques remain within ±10 N·m.
4. Analysis artifacts are generated under `analysis/balance_results/`.
5. A viewer script exists for visually inspecting the controlled model.
6. The README is updated with the new balance-analysis and controlled-viewer commands.

## Known Risks

- The current passive standing pose may not be a true dynamic equilibrium.
- A simple pitch controller may reduce falling but may not fully stabilize all body modes.
- Contact friction, mesh collision geometry, and wheel inertia may limit achievable balance.
- Sign convention for pitch and wheel torque is easy to get wrong; tests and short simulations must verify it.

If the simple controller cannot balance from the nominal pose without warnings or saturation, the implementation should report that clearly and leave the controller tunable rather than hiding the failure.
