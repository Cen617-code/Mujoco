# Base IMU Sensor Design

Date: 2026-06-27  
Project: `D:\Workspace\Mujoco`

## Goal

Add an IMU-style sensor mounted above `base_link` so the balance controller can read body attitude and angular velocity through MuJoCo sensor data instead of directly relying on the free-joint `qpos/qvel` state.

This makes the simulation/control interface closer to a real robot while preserving the existing free-base model.

## Current Context

The model currently has:

- Free base via `base_freejoint`.
- 8 torque motors.
- A Python balance controller in `scripts/balance_control.py`.
- Body pitch currently computed from `data.qpos[3:7]`.
- Body pitch rate currently computed from `data.qvel[4]`.

MuJoCo already exposes perfect simulated base pose through the freejoint, but an IMU site/sensor gives us a cleaner controller-facing interface.

## MJCF Changes

Add an IMU site as a child of `base_link`:

```xml
<site name="base_imu_site" pos="0 0 0.08" size="0.015" rgba="0 0.7 1 1"/>
```

Add MuJoCo sensors:

```xml
<sensor>
  <gyro name="base_imu_gyro" site="base_imu_site"/>
  <accelerometer name="base_imu_accel" site="base_imu_site"/>
  <framequat name="base_imu_quat" objtype="site" objname="base_imu_site"/>
</sensor>
```

The converter `scripts/convert_urdf_to_mjcf.py` should generate these elements so regenerated `8dof_URDF/mjcf/robot.xml` stays consistent.

## Sensor Names and Data

Use stable names:

- Site: `base_imu_site`
- Gyro: `base_imu_gyro`
- Accelerometer: `base_imu_accel`
- Orientation: `base_imu_quat`

The controller should resolve sensor addresses by name, not by hard-coded `sensordata` offsets.

Expected sensor dimensions:

- `base_imu_gyro`: 3
- `base_imu_accel`: 3
- `base_imu_quat`: 4

## Controller Changes

Add helper functions to `scripts/balance_control.py`:

- `sensor_slice(model, sensor_name)`
- `has_base_imu(model)`
- `base_imu_quat(model, data)`
- `base_imu_gyro(model, data)`
- `base_pitch_from_imu(model, data)`
- `base_pitch_rate_from_imu(model, data)`

Balance control should prefer IMU data:

1. If `base_imu_quat` and `base_imu_gyro` exist, compute pitch and pitch rate from them.
2. If sensors are missing, fall back to the existing freejoint helpers.

The fallback is important so helper tests can still construct minimal models later if needed.

## Coordinate Convention

Keep the current near-upright pitch convention:

- Positive pitch is positive rotation about the Y axis near upright.
- `quat_to_pitch()` remains the shared quaternion-to-pitch function.
- IMU pitch rate uses the Y component of `base_imu_gyro`.

The exact gyro frame convention must be verified in tests against a known small angular velocity. If MuJoCo reports gyro in the site frame, this is still acceptable near upright because the IMU site is aligned with `base_link`.

## Analysis and Viewer

Existing analysis and viewer scripts should automatically benefit because they call `apply_balance_control()`.

No new analysis script is required for this phase. Existing balance artifacts should be regenerated after the sensor/control change:

```text
analysis/balance_results/
```

## Tests

Tests should cover:

1. `robot.xml` contains `base_imu_site`.
2. MuJoCo model has sensors named:
   - `base_imu_gyro`
   - `base_imu_accel`
   - `base_imu_quat`
3. Sensor dimensions are 3, 3, and 4.
4. Initial `data.sensordata` is finite after `mj_forward`.
5. IMU quaternion pitch matches freejoint pitch for a small known base pitch.
6. Balance controller uses IMU pitch when sensors exist.
7. Existing balance simulation remains finite and warning-free.

## README Update

Update the README to mention:

- An IMU site/sensor exists above `base_link`.
- The balance controller uses IMU orientation/gyro when available.
- Direct MuJoCo viewer can show the IMU site marker.

## Acceptance Criteria

This phase is complete when:

1. Regenerated MJCF contains the IMU site and sensors.
2. Tests verify IMU sensor existence, dimensions, and finite data.
3. Balance controller reads pitch and pitch rate from IMU when available.
4. Existing tests pass.
5. Balance analysis artifacts are regenerated.
6. README documents the IMU.

## Non-goals

This phase does not add noisy sensors, bias estimation, EKF, sensor fusion, magnetometer, GPS, or realistic IMU drift. The first IMU is an ideal MuJoCo sensor interface for control validation.
