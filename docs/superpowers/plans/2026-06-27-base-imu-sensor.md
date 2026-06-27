# Base IMU Sensor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an ideal MuJoCo IMU site/sensor above `base_link` and make the balance controller prefer IMU quaternion/gyro data for body pitch and pitch-rate feedback.

**Architecture:** The URDF-to-MJCF converter owns the generated site and sensor XML. `scripts/balance_control.py` owns sensor lookup and IMU-based state extraction with freejoint fallback. Existing analysis/viewer scripts continue to use `apply_balance_control()` and therefore automatically benefit from IMU feedback.

**Tech Stack:** Python 3.10, MuJoCo Python API, NumPy, pytest, XML generation through `xml.etree.ElementTree`.

---

## File Structure

- Modify `scripts/convert_urdf_to_mjcf.py`
  - Add `base_imu_site` as a child of `base_link`.
  - Add `<sensor>` with gyro, accelerometer, and framequat.
- Modify `scripts/balance_control.py`
  - Add sensor lookup helpers.
  - Prefer IMU quaternion/gyro for pitch and pitch rate.
  - Preserve freejoint fallback.
- Modify `tests/test_balance_control.py`
  - Add IMU existence/dimension/finite-data tests.
  - Add IMU-vs-freejoint pitch tests.
  - Add controller-uses-IMU test.
- Modify `tests/test_motor_control_dynamics.py` or `tests/test_passive_mjcf.py`
  - Add structural tests for generated site/sensors if cleaner there.
- Regenerate `8dof_URDF/mjcf/robot.xml`.
- Regenerate `analysis/balance_results/`.
- Modify `README.md`.

---

### Task 1: Generate IMU site and sensors in MJCF

**Files:**
- Modify: `scripts/convert_urdf_to_mjcf.py`
- Modify: `tests/test_motor_control_dynamics.py`
- Generate: `8dof_URDF/mjcf/robot.xml`

- [ ] **Step 1: Add failing MJCF IMU structure tests**

Append to `tests/test_motor_control_dynamics.py`:

```python
def test_base_imu_site_and_sensors_exist(model):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "base_imu_site")
    assert site_id >= 0
    assert np.allclose(model.site_pos[site_id], [0.0, 0.0, 0.08])

    expected = {
        "base_imu_gyro": 3,
        "base_imu_accel": 3,
        "base_imu_quat": 4,
    }
    for sensor_name, sensor_dim in expected.items():
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        assert sensor_id >= 0
        assert model.sensor_dim[sensor_id] == sensor_dim


def test_base_imu_sensor_data_is_finite(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert model.nsensor >= 3
    assert model.nsensordata >= 10
    assert np.isfinite(data.sensordata).all()
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_motor_control_dynamics.py::test_base_imu_site_and_sensors_exist -v
```

Expected: fail because `base_imu_site` does not exist.

- [ ] **Step 3: Add site generation to `base_link`**

In `scripts/convert_urdf_to_mjcf.py`, inside `add_link()` after the `body = ET.SubElement(parent_body, "body", attributes)` line and after adding the freejoint for root body, add:

```python
        if link_name == "base_link":
            ET.SubElement(
                body,
                "site",
                {
                    "name": "base_imu_site",
                    "pos": "0 0 0.08",
                    "size": "0.015",
                    "rgba": "0 0.7 1 1",
                },
            )
```

Keep this site visual and non-colliding; a MuJoCo `<site>` is enough.

- [ ] **Step 4: Add sensor section**

After the equality section and before `tree = ET.ElementTree(model_root)`, add:

```python
    sensor = ET.SubElement(model_root, "sensor")
    ET.SubElement(sensor, "gyro", {"name": "base_imu_gyro", "site": "base_imu_site"})
    ET.SubElement(sensor, "accelerometer", {"name": "base_imu_accel", "site": "base_imu_site"})
    ET.SubElement(
        sensor,
        "framequat",
        {
            "name": "base_imu_quat",
            "objtype": "site",
            "objname": "base_imu_site",
        },
    )
```

- [ ] **Step 5: Regenerate MJCF**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\convert_urdf_to_mjcf.py
```

Expected: exits 0 and prints `D:\Workspace\Mujoco\8dof_URDF\mjcf\robot.xml`.

- [ ] **Step 6: Run IMU structure tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_motor_control_dynamics.py::test_base_imu_site_and_sensors_exist tests\test_motor_control_dynamics.py::test_base_imu_sensor_data_is_finite -v
```

Expected: both tests pass.

- [ ] **Step 7: Commit**

Run:

```powershell
git add scripts\convert_urdf_to_mjcf.py 8dof_URDF\mjcf\robot.xml tests\test_motor_control_dynamics.py
git commit -m "feat: add base imu sensors to mjcf"
```

---

### Task 2: Use IMU data in balance controller

**Files:**
- Modify: `scripts/balance_control.py`
- Modify: `tests/test_balance_control.py`

- [ ] **Step 1: Add failing IMU controller tests**

Append to `tests/test_balance_control.py`:

```python
def test_base_imu_pitch_matches_freejoint_pitch(model):
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.qpos[3:7] = quat_y_rotation(0.12)
    mujoco.mj_forward(model, data)
    from scripts.balance_control import base_pitch_from_imu, has_base_imu

    assert has_base_imu(model)
    assert base_pitch_from_imu(model, data) == pytest.approx(base_pitch(data), abs=1e-8)


def test_balance_controller_prefers_imu_pitch(model, monkeypatch):
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.qpos[3:7] = quat_y_rotation(0.1)
    mujoco.mj_forward(model, data)
    joint_map = build_joint_map(model)

    import scripts.balance_control as balance_control

    calls = {"imu": 0}
    original = balance_control.base_pitch_from_imu

    def recording_pitch_from_imu(model_arg, data_arg):
        calls["imu"] += 1
        return original(model_arg, data_arg)

    monkeypatch.setattr(balance_control, "base_pitch_from_imu", recording_pitch_from_imu)
    compute_balance_control(model, data, joint_map, BalanceConfig())
    assert calls["imu"] == 1
```

Also update the import block at the top of `tests/test_balance_control.py` to include:

```python
    base_imu_gyro,
    base_imu_quat,
    base_pitch_from_imu,
    base_pitch_rate_from_imu,
    has_base_imu,
    sensor_slice,
```

or use local imports as shown above.

- [ ] **Step 2: Run tests and verify missing helper failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py::test_base_imu_pitch_matches_freejoint_pitch -v
```

Expected: fail with missing `base_pitch_from_imu` / `has_base_imu`.

- [ ] **Step 3: Implement sensor helpers**

In `scripts/balance_control.py`, add:

```python
BASE_IMU_GYRO = "base_imu_gyro"
BASE_IMU_ACCEL = "base_imu_accel"
BASE_IMU_QUAT = "base_imu_quat"


def sensor_slice(model: mujoco.MjModel, sensor_name: str) -> slice:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sensor_id < 0:
        raise ValueError(f"Model is missing sensor {sensor_name!r}")
    start = int(model.sensor_adr[sensor_id])
    stop = start + int(model.sensor_dim[sensor_id])
    return slice(start, stop)


def has_base_imu(model: mujoco.MjModel) -> bool:
    return (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, BASE_IMU_GYRO) >= 0
        and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, BASE_IMU_QUAT) >= 0
    )


def base_imu_quat(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    values = data.sensordata[sensor_slice(model, BASE_IMU_QUAT)]
    if values.shape[0] != 4:
        raise ValueError("base_imu_quat must have dimension 4")
    return np.asarray(values, dtype=float)


def base_imu_gyro(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    values = data.sensordata[sensor_slice(model, BASE_IMU_GYRO)]
    if values.shape[0] != 3:
        raise ValueError("base_imu_gyro must have dimension 3")
    return np.asarray(values, dtype=float)


def base_pitch_from_imu(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    return quat_to_pitch(base_imu_quat(model, data))


def base_pitch_rate_from_imu(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    return float(base_imu_gyro(model, data)[1])
```

- [ ] **Step 4: Update `compute_balance_control` to prefer IMU**

Replace:

```python
    pitch = base_pitch(data)
    pitch_rate = base_pitch_rate(data)
```

with:

```python
    if has_base_imu(model):
        pitch = base_pitch_from_imu(model, data)
        pitch_rate = base_pitch_rate_from_imu(model, data)
    else:
        pitch = base_pitch(data)
        pitch_rate = base_pitch_rate(data)
```

- [ ] **Step 5: Run controller IMU tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py::test_base_imu_pitch_matches_freejoint_pitch tests\test_balance_control.py::test_balance_controller_prefers_imu_pitch -v
```

Expected: both tests pass.

- [ ] **Step 6: Run full balance tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py -v
```

Expected: all balance tests pass.

- [ ] **Step 7: Commit**

Run:

```powershell
git add scripts\balance_control.py tests\test_balance_control.py
git commit -m "feat: use base imu in balance controller"
```

---

### Task 3: Refresh analysis artifacts and README

**Files:**
- Modify: `README.md`
- Generate: `analysis/balance_results/balance_summary.csv`
- Generate: `analysis/balance_results/balance_timeseries.csv`
- Generate: `analysis/balance_results/balance_report.md`

- [ ] **Step 1: Regenerate balance analysis**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_balance.py --duration 2.0
```

Expected: exits 0 and writes `analysis\balance_results\*`.

- [ ] **Step 2: Update README**

In `README.md`, add to the model/control description:

```markdown
## IMU 传感器

模型在 `base_link` 上方添加了 `base_imu_site`，并生成理想 MuJoCo IMU 传感器：

- `base_imu_gyro`
- `base_imu_accel`
- `base_imu_quat`

平衡控制器会优先使用 IMU 的姿态四元数和角速度；如果模型没有这些传感器，则回退到 freejoint 的 `qpos/qvel`。
```

Also update the balance-control section to mention that pitch feedback now comes from IMU when available.

- [ ] **Step 3: Verify README and artifacts**

Run:

```powershell
Select-String -Path README.md -Pattern 'base_imu_site','base_imu_gyro','base_imu_accel','base_imu_quat','IMU'
Test-Path analysis\balance_results\balance_summary.csv
Test-Path analysis\balance_results\balance_timeseries.csv
Test-Path analysis\balance_results\balance_report.md
```

Expected: README patterns found and all artifact paths exist.

- [ ] **Step 4: Commit**

Run:

```powershell
git add README.md analysis\balance_results
git commit -m "docs: document base imu sensors"
```

---

### Task 4: Final verification

**Files:**
- No new files expected unless regeneration changes tracked artifacts.

- [ ] **Step 1: Run full tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

Expected: all tests pass.

- [ ] **Step 2: Run direct model/sensor diagnostic**

Run:

```powershell
@'
from pathlib import Path
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path(str(Path("8dof_URDF/mjcf/robot.xml")))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
print("nsite=", model.nsite, "nsensor=", model.nsensor, "nsensordata=", model.nsensordata)
for name in ["base_imu_site", "base_imu_gyro", "base_imu_accel", "base_imu_quat"]:
    obj = mujoco.mjtObj.mjOBJ_SITE if name.endswith("_site") else mujoco.mjtObj.mjOBJ_SENSOR
    print(name, mujoco.mj_name2id(model, obj, name))
print("sensordata_finite=", np.isfinite(data.sensordata).all())
'@ | .\.venv\Scripts\python.exe -
```

Expected:

- site id and sensor ids are nonnegative.
- `sensordata_finite=True`.

- [ ] **Step 3: Run balance analysis smoke**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_balance.py --duration 0.5 --output-dir analysis\imu_verify_results
```

Expected: exits 0 and writes the three balance files. Remove `analysis\imu_verify_results` after inspection.

- [ ] **Step 4: Check git status**

Run:

```powershell
git status --short
```

Expected: no output.

- [ ] **Step 5: Commit if verification regenerated tracked artifacts**

If tracked files changed:

```powershell
git add 8dof_URDF\mjcf\robot.xml analysis\balance_results README.md
git commit -m "test: verify base imu sensors"
```

If no tracked files changed, do not create an empty commit.

---

## Self-Review

- Spec coverage:
  - MJCF site/sensors: Task 1.
  - Sensor helper functions and IMU-preferred control: Task 2.
  - Balance artifact regeneration and README docs: Task 3.
  - Final tests and diagnostics: Task 4.
- Placeholder scan:
  - No unfinished placeholder markers.
  - Each implementation task includes concrete tests, commands, and code.
- Type consistency:
  - Sensor names are consistent across spec, tests, converter, and controller:
    - `base_imu_site`
    - `base_imu_gyro`
    - `base_imu_accel`
    - `base_imu_quat`
