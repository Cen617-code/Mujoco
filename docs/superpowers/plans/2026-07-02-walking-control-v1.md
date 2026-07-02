# Walking Control v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic wheel-speed walking control with analysis reports and a MuJoCo viewer entry point.

**Architecture:** Implement walking as a thin layer over the existing standing balance controller. `scripts/walking_control.py` converts semantic forward velocity into world-X velocity targets with a ramp; `scripts/analyze_walking.py` verifies the default run; `scripts/run_walking_viewer.py` lets the user watch the same controller in MuJoCo.

**Tech Stack:** Python, MuJoCo Python API, pytest, CSV/Markdown reports.

---

### Task 1: Walking control API

**Files:**
- Create: `scripts/walking_control.py`
- Modify: `tests/test_balance_control.py`

- [ ] **Step 1: Write failing tests**

Add tests that import:

```python
from scripts.walking_control import (
    DEFAULT_FORWARD_VELOCITY,
    DEFAULT_RAMP_TIME,
    DEFAULT_WALKING_KV,
    WalkingConfig,
    ramped_forward_velocity,
    balance_x_velocity_target,
)
```

Assert:

```python
assert DEFAULT_FORWARD_VELOCITY == pytest.approx(0.25)
assert DEFAULT_RAMP_TIME == pytest.approx(2.0)
assert DEFAULT_WALKING_KV == pytest.approx(6.0)
assert ramped_forward_velocity(WalkingConfig(forward_velocity=0.25, ramp_time=2.0), 1.0) == pytest.approx(0.125)
assert balance_x_velocity_target(WalkingConfig(forward_velocity=0.25), 3.0) == pytest.approx(0.25)
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py -k walking -v
```

Expected: fails because `scripts.walking_control` does not exist.

- [ ] **Step 3: Implement minimal walking control**

Create `WalkingConfig`, `WalkingState`, `ramped_forward_velocity()`, `balance_x_velocity_target()`, and `apply_walking_control()`.

- [ ] **Step 4: Verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py -k walking -v
```

Expected: walking control tests pass.

### Task 2: Walking analysis

**Files:**
- Create: `scripts/analyze_walking.py`
- Modify: `tests/test_balance_control.py`

- [ ] **Step 1: Write failing tests**

Add tests for:

```python
from scripts.analyze_walking import run_walking_simulation, write_walking_results
```

Assert default run meets:

```python
result = run_walking_simulation(model, duration=8.0)
assert result.meets_walking_objective
assert result.forward_distance > 1.0
assert abs(result.average_forward_velocity_last_window - 0.25) < 0.08
assert result.peak_abs_pitch < 0.3
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py -k walking -v
```

Expected: fails because `scripts.analyze_walking` does not exist.

- [ ] **Step 3: Implement analysis script**

Follow `scripts/analyze_disturbance.py` patterns for finite checks, contact checks, summary CSV, timeseries CSV, and Markdown report.

- [ ] **Step 4: Verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py -k walking -v
```

Expected: all walking tests pass.

### Task 3: Walking viewer

**Files:**
- Create: `scripts/run_walking_viewer.py`
- Modify: `tests/test_balance_control.py`

- [ ] **Step 1: Write failing help test**

Add a CLI help test that verifies:

```python
assert "--velocity" in completed.stdout
assert "--ramp-time" in completed.stdout
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py -k walking_viewer -v
```

Expected: fails because `scripts/run_walking_viewer.py` does not exist.

- [ ] **Step 3: Implement viewer script**

Create a passive viewer loop that calls `apply_walking_control()` every step.

- [ ] **Step 4: Verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py -k walking_viewer -v
```

Expected: viewer help test passes.

### Task 4: Results, docs, and full verification

**Files:**
- Modify: `README.md`
- Create/update: `analysis/walking_results/*`

- [ ] **Step 1: Generate walking report**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_walking.py
```

Expected output directory: `analysis\walking_results`.

- [ ] **Step 2: Update README**

Document:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_walking.py
.\.venv\Scripts\python.exe scripts\run_walking_viewer.py --no-regenerate --velocity 0.25
```

- [ ] **Step 3: Run full test suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit and push**

Run:

```powershell
git add README.md scripts/walking_control.py scripts/analyze_walking.py scripts/run_walking_viewer.py tests/test_balance_control.py analysis/walking_results
git add -f docs/superpowers/specs/2026-07-02-walking-control-v1-design.md docs/superpowers/plans/2026-07-02-walking-control-v1.md
git commit -m "feat: add wheel-speed walking control"
git push origin main
```
