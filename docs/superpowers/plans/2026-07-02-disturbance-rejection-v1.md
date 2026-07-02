# Disturbance Rejection v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible ±50 N push-disturbance analysis and viewer support for the current standing controller.

**Architecture:** Create a focused `scripts/analyze_disturbance.py` module that applies an external X-force to `base_link` during a configurable time window while reusing the existing standing controller. Extend `scripts/run_balance_viewer.py` with matching push arguments for visual validation.

**Tech Stack:** Python, MuJoCo Python API, pytest, CSV/Markdown reports.

---

### Task 1: Disturbance analysis tests

**Files:**
- Modify: `tests/test_balance_control.py`
- Create later: `scripts/analyze_disturbance.py`

- [ ] **Step 1: Write failing tests**

Add tests that import `run_disturbance_simulation`, `run_default_disturbance_suite`, `write_disturbance_results`, and `DEFAULT_PUSH_FORCES` from `scripts.analyze_disturbance`.

The tests should assert that:

```python
assert DEFAULT_PUSH_FORCES == (-50.0, 50.0)
result = run_disturbance_simulation(model, push_force=50.0, duration=6.0)
assert result.meets_disturbance_objective
rows = run_default_disturbance_suite(model, duration=6.0)
assert {row.push_force for row in rows} == {-50.0, 50.0}
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py -k disturbance -v
```

Expected: fail with `ModuleNotFoundError: No module named 'scripts.analyze_disturbance'`.

### Task 2: Implement disturbance analysis

**Files:**
- Create: `scripts/analyze_disturbance.py`

- [ ] **Step 1: Add dataclasses and constants**

Create:

```python
DEFAULT_PUSH_FORCES = (-50.0, 50.0)
DEFAULT_PUSH_START = 1.0
DEFAULT_PUSH_DURATION = 0.1
DEFAULT_DURATION = 6.0
DISTURBANCE_FINAL_ABS_PITCH_LIMIT = 0.18
DISTURBANCE_PEAK_ABS_PITCH_LIMIT = 0.45
DISTURBANCE_PEAK_ABS_X_DRIFT_LIMIT = 0.5
DISTURBANCE_WHEEL_SATURATION_LIMIT = 0.2
```

- [ ] **Step 2: Implement simulation loop**

Follow `scripts/analyze_balance.py` structure, but set:

```python
if push_start <= data.time < push_start + push_duration:
    data.xfrc_applied[base_body_id, 0] = push_force
else:
    data.xfrc_applied[:] = 0.0
```

- [ ] **Step 3: Write CSV and Markdown outputs**

Write one summary row per push force and one timeseries row per sample.

- [ ] **Step 4: Run GREEN tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py -k disturbance -v
```

Expected: all disturbance tests pass.

### Task 3: Viewer push arguments

**Files:**
- Modify: `scripts/run_balance_viewer.py`
- Modify: `tests/test_balance_control.py`

- [ ] **Step 1: Write failing help test**

Update `test_run_balance_viewer_help_succeeds` to assert `--push-force`, `--push-start`, and `--push-duration` appear in help output.

- [ ] **Step 2: Verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py::test_run_balance_viewer_help_succeeds -v
```

Expected: fail because the push arguments are absent.

- [ ] **Step 3: Implement viewer arguments**

Add optional arguments with defaults:

```python
--push-force 0.0
--push-start 1.0
--push-duration 0.1
```

Use `data.xfrc_applied[base_link, 0]` in the viewer loop.

- [ ] **Step 4: Verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py::test_run_balance_viewer_help_succeeds -v
```

Expected: pass.

### Task 4: Reports, docs, and full verification

**Files:**
- Modify: `README.md`
- Create/update: `analysis/disturbance_results/*`

- [ ] **Step 1: Run default analysis**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_disturbance.py
```

Expected output directory: `analysis\disturbance_results`.

- [ ] **Step 2: Update README**

Document:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_disturbance.py
.\.venv\Scripts\python.exe scripts\run_balance_viewer.py --no-regenerate --push-force 50 --push-start 1.0 --push-duration 0.1
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
git add docs/superpowers/specs/2026-07-02-disturbance-rejection-v1-design.md docs/superpowers/plans/2026-07-02-disturbance-rejection-v1.md scripts/analyze_disturbance.py scripts/run_balance_viewer.py tests/test_balance_control.py README.md analysis/disturbance_results
git commit -m "feat: add push disturbance analysis"
git push origin main
```
