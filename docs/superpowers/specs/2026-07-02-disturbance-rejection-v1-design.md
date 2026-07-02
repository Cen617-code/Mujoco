# Disturbance Rejection v1 Design

## Goal

让当前 `standing-stable-v1` 站立控制具备第一版可复现的抗扰动验证能力。抗扰动 v1 聚焦前后方向短时水平推力，不扩展到侧向、转向、行走或随机地形。

## Scope

- 扰动对象：`base_link`
- 扰动方向：世界坐标 X 方向，即机器人前后方向
- 默认扰动：`-50 N` 和 `+50 N`
- 默认持续时间：`0.1 s`
- 默认开始时间：仿真第 `1.0 s`
- 默认总仿真时间：`6.0 s`

## Architecture

新增一个独立分析脚本 `scripts/analyze_disturbance.py`。它复用现有的 MJCF、`default_standing_config()`、`standing_leg_targets()` 和 `apply_balance_control()`，在指定时间窗口内通过 `data.xfrc_applied[base_link, 0]` 施加水平外力，并记录恢复指标。

`scripts/run_balance_viewer.py` 增加相同的扰动参数，让 MuJoCo Viewer 里能直接看到“被推一下后是否恢复”。默认不推；只有显式传入 `--push-force` 时才施加外力。

## Acceptance Criteria

每个默认扰动场景都必须满足：

- MuJoCo warning 数为 `0`
- `qpos/qvel/control` 保持 finite
- 只有左右轮允许接触地面
- peak `|pitch| < 0.45 rad`
- peak `|x drift| < 0.5 m`
- final `|pitch| < 0.18 rad`
- 轮子力矩饱和比例 `< 0.2`

## Outputs

`scripts/analyze_disturbance.py` 输出到 `analysis/disturbance_results/`：

- `disturbance_summary.csv`
- `disturbance_timeseries.csv`
- `disturbance_report.md`

README 增加一节说明如何运行默认 ±50 N 扰动分析，以及如何在 viewer 中手动施加单次推力。

## Non-goals

- 不做侧向 roll 抗扰动
- 不做随机推力或 Monte Carlo sweep
- 不做 MPC、LQR、全身 QP 或状态估计器扩展
- 不修改当前稳定站立零位姿态
