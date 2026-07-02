# Walking Control v1 Design

## Goal

在当前稳定站立和 `±50 N` 前后短推抗扰动基础上，实现第一版可复现的轮式行走控制。行走 v1 的目标是让机器人在保持当前对称腿姿和 pitch 平衡的同时，沿自身前进方向稳定滚动。

## Scope

- 行走形式：轮式速度行走，不做摆腿步态。
- 默认命令：`forward_velocity = 0.25 m/s`。
- 默认速度斜坡：`ramp_time = 2.0 s`。
- 默认仿真时间：`duration = 8.0 s`。
- 默认速度增益：`kv = 6.0`。
- 腿部目标：沿用 `standing_leg_targets()` 的近零位对称姿态。

## Coordinate Convention

当前模型中，视觉上的“向前”对应 world X 负方向。因此行走控制 API 采用：

```text
positive forward_velocity => target world x velocity is negative
```

也就是说用户运行 `--velocity 0.25` 时，机器人应该朝 world X 负方向前进。由于现有平衡控制器内部已经处理了左右轮 actuator 符号，底层 `BalanceConfig.x_velocity_target` 使用正的控制命令；这不是最终 world-X 速度的符号。

## Architecture

新增 `scripts/walking_control.py`：

- 定义 `WalkingConfig`
- 定义 `WalkingState`
- 将用户语义的 `forward_velocity` 转成当前平衡控制器需要的 x 速度命令
- 每个仿真步复用 `apply_balance_control()`，只修改 `BalanceConfig.x_velocity_target`
- 固定 `x_target=None` 和 `kx=0.0`，避免站立控制里的原点锁阻止前进

新增 `scripts/analyze_walking.py`：

- 运行 deterministic walking v1 仿真
- 输出 summary、timeseries 和 Markdown 报告
- 验证 warning、finite、触地、pitch、速度跟踪、位移和轮子力矩饱和

新增 `scripts/run_walking_viewer.py`：

- 在 MuJoCo Viewer 中运行同一套行走控制循环
- 支持 `--velocity`、`--ramp-time`、`--duration`

## Acceptance Criteria

默认行走 v1 必须满足：

- MuJoCo warning 数为 `0`
- `qpos/qvel/control` 保持 finite
- 只有左右轮允许接触地面
- peak `|pitch| < 0.3 rad`
- forward distance `> 1.0 m`
- 后 2 秒平均 forward velocity 在目标速度 `0.25 m/s` 的 `±0.08 m/s` 内
- 轮子力矩饱和比例 `< 0.2`

## Outputs

`scripts/analyze_walking.py` 输出到 `analysis/walking_results/`：

- `walking_summary.csv`
- `walking_timeseries.csv`
- `walking_report.md`

README 增加行走控制 v1 的运行命令、viewer 命令和最近一次默认分析指标。

## Non-goals

- 不做摆腿/腿部周期轨迹
- 不做转向
- 不做停止状态机
- 不做侧向抗扰动或随机扰动
- 不做 MPC、LQR、全身 QP 或轨迹优化
