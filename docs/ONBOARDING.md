# MuJoCo 8-DOF Wheeled Biped 快速上手指南

这份文档是给“重新打开这个项目”或“新接手项目”的入口说明。目标不是替代 `README.md`，而是帮助你用最短路径跑起来、看懂结构、知道下一步该从哪里下手。

> 说明：当前仓库还没有 `.understand-anything/knowledge-graph.json`，所以本指南基于项目源码、README、测试和分析脚本整理；后续可以运行 `/understand` 生成交互式知识图谱。

## Project Overview

这个项目是在 MuJoCo 中搭建一个 8 自由度双足轮式机器人模型。当前阶段的重点是模型正确性、动力学有限性、力矩电机控制链路，以及第一版原地机身 pitch 平衡。

目前已经具备：

- URDF 到 MJCF 的自动转换。
- `base_link` free-base 模型，不默认固定。
- 双轮初始刚好接触地面。
- 8 个 torque motor。
- 关节 PD 控制。
- 固定基座阶跃响应分析。
- free-base 动力学有限性验证。
- `base_link` 上方理想 IMU。
- 第一版基于双轮力矩的 pitch 平衡控制。

当前还没有完成：

- 稳健站立控制。
- 行走控制。
- 轨迹跟踪。
- 系统化平衡控制调参。

## Quick Start

所有命令默认在仓库根目录执行：

```powershell
cd D:\Workspace\Mujoco
```

先确认测试通过：

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

当前应看到类似：

```text
37 passed
```

打开纯 MuJoCo XML 模型：

```powershell
.\.venv\Scripts\python.exe -m mujoco.viewer --mjcf=8dof_URDF\mjcf\robot.xml
```

注意：这个方式只加载 XML，不运行 Python 控制器。

打开带 Python 平衡控制器的 viewer：

```powershell
.\.venv\Scripts\python.exe scripts\run_balance_viewer.py
```

重新生成 MJCF：

```powershell
.\.venv\Scripts\python.exe scripts\convert_urdf_to_mjcf.py
```

运行动力学分析：

```powershell
.\.venv\Scripts\python.exe scripts\analyze_dynamics.py --duration 1.0
```

运行平衡分析：

```powershell
.\.venv\Scripts\python.exe scripts\analyze_balance.py --duration 2.0
```

## Architecture Layers

### 1. Source Model Layer

关键文件：

- `8dof_URDF/urdf/robot.urdf`
- `8dof_URDF/meshes/*.STL`

这是原始机器人描述和简化网格来源。一般不要直接在生成后的 MJCF 里长期维护结构变化；结构性修改应优先回到转换脚本里实现。

### 2. MJCF Generation Layer

关键文件：

- `scripts/convert_urdf_to_mjcf.py`
- `8dof_URDF/mjcf/robot.xml`

`convert_urdf_to_mjcf.py` 是模型生成入口。它会：

- 校验 URDF 拓扑是否符合 9 link / 8 joint 预期。
- 把早期误写的 `yaw` 命名改成 `roll`。
- 保留 `base_link` 为 free base。
- 添加 8 个 torque motor。
- 覆盖 hip pitch 限位为 `[-1.22, 0.87]`。
- 添加默认 inactive 的 `fixed_base_weld`。
- 添加 `base_imu_site` 和三个 IMU sensor。
- 自动计算 base 初始高度，让左右轮刚好接触地面。
- 检查非轮子碰撞体是否初始穿地。

### 3. Joint Control Layer

关键文件：

- `scripts/pd_control.py`

这一层只处理“关节目标角 -> actuator torque”。它不做机身平衡，只提供通用工具：

- 关节和 actuator 的固定顺序映射。
- 目标角限幅。
- 默认 PD 增益计算。
- PD 力矩计算。
- `fixed_base_weld` 开关。

核心控制律：

```text
tau = Kp * (q_target - q) - Kd * qdot
tau = clip(tau, ctrlrange)
```

### 4. Balance Control Layer

关键文件：

- `scripts/balance_control.py`

这是当前第一版机身 pitch 平衡控制。它的策略是：

- 6 个腿部关节用 PD 保持名义站立姿态。
- 2 个轮子负责机身 pitch 调节。
- pitch 优先来自 IMU quaternion。
- pitch rate 优先来自 IMU gyro。
- 如果没有 IMU，则回退到 freejoint 的 `qpos/qvel`。
- 左右轮 actuator 使用相反符号，因为左右轮关节轴方向镜像。

当前平衡控制量大致为：

```text
tau_balance =
    kp_pitch * (pitch_target - pitch)
  + kd_pitch * (pitch_rate_target - pitch_rate)
  + kx       * (x_target - x)
  + kv       * (x_velocity_target - x_velocity)
```

然后写入左右轮：

```text
left_wheel_tau  = +tau_balance
right_wheel_tau = -tau_balance
```

### 5. Analysis Layer

关键文件：

- `scripts/analyze_dynamics.py`
- `scripts/analyze_balance.py`
- `analysis/results/`
- `analysis/balance_results/`

`analyze_dynamics.py` 做两类检查：

1. 固定基座单关节阶跃响应。
2. free-base 姿态保持有限性检查。

`analyze_balance.py` 跑当前平衡控制器，并输出：

- `balance_summary.csv`
- `balance_timeseries.csv`
- `balance_report.md`

这些报告用于判断数值是否有限、是否触发 MuJoCo warning、轮子是否饱和、pitch 是否变大。

### 6. Test Layer

关键文件：

- `tests/test_passive_mjcf.py`
- `tests/test_motor_control_dynamics.py`
- `tests/test_balance_control.py`

测试覆盖三块：

- 被动 MJCF 结构、命名、接触、初始姿态。
- 电机、限位、IMU、PD、固定基座动力学。
- IMU pitch/pitch-rate、左右轮力矩符号、平衡仿真有限性。

## Key Concepts

### Free Base

`base_link` 通过 `base_freejoint` 与世界连接，不默认固定。机器人会真实受到重力、接触和轮子力矩影响。

### fixed_base_weld

模型里有一个默认关闭的 equality weld：

```text
fixed_base_weld
```

分析固定基座阶跃响应时会临时启用它。正常 free-base 仿真中它保持关闭。

### Torque Motor

MuJoCo actuator 是力矩电机，不是位置伺服。位置控制由 Python 侧计算力矩并写入 `data.ctrl`。

### IMU

当前模型在 `base_link` 上方有理想 IMU：

```text
base_imu_site
base_imu_gyro
base_imu_accel
base_imu_quat
```

平衡控制器优先使用 IMU 姿态和角速度，这更接近真实机器人接口。

### Mirrored Wheel Signs

左右轮关节轴在世界坐标下方向相反。为了让两个轮子的物理滚动力矩方向一致，控制器对左右轮 actuator 使用反向符号。

## Guided Tour

建议按下面顺序读代码。

### Step 1: 从 README 开始

文件：

- `README.md`

先了解当前状态、命令入口、模型限制和分析脚本。

### Step 2: 看模型怎么生成

文件：

- `scripts/convert_urdf_to_mjcf.py`

重点看：

- `EXPECTED_LINKS`
- `EXPECTED_JOINTS`
- `CONTROLLED_JOINTS`
- `HIP_PITCH_RANGE`
- `TORQUE_LIMITS`
- `convert_urdf()`
- `add_link()`

这一步会解释为什么生成的 `robot.xml` 是现在这个结构。

### Step 3: 看关节控制

文件：

- `scripts/pd_control.py`

重点看：

- `JointControlMap`
- `build_joint_map()`
- `default_pd_gains()`
- `compute_pd_control()`
- `set_base_weld_active()`

理解这层后，就知道 8 个 actuator 的控制力矩是怎么来的。

### Step 4: 看平衡控制

文件：

- `scripts/balance_control.py`

重点看：

- `BalanceConfig`
- `quat_to_pitch()`
- `base_pitch_from_imu()`
- `base_pitch_rate_from_imu()`
- `compute_balance_control()`
- `WHEEL_ACTUATOR_SIGNS`

这是后续调平衡控制最常改的地方。

### Step 5: 看分析脚本

文件：

- `scripts/analyze_dynamics.py`
- `scripts/analyze_balance.py`

先跑脚本，再对照生成的 CSV/Markdown 报告看代码，会比只读代码更快。

### Step 6: 看测试

文件：

- `tests/test_passive_mjcf.py`
- `tests/test_motor_control_dynamics.py`
- `tests/test_balance_control.py`

测试是这个项目最可靠的“行为规格”。改模型或控制器之前，先看对应测试约束。

## File Map

| 文件 | 作用 |
| --- | --- |
| `README.md` | 项目总览、常用命令、当前状态 |
| `8dof_URDF/urdf/robot.urdf` | 原始 URDF 机器人描述 |
| `8dof_URDF/mjcf/robot.xml` | 当前 MuJoCo 主模型，由转换脚本生成 |
| `scripts/convert_urdf_to_mjcf.py` | URDF 到 MJCF 转换和模型校验 |
| `scripts/pd_control.py` | 关节 PD 控制工具 |
| `scripts/balance_control.py` | 第一版机身 pitch 平衡控制 |
| `scripts/analyze_dynamics.py` | 固定基座阶跃响应和 free-base 动力学检查 |
| `scripts/analyze_balance.py` | free-base 平衡控制仿真和报告生成 |
| `scripts/run_balance_viewer.py` | 带 Python 控制器的 MuJoCo viewer |
| `tests/test_passive_mjcf.py` | 被动模型结构/接触/稳定性测试 |
| `tests/test_motor_control_dynamics.py` | 电机、IMU、PD、动力学分析测试 |
| `tests/test_balance_control.py` | 平衡控制测试 |

## Complexity Hotspots

### `scripts/convert_urdf_to_mjcf.py`

这是最容易影响全局的文件。它同时处理：

- URDF 解析。
- MJCF 生成。
- 初始高度计算。
- 接触/穿地检查。
- actuator/equality/sensor 生成。

修改它之后必须跑：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_passive_mjcf.py tests\test_motor_control_dynamics.py -v
```

### `scripts/balance_control.py`

这是控制效果最敏感的文件。改增益、符号、pitch 约定、IMU 读取都可能改变机器人运动方向。

修改它之后优先跑：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py -v
.\.venv\Scripts\python.exe scripts\analyze_balance.py --duration 2.0
```

### 左右轮力矩方向

不要轻易删掉 `WHEEL_ACTUATOR_SIGNS`。左右轮关节轴是镜像的，控制器需要左右 actuator 反向写入，才能产生同方向的物理滚动力矩。

### 直接编辑 `robot.xml`

`robot.xml` 是生成产物。长期修改应回到 `convert_urdf_to_mjcf.py`，否则下次转换会覆盖手工改动。

## Common Pitfalls

- 直接打开 XML 不会运行 Python 控制器。
- 当前平衡控制还不能证明机器人稳健站立。
- 阶跃响应里的 `nan` 不一定是仿真错误，可能只是分析时间内没有达到指标。
- `base_imu_accel` 当前已生成，但第一版平衡控制主要使用 gyro 和 quaternion。
- 改 actuator 顺序会导致控制器把力矩打到错误关节。
- 当前仓库可能有一个未处理的 `base_link_mujoco.STL` 删除状态，需要单独决定恢复或正式删除。

## Recommended Next Work

建议后续按这个顺序继续：

1. 清理仓库状态，包括确认 `base_link_mujoco.STL` 是否恢复或正式删除。
2. 更新 README 中陈旧的测试数量。
3. 给 `analyze_balance.py` 增加更多诊断指标：
   - wheel velocity
   - base x drift
   - control saturation ratio
   - pitch zero-crossing
4. 系统检查 pitch 控制方向：
   - 小 pitch 初始扰动
   - 左右轮力矩方向
   - base pitch 响应方向
5. 做 gain sweep，而不是手调一组参数。
6. 如果 PD 原型不够，再考虑线性化倒立摆、LQR 或 MPC。

## Before You Commit Changes

常用验证命令：

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

如果只改模型生成：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_passive_mjcf.py tests\test_motor_control_dynamics.py -v
```

如果只改平衡控制：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_balance_control.py -v
```

如果想确认 MuJoCo 模型能直接加载：

```powershell
.\.venv\Scripts\python.exe -m mujoco.viewer --mjcf=8dof_URDF\mjcf\robot.xml
```

