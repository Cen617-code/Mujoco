# MuJoCo 8-DOF Wheeled Biped

一个基于 MuJoCo 的 8 自由度双足轮式机器人模型项目，包含 URDF 到 MJCF 转换、free-base 动力学模型、8 个力矩电机、Python PD 关节控制、固定基座阶跃响应分析、free-base 数值稳定性验证，以及第一版轮式站立平衡诊断/调参流程。

## 当前状态

- 已从 `8dof_URDF/urdf/robot.urdf` 生成原生 MJCF：`8dof_URDF/mjcf/robot.xml`
- `base_link` 是 free base，不默认固定
- 左右 `hip_pitch` 限位为 `[-1.22, 0.87]` rad
- 包含 8 个 torque motor：
  - roll：±20 N·m
  - hip_pitch：±30 N·m
  - knee：±30 N·m
  - wheel：±10 N·m
- 包含外部 Python PD 控制工具
- 在 `base_link` 上方包含理想 IMU 传感器
- 包含固定基座单关节阶跃响应分析
- 包含 free-base 姿态保持有限动力学验证
- 包含原地站立平衡分析、确定性参数 sweep 和结果报告

## 快速开始

所有命令默认在仓库根目录运行：

```powershell
cd D:\Workspace\Mujoco
```

### 打开 MuJoCo Viewer

```powershell
.\.venv\Scripts\python.exe -m mujoco.viewer --mjcf=8dof_URDF\mjcf\robot.xml
```

Viewer 直接打开 XML 时，只会加载 MuJoCo 模型本身；Python PD 控制器不会自动运行。

### 重新生成 MJCF

```powershell
.\.venv\Scripts\python.exe scripts\convert_urdf_to_mjcf.py
```

输出文件：

```text
8dof_URDF\mjcf\robot.xml
```

### 运行动力学分析

```powershell
.\.venv\Scripts\python.exe scripts\analyze_dynamics.py --duration 1.0
```

输出目录：

```text
analysis\results\
```

### 运行测试

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

## 项目结构

```text
8dof_URDF/
  urdf/
    robot.urdf
  meshes/
    *.STL
  mjcf/
    robot.xml
analysis/
  results/
    dynamics_report.md
    step_response_metrics.csv
    free_base_summary.csv
  balance_results/
    balance_report.md
    balance_summary.csv
    balance_timeseries.csv
  standing_tuning/
    standing_tuning_report.md
    standing_tuning_results.csv
    standing_best_config.json
scripts/
  convert_urdf_to_mjcf.py
  pd_control.py
  analyze_dynamics.py
  analyze_balance.py
  tune_standing_balance.py
  run_balance_viewer.py
tests/
  test_passive_mjcf.py
  test_motor_control_dynamics.py
  test_balance_control.py
```

## 模型说明

`robot.xml` 是当前主要 MuJoCo 模型。它保留 free base，并通过一个默认关闭的 `fixed_base_weld` equality 支持固定基座分析。

语义上原 URDF 中写错的 `yaw` 已在 MJCF 命名中改为 `roll`。roll 关节轴保持 URDF 原始局部轴设置，因为关节坐标系会把它映射到世界坐标的 x 方向。

## IMU 传感器

模型在 `base_link` 上方添加了 `base_imu_site`，并生成理想 MuJoCo IMU 传感器：

- `base_imu_gyro`
- `base_imu_accel`
- `base_imu_quat`

平衡控制器会优先使用 IMU 的姿态四元数和角速度；如果模型没有这些传感器，则回退到 freejoint 的 `qpos/qvel`。

## 控制说明

电机是 MuJoCo 原生 torque motor。PD 控制在 Python 中实现：

```text
tau = Kp * (q_target - q) - Kd * qdot
tau = clip(tau, ctrlrange)
```

相关代码在：

```text
scripts\pd_control.py
```

## 动力学分析

分析脚本会执行两类验证：

1. 固定基座单关节阶跃响应
   - 临时启用 `fixed_base_weld`
   - 8 个关节逐个阶跃
   - 输出 rise time、overshoot、settling time、稳态误差、峰值力矩、饱和比例
2. free-base 姿态保持验证
   - 关闭 `fixed_base_weld`
   - 只做关节 PD，不做机身平衡控制
   - 检查 qpos/qvel/control 是否有限，以及 MuJoCo warning 是否为 0

分析报告：

```text
analysis\results\dynamics_report.md
```

## 机身平衡控制

第一版机身平衡控制使用腿部 PD 保持名义站立关节姿态，并用左右轮调节机身 pitch。pitch 反馈在模型提供 IMU 时来自 `base_imu_quat` 和 `base_imu_gyro`；如果没有 IMU 传感器，则回退到 freejoint 的 `qpos/qvel`。由于左右轮关节轴在世界坐标中方向相反，控制器会给左右轮 actuator 反向控制量，使物理轮滚动力矩方向一致。它是原地平衡原型，不是行走控制器。

当前默认站立姿态会让 hip/knee 稍微弯曲：

```text
hip_pitch = -0.15 rad
knee      =  0.35 rad
```

运行平衡分析：

```powershell
.\.venv\Scripts\python.exe scripts\analyze_balance.py --duration 2.0
```

查看受控仿真：

```powershell
.\.venv\Scripts\python.exe scripts\run_balance_viewer.py
```

输出目录：

```text
analysis\balance_results\
```

### 稳健站立控制 v1

稳健站立 v1 的中等验收目标是 2 秒 free-base 仿真：

- MuJoCo warning 数为 0
- 状态保持 finite
- final `|pitch| < 0.25 rad`
- peak `|pitch| < 0.5 rad`
- peak `|x drift| < 0.3 m`

当前已实现确定性 sweep 调参入口：

```powershell
.\.venv\Scripts\python.exe scripts\tune_standing_balance.py --duration 2.0
```

输出目录：

```text
analysis\standing_tuning\
```

最近一次 216 组候选 sweep 已达到目标，最佳候选已固化为默认站立控制参数：

```text
hip_pitch=0.20, knee=0.35, kp_pitch=20.0, kd_pitch=6.0, kx=0.0, kv=20.0
```

对应指标：

```text
warning_count=0
finite=True
final |pitch|=0.01333 rad
peak |pitch|=0.133335 rad
peak |x drift|=0.085737 m
wheel torque saturation fraction=0.000
standing objective met=True
```

默认 2 秒分析也通过中等验收目标：

```powershell
.\.venv\Scripts\python.exe scripts\analyze_balance.py --duration 2.0
```

如果要在 MuJoCo Viewer 里看受控站立，不要直接打开 XML；请运行带 Python 控制循环的 viewer：

```powershell
.\.venv\Scripts\python.exe scripts\run_balance_viewer.py --no-regenerate
```

这个 viewer 会使用 `scripts/balance_control.py` 中的默认稳健站立参数，每个仿真步写入腿部 PD 和左右轮平衡力矩。

## 当前限制

- 目前只是第一版原地 pitch 平衡原型，不是完整轮式双足平衡/行走控制器。
- free-base 仿真允许机器人按真实动力学自然倒下。
- Viewer 直接打开 XML 时不会自动运行 Python PD 控制器。
- 当前稳健站立控制已通过 2 秒中等验收目标，但仍只是原地站立基线，不代表已经具备抗大扰动或行走能力。
- 阶跃响应中的 `nan` 表示该关节在分析时间内没有达到对应指标，例如没有达到 90% 上升或没有进入 2% 稳态区间。

## 最近验证结果

当前版本测试结果：

```text
57 passed in tests
```

模型诊断摘要：

```text
nq=15, nv=14, nu=8, neq=1
fixed_base_weld 默认 inactive
MuJoCo warnings=0
```
