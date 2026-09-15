# Jug：去掉速度 reward 与 3 m 移动实验（2026-09-15）

## 桶颈接近点是怎么来的

是手工选择的 jug 局部坐标 `(0, 0, 0.17)` m，现已作为
`SpotJugManipulationConfig.gripper_target_local` 暴露。
坐标原点是**网格几何中心**，不是底面，也不是偏低 3 cm 的惯性质心；局部 +z 指向桶口。
桶直立落地时该点离地约 `0.2415+0.17=0.4115 m`，距网格最高点约 7.14 cm。
桶发生转动时，世界坐标为 `p_jug + R_jug @ [0,0,0.17]`，并不是一个固定世界坐标。

它在桶颈中轴内部，作为指尖接近奖励的目标；不是表面接触点、感知检测关键点、焊接或抓取约束。
本次读取实际网格验证：z∈[0.167,0.173] m 的顶点到 z 轴半径约 2.62～3.25 cm，属于桶颈区域。
[网格与目标点示意](../out/jug_tasks/no_velocity_20260915/neck_target.jpg)。

## 本次消融定义

三个原任务均采用上一轮的 seed=0、2 s horizon、48 rollouts、4 knots、每次 2 轮 CEM。
初始状态和其他 reward 权重保留，仅通过 `--no-velocity-reward` 关闭：

```text
w_linear_velocity = 0
w_angular_velocity = 0
w_roll = 0   # 对 rolling 任务：它也使用线/角速度
w_slip = 0   # 对 rolling 任务：它也使用线/角速度
```

保留机身命令正则 `-0.3 mean(||[vx,vy,wz]_command||)`。
这是控制输入的正则项；关闭的是 jug 状态速度相关项。
成功判据中的速度阈值仍保留：线速度 <0.15 m/s、角速度 <0.6 rad/s、连续通过 ≥0.75 s。
它们只用于判断结果，不参与 MPC reward。滚动成功所需的历史累计量也保留。

测试验证：保持位姿、传感器与控制不变，把 jug 的六维速度随机改成大数，四个任务消融后的
reward 完全不变；`linear_velocity/angular_velocity/roll/slip` 分项均为零。
原任务的默认权重没有覆盖，原版本仍可用；这次实验通过独立开关与输出目录保存。

## 结果与视频

| 任务 | 无速度 reward 的结果 | 最终误差 | 视频 |
|---|---|---:|---|
| Upright | 6.34 s 成功，倾角 0.50°，连续合格 0.76 s | 1.20 cm | [MP4](../out/jug_tasks/no_velocity_20260915/spot_jug_upright_seed0.mp4) |
| Lay down | 2.34 s 成功，倾角 90.02°，连续合格 0.76 s | 9.28 cm | [MP4](../out/jug_tasks/no_velocity_20260915/spot_jug_lay_down_seed0.mp4) |
| Roll，A→B=1.2 m | 20 s 结束，桶已到位静止，**严格滚动判据未过** | 1.02 cm | [MP4](../out/jug_tasks/no_velocity_20260915/spot_jug_roll_seed0.mp4) |
| Move，A→B=3 m，2 s horizon | 28.80 s 成功，过程明显过冲后回推 | 1.50 cm | [MP4](../out/jug_tasks/no_velocity_20260915/spot_jug_move_seed0.mp4) |
| Move，A→B=3 m，4 s horizon | 5.74 s 成功，连续合格 0.76 s | 5.18 cm | [MP4](../out/jug_tasks/no_velocity_20260915/horizon4/spot_jug_move_seed0.mp4) |

[视频总览](../out/jug_tasks/no_velocity_20260915/index.html)。每个视频同目录都保留 JSON 参数/结果和 NPZ 原始轨迹。
上一轮带速度 reward 的 upright/lay down/roll 分别在 4.54/2.20/12.78 s 通过。
这里只是各一次仿真记录，不能据此推断成功率或把全部轨迹差异归因于单个因素；原生 rollout 有多线程与墙钟截止预算。

滚动失败的唯一剩余条件：

```text
有效滚动积分 = 0.499061967 m < 原阈值 0.500000000 m
轴向累计转动 = 12.6688 rad > π
最终到 B 误差 = 0.0102165 m
最终线速度/角速度 ≈ 0
最终倾角 = 90.5324°，桶高 = 0.139142 m
```

差约 0.94 mm。没有放宽阈值或把这个结果标为成功。
这也说明“移动到 B”和“通过人为设定的有效滚动积分门槛”是两个不同的验收目标。

## 新任务：`spot_jug_move`

初始桶直立，A=(0.95,0)，B=(3.95,0)，相距 **3.00 m**。
动作仍为三维机身速度，机械臂收起。没有 jug 朝向、高度、线速度、角速度、旋转耦合或滑动 reward。
允许推滑、放倒、滚动等方式移动物体，最终姿态不限。

```text
d = ||p_jug_xy - B||
e = (B - p_jug_xy) / max(d, 0.1)
d_push = ||p_robot_xy - (p_jug_xy - 0.65 e)||

R_move = mean(-100 d - 15 d_push - 0.3 ||u_base||)
         - 2500 × I(任一预测帧机器人高度 ≤ 0.35 m)
```

成功仍要求目标误差 <0.35 m、物体线速度 <0.15 m/s、角速度 <0.6 rad/s、机器人高度 >0.35 m，
持续至少 0.75 s；物体中心高在 0.08～0.32 m，允许直立或横躺等落地姿态，没有轴向旋转积分要求。

2 s horizon 首轮约 2.50 s 进入目标区，但带着较大速度继续前进，桶最大 x≈12.205 m；
机器人追回、再回推，最后于 28.80 s 成功。
后续只将 horizon 配置改成 4 s，reward 一字未改；这一轮约 4.10 s 进入目标区、5.74 s 成功，
最大 x≈3.995 m（B 的 x=3.95 m），明显减少了这一轮的过冲。
最终线速度 0.0427 m/s、角速度 0.3068 rad/s。

## 运行命令

在 `sumo/` 根目录：

```bash
MUJOCO_GL=egl pixi run python -m tools.jug_demo spot_jug_upright \
  --seed 0 --seconds 20 --no-velocity-reward --render --out out/jug_tasks/no_velocity_reproduce
MUJOCO_GL=egl pixi run python -m tools.jug_demo spot_jug_lay_down \
  --seed 0 --seconds 20 --no-velocity-reward --render --out out/jug_tasks/no_velocity_reproduce
MUJOCO_GL=egl pixi run python -m tools.jug_demo spot_jug_roll \
  --seed 0 --seconds 20 --no-velocity-reward --render --out out/jug_tasks/no_velocity_reproduce
MUJOCO_GL=egl pixi run python -m tools.jug_demo spot_jug_move \
  --seed 0 --seconds 30 --horizon 4 --no-velocity-reward --render --out out/jug_tasks/no_velocity_reproduce/horizon4
```

新移动任务本身默认 `w_linear_velocity=w_angular_velocity=0`，命令中的开关用于明确记录实验条件。
`--set 'gripper_target_local=[0,0,0.17]'` 可修改桶颈目标；本次没有改变其数值。
测试：29 个 jug 相关测试通过；Ruff 通过；修改代码的 Pyright 0 errors（6 个已有 MuJoCo 导出类型警告）。
