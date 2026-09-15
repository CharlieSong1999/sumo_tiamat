# Jug：10% 水球，固定 2 s MPC（2026-09-15）

这轮四个任务均保持 `horizon=2.0`，没有使用上一轮的 4 s 配置。
延续无 jug 速度 reward 设置，没有重新调整任务 reward 权重。

## 结果与视频

四项均使用 seed=0 成功一次，全过程 19/19 个水球留在桶内；不是成功率统计。
完成时间是仿真时间，包含连续通过判据的 0.76 s。

| 任务 | 完成时间 | 最终位置误差 | 最终倾角 | 视频 |
|---|---:|---:|---:|---|
| Upright | 12.64 s | 1.18 cm | 0.52° | [MP4](../out/jug_tasks/water10_h2_20260915/spot_jug_upright_seed0.mp4) |
| Lay down | 2.36 s | 9.84 cm | 89.81° | [MP4](../out/jug_tasks/water10_h2_20260915/spot_jug_lay_down_seed0.mp4) |
| Roll，A→B=1.2 m | 6.04 s | 0.60 cm | 89.70° | [MP4](../out/jug_tasks/water10_h2_20260915/spot_jug_roll_seed0.mp4) |
| Move，A→B=3.0 m | 7.64 s | 2.42 cm | 0.47° | [MP4](../out/jug_tasks/water10_h2_20260915/spot_jug_move_seed0.mp4) |

[视频总览](../out/jug_tasks/water10_h2_20260915/index.html)。
roll 的有效滚动累计 **1.28985 m**、轴向累计转动 **10.78497 rad**，通过原来的 0.5 m / π 判据。
3 m move 的 A=(0.95,0)、B=(3.95,0)；首次进入目标半径约 2.20 s，之后仍有调整，
直到 7.64 s 才持续稳定通过。桶最大 x=4.05538 m；过程最大倾角约 51.94°，最终恢复直立。
move 的水球局部质心三个坐标的全程变化范围分别为 5.32 / 5.24 / 2.41 cm，水并非固定配重。

运行用时约 upright 128 s、lay down 33 s、roll 58 s、move 64 s，不含视频渲染，且部分作业同时运行。
已记录预测末尾是否完全冻结的 upright/lay down/roll 三次均为 0；move 运行启动时尚未加该日志计数，
但同样已经关闭计算截止。四次原始轨迹均保留，视频显示 H=2s。

## 水球和碰撞模型

- 保留原 tracking mesh、坐标系、外部凸包碰撞、空桶 1.0 kg 质量及其惯量。
- 原外部凸包是实心，不能作为容纳水球的内壁。另加 16 边分段空腔，仅与水球碰撞。
  局部 `(z, 内切半径)` 轮廓为 `(-.224,.127), (.125,.127), (.175,.022), (.232,.022)` m，
  近似桶身、肩部及桶颈；底面和盖子封闭。空腔几何容积约 **19.032 L**。
- 水量为此容积的 10%，即 **1.903 L / 1.903 kg**。用 **19 个半径 2.5 cm** 的自由水球表示，
  每球约 0.10017 kg；桶加水总质量约 **2.903 kg**。没有再给桶刚体重复加水质量。
- 球的接触 `condim=1`，无切向接触摩擦；可以相互碰撞、与内壁交换动量、改变水的相对质心。
  碰撞位隔离确保它们不与原来的实心外凸包接触；地面仍能接住意外逸出的球。
- **按带盖桶处理。** 这是粗粒度水球/晃动近似，不是 CFD，不精确模拟自由液面、黏性或从细桶颈出水。
  水球直径甚至大于近似桶颈内径，因此不适合据此研究倒水。
- episode 开始前在固定桶内预沉降水球，随后速度清零。实际 MPC 执行过程中没有固定桶、外加物体力、
  改写物体轨迹或偷偷补水；只有机器人控制和 MuJoCo 动力学。
- 真实仿真和 MPC 都包含相同的 19 个球，不是“planner 空桶、plant 加水”。采用 CG / sparse、50 次求解迭代，
  物理步长仍为 0.01 s，每个低层控制步 2 个物理子步。

视频中的桶壳透明度仅用于显示内部水球，不改变动力学。
初始桶颈接近点仍是人工选取的 jug 局部坐标 `(0,0,0.17)` m，详见
[前一轮说明](jug_no_velocity.md#桶颈接近点是怎么来的)。

## 时域、计算预算与验收

每次规划 48 条候选、4 个节点、2 轮 CEM，seed=0；25 Hz 更新控制，**物理预测时域始终 2 s**。
加水后，离线 runner 关闭原生 rollout 的 125 ms 墙钟计算截止，避免计算较慢时将剩余预测状态冻结。
这只增加允许花费的计算时间，不改变预测时域；这些运行不宣称满足硬实时。
同时改变了接触求解器及计算截止，因此不能把与上一轮空桶轨迹的差异全部归因于水。

成功判据不放宽：位置误差 <0.35 m，jug 线速度 <0.15 m/s、角速度 <0.6 rad/s，
机器人不倒，连续合格至少 0.75 s（离散计数实际 0.76 s）。
upright 要求距直立 <15°；放倒/roll 要求距水平 <15°，并检查落地高度。
move 不限制最终朝向。roll 还要求有效向前滚动累计 ≥0.5 m、轴向累计转动 ≥π。
水球保持率是报告指标，不是 reward。只要发生泄漏，即使动作指标通过也不应把该水模型视为合格。

## 每个任务实际 reward

`mean` 对整个 2 s 预测序列取平均。定义：

```text
c = jug 局部 +z 轴在世界竖直方向的分量
d = ||p_jug_xy - goal_xy||
e = (goal_xy - p_jug_xy) / max(d, 0.1)
d_push = ||p_robot_xy - (p_jug_xy - 0.65 e)||
d_neck = ||finger_in_jug_frame - (0,0,0.17)||
d_body = ||p_robot_xy - p_jug_xy||
h = jug 原点的世界高度
U = ||[vx,vy,wz]_base_command||
F = 2500 × I(任一预测帧机器人高度 ≤ 0.35 m)

Upright:
R = mean[-150(1-c) -15d -12d_neck
         -100|h-0.2415|clip(c,0,1) -0.3U] - F

Lay down:
R = mean[-150|c| -80d -30d_body|c|
         -100max(h-0.17,0) -0.3U] - F

Roll (A→B = 1.2 m):
R = mean[-40|c| -100d -15d_push
         -100max(h-0.17,0) -0.3U] - F

Move (A→B = 3.0 m):
R = mean[-100d -15d_push -0.3U] - F
```

`w_linear_velocity=w_angular_velocity=0`，roll 的 `w_roll=w_slip=0`。
**没有 jug 或水球线速度/角速度 reward**；保留与上一轮一致的机身命令正则 `-0.3U`。
成功判据中的速度阈值和 roll 的运动历史指标不参与 MPC reward。
继承配置中的 `w_goal=60` 并未被这些任务的 reward 函数引用，不是隐藏附加项。

## 重现

在 `sumo/` 根目录运行；更换 task 名可运行另外三个任务：

```bash
pixi run build
pixi run python tools/jug_demo.py spot_jug_move \
  --seed 0 --water-fill 0.1 --no-velocity-reward \
  --horizon 2 --seconds 40 --render \
  --out out/jug_tasks/water10_h2_reproduce
```

任务名：`spot_jug_upright`、`spot_jug_lay_down`、`spot_jug_roll`、`spot_jug_move`。
`water_fill_ratio`、`water_ball_radius` 是**构造时**参数，改水量需重建任务模型，不能只修改运行中的 config。
每个结果保留 MP4、JSON 参数/结果、NPZ（全部 qpos/qvel，包括水球，以及 action/metrics）。
修改代码通过 Ruff；Pyright 0 errors，16 个 MuJoCo 导出类型提示；全量测试 **284 passed**。
