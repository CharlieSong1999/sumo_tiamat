# Upright：24 rollouts × 1 CEM 与瓶口抓取（2026-09-16）

## 一轮 CEM 的确切含义

这里的“一轮”不是一整次 episode，也不是一帧物理仿真，而是一次采样分布更新：

1. 把上一规划时刻的动作样条前移，作为当前 nominal / warm start。
2. 在控制节点上采样候选动作序列。24 条里有 1 条原 nominal、23 条带高斯噪声的候选。
3. 每条都从相同当前状态预测未来 **2 s**，计算整个预测序列的平均 reward。
4. 按 reward 排序，用 top-k elite 的节点均值更新 nominal、方差更新下一次采样的 sigma。
5. 一轮配置此时就结束，执行更新后动作样条的当前部分；不是直接执行某一条最优候选。

两轮配置会在第 4 步后，使用新均值和方差，再采样、预测、选优一次，才执行动作。
**两轮之间不推进真实仿真状态；也不是把物理预测时域接成 4 s。**
下一 MPC 时刻仍继续使用上一时刻结果热启动，因此 24×1 并非整个任务只探索 24 条轨迹。

本 runner 将 `ControllerConfig.max_opt_iters` 固定为 1，外层 `--iterations` 控制同一状态上
调用 `controller.update_action()` 的次数。JSON 里的 `iterations=2` 才表示本实验的有效两轮，
不能只读 `controller.max_opt_iters=1` 就认为之前是单轮。

当前 low-level dt=0.02 s，MuJoCo dt=0.01 s，每条候选有 100 个低层命令 / 200 个物理步：

| 配置 | 每次规划候选评估 | 每次规划候选物理步 |
|---|---:|---:|
| 48×2 | 96 | 19,200 |
| 24×1 | 24 | 4,800 |

预测工作量约减少 75%；墙钟耗时还取决于线程并行、接触、模型和推理开销，不能直接保证 4 倍加速。

## 可比实验条件

始终保持 H=2 s、10% 水球（19 球 / 1.903 kg）、无 jug 线/角速度 reward、4 个样条节点、
sigma_min=.12、sigma_max=1、noise_ramp=2、原来的验收标准。

仓库 HEAD=16c8d2d 已加入真实机器人部署默认值（jug_mass=1.5、rolling_friction=.01、look reward=10，
以及 condim=6）。为与上次加水 demo 比较，**仅在本实验覆写**：

```text
jug_mass=1.0
rolling_friction=0.0001
w_look_object=0
jug_contact_dim=3
```

没有覆盖仓库新的部署默认值。加水粒子与内部空腔依旧同时出现在 planner / plant，
离线关闭 125 ms 截断，保证完整积分 2 s。这里的 25 Hz 是模拟时间中的重规划节奏，不代表墙钟实时。

## 不改 reward 的 24×1 结果

直接从 48×2 改为 24×1、仍保留 4 elite：seed=0 在 30 s 内没立起，最终倾角约 89.84°。
随后仅将 elite 改为 **2**，保持 top-k 比例：`4/48 = 2/24`，reward 不变。

| 配置 | seed | 结果 | 仿真结束时间 | 最终位置误差 | 每次规划平均耗时 |
|---|---:|---|---:|---:|---:|
| 48×2，4 elite | 0 | 成功，重现上轮轨迹 | 12.64 s | 1.18 cm | 355.4 ms |
| 24×1，2 elite | 0 | 成功 | 7.26 s | 34.88 cm | 94.7 ms |
| 24×1，2 elite | 1 | 未通过连续停稳验收；桶已立起 | 30.00 s | 1.14 cm | 93.3 ms |
| 24×1，2 elite | 2 | 成功 | 9.64 s | 2.81 cm | 98.3 ms |

只测了 3 个 seed，2 个成功，不能视为可靠成功率估计。seed=0 只是刚过 35 cm 的位置门槛。
这证明 48×2 并非完成 upright 的必要条件，但降低预算后不应假定成功率、精度和轨迹质量不变。

[seed=2 成功视频](../out/jug_tasks/upright_24x1_20260916/elites2_verified/spot_jug_upright_seed2.mp4)。
该视频不要求真实夹住瓶口，只验证原 upright 验收标准。

## 参考 upright / drag 后的发现

- `spot_barrel_upright` 主要是朝向 + 接近距离，并没有要求夹住。
- `spot_bucket_drag` / `spot_barrel_drag_v1` 有夹爪朝向、闭合阻力、空抓惩罚的设计思路，
  但其中的阻力符号不能直接用于当前模型：当前 arm_f1x 为 **0 闭合、负值张开**，阻挡闭合时
  应检查 `q_command - q_actual > 0`。`spot_pit_move` 已使用这一方向。
- 第 10 号（零基）高层控制维是夹爪选择通道；它小于 0 时会把实际闭合命令覆盖为 0。
  奖励必须用覆盖后的有效命令，而不是直接读取 controls[...,9]。
- `site_arm_link_fngr` 实际是活动指的铰链位置，不是夹取中心。
- 原 jug 单个凸包将细瓶颈补成斜面：局部 z=.20 m，视觉半径中位数约 **2.45 cm**，
  凸包截面半径约 **6.26～6.67 cm**。在这个模型上只奖励“抓住瓶口”会面对错误的碰撞形状。

## 可选 neck_grasp 实验

新增 `neck_grasp=true`，仅对 upright 生效，默认 **false**。保留原视觉网格、质量、惯量和坐标系；
将外部碰撞拆为 z≤.17 的桶身凸包 + 半径 .029 m 的瓶颈柱体。水球仍只与独立内壁碰撞。
**因此抓取版与旧版不只是 reward 消融，碰撞几何也修正了。** 没有焊接、抓取约束或增大夹爪驱动力。

夹取中心设为 wrist 局部 `(.185,0,-.008)` m，目标为 jug 局部 `(0,0,.205)` m。
新增四个 jaw-to-neck 几何距离传感器，分别覆盖活动指两侧与固定下颌。

有效夹持代理条件同时要求：

```text
夹取中心距瓶口目标 < .065 m
有效命令 > -.25 rad（要求闭合）
q_command - q_actual > .12 rad（闭合受到阻挡）
q_actual < -.05 rad（没有空闭合到底）
活动指与固定下颌到 neck geom 的距离均 < .002 m
```

这是包含 2 mm 容差的代理判据，不等于严格接触力检测，也不等于证明全程稳定悬空抓举。
严格复核可从 NPZ 的位姿重建实际接触对，要求两侧 contact.dist≤0 且接触法向相对。

原 upright 的朝向、目标位置、高度、机身命令正则、摔倒惩罚保留；用夹取中心引导替换原铰链接近项。
共同基础项为：

```text
R_base = -150 mean(1-c) - 15 mean(||jug_xy-goal_xy||)
         -100 mean(abs(jug_z-.2415) * clip(c,0,1))
         -.3 mean(||u_base||) -2500 any(robot_z<=.35)
```

本实验 look、jug 线速度、jug 角速度项权重均为 0。成功仍检查速度是否低于原验收阈值，
并连续满足 .75 s；这是验收条件，不是 reward。
当前可选抓取版的额外项为：

```text
c = 桶长轴的世界竖直分量
d = 夹取中心到 neck target 的距离
n = exp(-(d/.10)^2)
s = clip((c-.85)/.13, 0, 1)           # 立起后的释放阶段
target_reach = (0,0,.205+.10*s)      # 沿瓶口轴撤离 10 cm

-60 mean(||pinch_local - target_reach||)
-20 mean((1 + dot(wrist_x, jug_z)) * exp(-(d/.4)^2))
- 5 mean((1 - n*(1-s)) * (q_actual+1.1)^2)
+40 mean(grasp_proxy * clip(resistance/.5,0,1) * clip((1-c)/fade_width,0,1))
-15 mean(n * closing * empty_close * not_bilateral * clip((1-c)/.25,0,1))
```

`w_neck_straddle=w_neck_close=0` 为当前选定组合；这两个可选连续项曾试过 80 / 5，但没有改善 seed=0。
所有上述项均不使用 jug 或机器人状态速度；原机身命令正则仍保留。

### 已观察到的效果与局限

最初不撤离的抓取版本在约 4 s 把桶立起，但持续干扰桶，25 s 未过连续停稳验收。
在立桶前有 24 个采样帧（.48 s）通过宽松夹持代理；严格复核其中 **5 帧（.10 s）**
确有活动指/固定下颌同时接触瓶颈且法向相对，首次 3.04 s，最小法向点积约 -.94。
所以有真实短暂夹持辅助翻正，但不应把 .48 s 全都说成严格力闭合。

加入撤离、fade_width=.25 的版本，seed=0 在 **10.18 s 成功**，倾角 **.50°**，位置误差 **11.78 cm**，
平均规划 **101.4 ms**，全程水球保持 100%。
[视频](../out/jug_tasks/upright_24x1_20260916/neck_grasp_release_only/spot_jug_upright_seed0.mp4)。
这是一条成功样例，不是稳定方案：seed=1 会夹住后停在约 25～40° 的半直立状态，seed=2 也未完成。

分析发现 fade_width=.25 时，40 分的夹持奖励在最后 .25 的 cosine 区间消失，最大斜率为 160，
大于立正奖励的 150，反而可能奖励“夹住但不完全立正”。因此当前将 fade_width 改成 **1.0**，
将奖励衰减分摊到整个翻正过程，避免这个明确的目标冲突；配有单调性回归测试。

### 最终 fade_width=1 的验证

| seed | 结果 | 结束时间 | 最终倾角 | 位置误差 |
|---|---|---:|---:|---:|
| 0 | 成功，连续满足验收 .76 s | 8.08 s | 1.42° | 14.03 cm |
| 1 | 已立起，但偏离目标 | 25 s | .79° | 71.73 cm |
| 2 | 未立起，瓶口接近阶段把桶推走 | 25 s | 90.22° | 99.38 cm |

seed=0 的平均规划时间 101.7 ms，p95=111.5 ms；全程 19 个水球保留，预测末端没有冻结。
[完整成功视频](../out/jug_tasks/upright_24x1_20260916/neck_fade_seed0/spot_jug_upright_seed0.mp4)，
[瓶口特写](../out/jug_tasks/upright_24x1_20260916/neck_fade_seed0/closeup/spot_jug_upright_seed0.mp4)。

对这个 **8.08 s 的实际轨迹** 用 `tools/jug_grasp_audit.py` 复核：立起前有 7 个采样帧，
合计约 .14 s，满足双侧真实几何接触（dist≤0）、法向相对（dot<-.5）且有闭合阻力；
最长连续 5 帧（约 .10 s），首次 3.04 s，最小法向点积 -.960。
宽松夹持代理累计 .50 s。这里的“真实接触”仍不是接触力闭合检测，也不证明持续悬空抓举。
现有 upright 成功判据 **不要求抓取历史**，因此不能把仅仅完成 upright 当成抓取成功。

额外试过最终版本 `w_position=60`：seed=0 在 25 s 内未立起，最终 90.56°、位置误差 7.84 cm，
没有双侧夹持。过强位置项使本次搜索停在接近瓶口阶段，因此未采用。

结论：24×1 原奖励配置（2 elite）有 2/3 seed 成功，当前抓取配置有 1/3 seed 成功；
这些小样本只能说明可行性，**尚无证据表明抓取 shaping 已提高低预算稳定性**。
本次不再继续盲目加权。后续应单独验证“对准→闭合→翻正→落地释放”的阶段引导，
并固定种子集做对照；夹爪两种控制通道的冗余搜索也是可研究方向，尚未实现。

### 复现命令

```bash
pixi run build
pixi run python tools/jug_demo.py spot_jug_upright \
  --rollouts 24 --iterations 1 --elites 2 --horizon 2 \
  --water-fill .1 --no-velocity-reward --seconds 25 \
  --set jug_mass=1.0 --set rolling_friction=0.0001 --set w_look_object=0 \
  --jug-contact-dim 3 --set neck_grasp=true \
  --render --out out/jug_tasks/upright_24x1_reproduce
```

默认 seed=0、fade_width=1.0 对应最终 8.08 s 样例。
重现上面 10.18 s 的中间版本时，额外指定 `--set neck_grasp_fade_width=0.25`。
要测不改 reward 的 24×1 方案，去掉 `--set neck_grasp=true`，可用 `--seed 2` 对照上述视频。

```bash
# 只读复核已存轨迹的双侧瓶颈接触
pixi run python tools/jug_grasp_audit.py \
  out/jug_tasks/upright_24x1_20260916/neck_fade_seed0/spot_jug_upright_seed0.npz
# 只改变回放镜头，不重跑物理
pixi run python tools/jug_demo.py spot_jug_upright \
  --replay out/jug_tasks/upright_24x1_20260916/neck_fade_seed0/spot_jug_upright_seed0.npz \
  --view neck --out out/jug_tasks/upright_24x1_20260916/neck_fade_seed0/closeup
```

JSON 保存参数、结果、规划耗时均值/p50/p95/max；NPZ 保存完整物体/机器人/水球状态、动作、
每次规划和每轮 CEM 的耗时。四 elite 初始探索在写 JSON 时遇到新统计字段的 NumPy 整数序列化问题，
只有完整 progress 日志，没有 NPZ；已修复并加入实际短仿真保存回归测试，不用于耗时统计。

参考 48×2 平均约 355 ms；24×1 成功样例为约 95～102 ms，**仍不能在当前机器保证 25 Hz / 40 ms 硬实时**。
此外，接触模型和抓取阶段本身还需要更多种子及真实参数验证，当前不修改真实机器人的部署配置。

完整测试 `pixi run pytest -rsx`：310 passed；修改文件 Ruff 检查及格式检查通过，
Pyright 为 0 error、19 条 MuJoCo Python 导出类型警告。中间探索曾产生一条 MuJoCo QACC 不稳定告警，
保存在 `out/jug_tasks/upright_24x1_20260916/exploratory_MUJOCO_LOG.TXT`；
无法从原日志确定具体候选，不能当作没有发生。最终 8.08 s 成功运行未新增告警。
