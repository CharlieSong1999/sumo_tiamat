# Jug manipulation：SUMO 闭环验证（2026-09-15）

后续实验：[关闭速度 reward、桶颈目标说明，以及 3 m 纯位置移动](jug_no_velocity.md)。

已阅读 `tiamat_dev_local` 的机器人接口与 `sumo_server`、`auto_sumo/docs` 的感知/真机会话、
`bag2sim` 的公制网格与跟踪流程，以及 SUMO 的任务、MPC 和物理后端。
用户提到的 `spot_kick_off_jug` 在代码中对应 `spot_jug_kick`。
新增三个任务复用它的 `spot_jug_kick.xml`、`jug_joint`、传感器、网格、质量和碰撞参数。
没有修改真机控制链，也没有修改旧 kick 任务。

资产为原有 1 kg 近空 5-gallon jug，约 0.281 m 直径、0.483 m 高；原点与
FoundationPose++ 跟踪网格一致，局部 z 轴沿桶的长轴。

## 已验证的运行与视频

均为 CEM MPC → 原有 ONNX locomotion policy → MuJoCo 接触动力学闭环。
对象仅在 reset 时设置位姿，初速度为零；执行时未施加外力、未修改对象位姿、未用焊接或 mocap 代替接触。
视频是保存的仿真轨迹渲染，25 fps、960×640、H.264。标记和跟随镜头仅影响显示。

| 任务 | 初始条件与目标（世界坐标，米） | 通过时刻 | 最后状态 | 视频 |
|---|---|---:|---|---|
| `spot_jug_upright` | A=(0.95,0)，桶横躺，桶口朝 Spot；目标 xy=(0.95,0) | 4.54 s | 倾角 0.49°，位置误差 0.317 m，速度 0.0051 m/s | [upright](../out/jug_tasks/20260915/spot_jug_upright_seed0.mp4) |
| `spot_jug_lay_down` | A=(0.95,0)，桶直立；落地点目标 xy=(1.33,0) | 2.20 s | 倾角 89.89°，位置误差 0.104 m，速度 0.0176 m/s | [lay down](../out/jug_tasks/20260915/spot_jug_lay_down_seed0.mp4) |
| `spot_jug_roll` | A=(0.95,0) → B=(2.15,0)，相距 1.20 m；桶横躺，长轴垂直于路线 | 12.78 s | 倾角 89.69°，位置误差 0.0304 m，速度 0.0105 m/s | [roll A→B](../out/jug_tasks/20260915/spot_jug_roll_seed0.mp4) |

[视频总览](../out/jug_tasks/20260915/index.html)。同目录的 JSON 保存参数和结果，NPZ 保存 qpos、qvel、
动作及逐帧指标。三个成功样例的随机种子均为 0；这是一轮成功验证，不是多初始状态成功率评估。
滚动样例曾越过 B（最大 x≈3.73 m），随后回推并停稳；它尚不是直接停到 B 的平顺轨迹。
扶正最终只有桶与地面的接触，没有机械臂托住桶。

放倒目标的 0.38 m 前移来自绕前方底缘翻倒的几何位移：r+h/2≈0.14+0.2415 m。
初版把目标仍放在原桶心，虽然已放倒停稳，却会因桶心正常平移而失败；最终保留位置容差，修正了目标中心。

## 动作空间

- Upright：`use_arm=True, use_gripper=True`，11 维（机身速度 3 + 臂/夹爪命令 7 + 夹爪开闭选择 1）。
- Lay down / roll：`use_arm=False`，3 维机身速度 `[vx, vy, wz]`，机械臂收起。

## Reward 的精确定义

所有 `mean(...)` 都是对一条候选轨迹的时间维取平均，输出每条轨迹一个标量，越大越好。
所有距离/速度使用米与秒，角速度使用 rad/s。

符号：

- `p=(px,py,h)`：jug 原点位置；`b`：Spot 机身 xy；`g`：配置 `goal_pos`。
- `a`：jug 局部 z 轴在世界坐标的单位向量；`c=clip(a_z,-1,1)`。
  直立为 1，横躺为 0，倒立为 -1。
- `d=||p_xy-g_xy||`；`v` 为 jug 线速度；`ω` 为自由关节的**局部坐标**角速度。
- `u=[vx,vy,wz]` 为机身命令；只惩罚这三维，不把机械臂关节目标角当作力矩惩罚。
- `n=exp(-(d/0.35)^2)` 为接近目标程度。
- `F=2500 × I(轨迹中任一帧 Spot 机身高度≤0.35)`：整条轨迹扣一次摔倒惩罚。
- `q_up=I(c>cos(15°))`；`q_side=I(|c|<sin(15°))`。
- 继承的 `w_goal=60` 字段未参与新任务 reward；物体目标距离用 `w_position`。

### 1. Upright

`f` 为夹爪指尖在 jug 坐标系中的位置，`d_grip=||f-(0,0,0.17)||`，引导机械臂接近桶颈。

```text
R_up = mean(
    -150 × (1-c)                         # 扶正长轴，倒立比横躺更差
    - 15 × d                             # 桶心靠近目标 xy
    - 12 × d_grip                        # 指尖接近桶颈
    -100 × |h-0.2415| × clip(c,0,1)       # 扶正后落回地面高度
    -  8 × n × q_up × ||v||              # 目标附近直立时停稳
    -  2 × n × q_up × ||ω||
    -0.3 × ||u||
) - F
```

### 2. Lay down

```text
R_down = mean(
    -150 × |c|                           # 长轴接近水平；倒立不算横躺
    - 80 × d                             # 桶心靠近预期落地点
    - 30 × ||b-p_xy|| × |c|              # 尚未放倒时，机身靠近桶
    -100 × max(h-0.17,0)                 # 避免把桶打飞
    - 25 × ||v||                         # 全程抑制过大的撞击/滑动速度
    -  5 × ||ω||
    -0.3 × ||u||
) - F
```

### 3. Roll A→B

`e=(g_xy-p_xy)/max(d,0.1)`；`d_push=||b-(p_xy-0.65e)||`：让 Spot 处于桶背离目标的一侧。
`t=(B_xy-A_xy)/||B_xy-A_xy||` 为固定路线方向。

```text
v_roll = 0.14 × ω_local_z × (a × world_Z)
s_move = v_xy · t
s_spin = v_roll_xy · t
q_ground = I(h<0.20 and |c|<0.3)
ρ = q_ground × clip(min(s_move,s_spin),0,1)
slip = ||v_xy-v_roll_xy||

R_roll = mean(
    - 40 × |c|                           # 桶保持横躺
    -100 × d                             # 桶心到 B
    - 15 × d_push                        # Spot 从后方推动
    -100 × max(h-0.17,0)                 # 保持贴地
    + 25 × (1-n) × ρ                     # 只奖励平移与轴转动同时朝 B；近 B 衰减
    -  4 × slip                          # 惩罚滑动/空转
    -  8 × n × q_side × ||v||            # 靠近 B 时停稳
    -  2 × n × q_side × ||ω||
    -0.3 × ||u||
) - F
```

只有平移、只有原地自转或在空中翻转时，`ρ=0`。该项允许真实接触中的部分滑动，
不会把无旋转的推滑算作有效滚动。桶的非理想圆柱外形用现有网格凸包处理，0.14 m 是奖励中采用的近似半径。

## 成功判据

三者都要求 Spot 机身高于 0.35 m、桶心目标 xy 误差小于 0.35 m、jug 线速度小于 0.15 m/s、
角速度小于 0.6 rad/s。运行器要求这些条件**连续满足至少 0.75 s**，本轮均记录为 0.76 s。

- Upright：长轴与世界竖直夹角 <15°；桶心高度距 0.2415 m 小于 0.04 m。
- Lay down：夹角在 75°～105°，桶心高度在 0.08～0.20 m。
- Roll：同 lay down 的姿态/高度；再要求本次 episode 的 `∫ρ dt≥0.50 m`，且
  `∫|ω_local_z|dt≥π rad`。滑动到 B 或一开始就在 B 不满足滚动历史要求。

滚动样例累计 `∫ρ dt=3.286 m`、轴转动 `44.94 rad`；这是含越界回推过程的累计量，
不是 A 到 B 的净位移（净目标距离为 1.20 m）。

## 复现

在 `sumo/` 根目录执行：

```bash
pixi run build
MUJOCO_GL=egl pixi run python -m tools.jug_demo spot_jug_upright --seed 0 --render --out out/jug_tasks/reproduce
MUJOCO_GL=egl pixi run python -m tools.jug_demo spot_jug_lay_down --seed 0 --render --out out/jug_tasks/reproduce
MUJOCO_GL=egl pixi run python -m tools.jug_demo spot_jug_roll --seed 0 --render --out out/jug_tasks/reproduce
```

运行器默认 2 s horizon、48 条 rollout、4 个线性样条节点、4 elites、每次重规划 2 轮 CEM，
25 Hz 重规划、50 Hz 策略控制，MuJoCo 每个策略步走 2×0.01 s。
每轮优化都用实际执行器的上一条神经网络输出初始化候选的递归状态。
`--set field=JSON` 可以覆盖任务参数；所有生效值会保存在结果 JSON。
物理后端为多线程且有计算截止时间，保存轨迹用于逐帧精确重放。

只重渲染已保存轨迹：

```bash
MUJOCO_GL=egl pixi run python -m tools.jug_demo spot_jug_upright \
  --replay out/jug_tasks/20260915/spot_jug_upright_seed0.npz --out out/jug_tasks/replay
```

交互式任务入口也已注册：`pixi run sumo task=spot_jug_upright`（另两个任务同理）。
交互界面沿用通用 Spot 优化器默认值；上面的演示脚本明确指定本次成功运行的采样预算。

代码：`sumo/tasks/spot/spot_jug_manipulation.py`；运行/录制：`tools/jug_demo.py`；
测试：`tests/test_jug_manipulation.py`。验证包含注册、reset、动作维度、reward batch 维度、
摔倒惩罚、倒立/悬空/过快拒绝、滑动/原地转动拒绝、滚动历史与 reset 清零。
