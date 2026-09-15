# Jug upright：SUMO 论文参数对照（2026-09-16）

## 核对的原始来源

通过 huggingface-papers 技能查找原文；Hugging Face 页面不可读，按技能回退到 arXiv。
[SUMO 论文 IV-A](https://arxiv.org/html/2604.08508v1#S4.SS1) 明确给出：

| 参数 | 论文 | 上轮抓取实验 |
|---|---:|---:|
| 预测时域 | 1.5 s | 2 s |
| 每批候选路径 | 32 | 24 |
| elite | 3 | 2 |
| 控制节点 | 4 | 4 |
| 高层规划频率 | 20 Hz | 25 Hz |
| 低层策略频率 | 50 Hz | 50 Hz |
| 噪声 | 沿时域线性增加的 variance .02→.6 | elite 拟合 sigma，再乘时间 ramp 并限幅 |

论文并未明确给出每个更新时刻的内层 CEM 重复次数；本次保持 1 轮。
每次规划两者均需 4,800 个候选物理步：32×150 与 24×200。
20 Hz 对应每模拟秒约 96,000 步，旧 25 Hz 为 120,000 步；这不是墙钟耗时保证。

公开发布代码也不能直接当作论文配置：[optimizer defaults](https://raw.githubusercontent.com/rai-opensource/sumo/main/sumo/controller/optimizer_overrides.py)
是 24 条 / 3 节点 / noise_ramp=3.5，
[controller defaults](https://raw.githubusercontent.com/rai-opensource/sumo/main/sumo/controller/overrides.py)
是 Spot H=2 s。此次以论文文字为参数依据，并明确记录实现假设。

## 实验边界与复现差异

- 只测试规划设置，不改当前 `neck_grasp=true, neck_grasp_fade_width=1` 的 reward、动作空间或抓取几何。
- 保持约 10% 水：19 球、1.903 kg；planner 与 plant 使用同样的粒子。
- jug 线/角速度 reward 为 0，原有成功条件与连续 .75 s 验收不变；每个 seed 最长 25 s。
  这不是论文中其他物体、其他验收条件的 benchmark 复现。
- 上轮物理：jug_mass=1、rolling_friction=.0001、jug_contact_dim=3、w_look_object=0。
- 当前 HEAD `adb142f` 又加入了 yaw 命令下限。为了和上轮比较，仅在这些离线实验中
  指定 yaw_rate_min=0、yaw_rate_deadzone=0；不改部署默认值。
- 完整积分请求的 1.5 s 时域，关闭离线预测的墙钟截断；报告 frozen_prediction_tails。
- 当前 backend 的 plant 一次步进 .02 s，不改变其 50 Hz 策略与 .01 s 物理步。
  20 Hz 规划落在 .00/.06/.10/.16/.20… 的时刻，即交替 60/40 ms，平均 20 Hz，
  不是严格每 50 ms 的独立时钟。NPZ 保存实际 `planning_sim_times`。
- 论文未交代方差调度与 CEM 拟合方差的组合。为避免混淆，分别测试：
  `adaptive`：原 Judo 自适应 sigma；
  `paper-variance`：按文字在四个节点固定 variance=[.02,.213333,.406667,.6]，
  标准差为其平方根，每次仍用 3 个 elite 更新均值，但下一轮重新使用固定协方差。
  后者是明确的实现解释，不是作者确认的原始优化器。
- 保留当前 11 维抓取控制，不改成论文分析实验的默认闭合夹爪，也不引入脚本夹取、焊接或外力。

## 运行命令

```bash
pixi run build
pixi run python tools/jug_demo.py spot_jug_upright \
  --seed 0 --rollouts 32 --iterations 1 --elites 3 --nodes 4 \
  --horizon 1.5 --control-freq 20 --noise-profile paper-variance \
  --water-fill .1 --no-velocity-reward --seconds 25 \
  --set neck_grasp=true --set jug_mass=1.0 --set rolling_friction=0.0001 \
  --set w_look_object=0 --set yaw_rate_min=0 --set yaw_rate_deadzone=0 \
  --jug-contact-dim 3 --render \
  --out out/jug_tasks/upright_paper_20260916/paper_variance20hz
```

`--noise-profile adaptive` 是保留当前 CEM 噪声的对照，输出到另一个目录避免覆盖。
`budget_only25hz` 额外只改变时域、候选数和 elite，保留 25 Hz。
所有参数与状态记录在各自 JSON / NPZ 中；视频中的时间是仿真时间，不是墙钟实时。

## 结果

全部使用同一抓取 reward、同一初始状态，成功条件未放宽。每组 seed=0/1/2；
额外 25 Hz 对照只跑 seed=0。仅以三次试验观察结果，不估计可靠成功率。

| 配置 | seed | 通过连续验收 | 结束时间 | 最终倾角 | 目标误差 | 夹持代理累计 |
|---|---:|---|---:|---:|---:|---:|
| 1.5 s / 32×1 / 3 elite / 20 Hz，原 adaptive 噪声 | 0 | 是 | 21.44 s | .27° | 2.31 cm | .18 s |
| 同上 | 1 | 否 | 25 s | 89.71° | 60.06 cm | 0 |
| 同上 | 2 | 否 | 25 s | 89.77° | 77.44 cm | 0 |
| 同预算 / 20 Hz，固定 paper-variance 解释 | 0 | 否 | 25 s | 90.16° | 144.31 cm | 0 |
| 同上 | 1 | 否 | 25 s | 90.37° | 104.32 cm | 0 |
| 同上 | 2 | 否 | 25 s | 89.70° | 110.91 cm | 0 |
| 同预算 / 原 adaptive 噪声，保留 25 Hz | 0 | 否，已立起但未连续停稳 | 25 s | .95° | 2.29 cm | .96 s |

上轮 2 s / 24×1 / 2 elite / 25 Hz 的同一抓取 reward 为 1/3 seed 成功，
seed=0 于 8.08 s 完成，位置误差 14.03 cm。
本次 adaptive 配置也是 1/3：有可行样例，但不能认定更稳定；固定方差解释没有成功样例。
不能据此否定论文结论，因为对象、reward、控制空间及噪声实现并非原 benchmark。

### 成功 upright 不等于成功抓取

新 adaptive seed=0 首次达到倾角<15° 是 9.92 s，直到 21.44 s 才通过连续 .76 s 验收。
最终线速度 .0285 m/s、角速度 .0684 rad/s，满足当前 jug 的原验收阈值。
其宽松夹持代理累计 .18 s，但 `jug_grasp_audit.py` 在立起前没有检测到同时满足
双侧 dist≤0、法向相对、闭合受阻的采样帧。因此只能叫 **upright 成功**，不能叫已验证的瓶口抓举成功。
额外 25 Hz 对照的严格立起前夹持复核同样为 0；最长连续验收时间只有 .50 s，没有达到 .75 s。

### 视频

- [adaptive / seed=0：upright 成功](../out/jug_tasks/upright_paper_20260916/adaptive20hz/spot_jug_upright_seed0.mp4)
- [同一轨迹瓶口特写](../out/jug_tasks/upright_paper_20260916/adaptive20hz/closeup_seed0/spot_jug_upright_seed0.mp4)
- [adaptive / seed=1：失败](../out/jug_tasks/upright_paper_20260916/adaptive20hz/spot_jug_upright_seed1.mp4)
- [adaptive / seed=2：失败](../out/jug_tasks/upright_paper_20260916/adaptive20hz/spot_jug_upright_seed2.mp4)
- [固定方差 / seed=0：失败](../out/jug_tasks/upright_paper_20260916/paper_variance20hz/spot_jug_upright_seed0.mp4)
- [固定方差 / seed=1：失败](../out/jug_tasks/upright_paper_20260916/paper_variance20hz/spot_jug_upright_seed1.mp4)
- [固定方差 / seed=2：失败](../out/jug_tasks/upright_paper_20260916/paper_variance20hz/spot_jug_upright_seed2.mp4)
- [额外 25 Hz 对照：立起但未停稳](../out/jug_tasks/upright_paper_20260916/budget_only25hz/spot_jug_upright_seed0.mp4)

### 计算与验证

全部实验水球保留率 100%，frozen_prediction_tails=0；没有新增根目录 MuJoCo 数值告警。
adaptive 三个 seed 每次规划平均 108.0 / 97.5 / 101.9 ms；固定方差约 92.6～93.6 ms。
这些是本机粗测，部分回放/分析与实验并行，不能当作严格的硬件 benchmark。
无一次规划满足 50 ms 墙钟预算，因此 **仍不具备已验证的 20 Hz 墙钟实时能力**。
不能把论文其他模型上的时延直接套用到含 19 个独立水球的当前模型。

新增选项只存在于离线 runner，默认仍为原 adaptive / 25 Hz；没有修改部署参数或抓取 reward。
`pixi run pytest -rsx`：319 passed；runner 与测试 Ruff 检查、格式检查通过；
Pyright 0 error，6 条 MuJoCo Python 导出类型警告。
