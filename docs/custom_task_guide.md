# 自定义任务编写指南 —— 以 `spot_barrel_push` 为例

本文以"让 Spot 推动一个水桶到目标点"这个任务为例，逐行讲解构成一个任务所需的四个文件，
并提炼出可复用到任何新物体 / 新任务的通用规则。

读完你应该能独立回答：

- 一个任务由哪些文件组成，各管什么？
- MuJoCo 的 `body` / `joint` / `geom` / `site` / `inertial` / `sensor` / `default` 分别是什么？
- XML 和 Python 之间靠什么对接？
- 物体的位姿、速度在 Python 里如何按下标取出？
- 想换一个新物体 / 改奖励，要动哪几行？

---

## 0. 全局结构与数据流

一个任务 = **三层资产 + 一层逻辑**：

```
meshes/<obj>/*.obj          ← ① 几何形状（纯数据，无语义）
        │  被引用
        ▼
xml/objects/<obj>/<obj>.xml ← ② 物体：把 mesh 包成有质量/关节/碰撞/标记点的刚体
        │  被 include
        ▼
xml/spot_tasks/<task>.xml   ← ③ 场景：机器人 + 物体 + 地面 + 传感器，拼成完整物理世界
        │  model_path 指向它
        ▼
tasks/spot/<task>.py        ← ④ 逻辑：动作空间、奖励、初始姿态、成功判据
        │  register_task 注册
        ▼
tasks/__init__.py           ← 注册表，把名字映射到 ④
```

运行时的数据流（每个 MPC 步）：

```
Python 给出候选控制 ──► MuJoCo 用 ③ 的物理模型 rollout 出一堆轨迹
                                   │
                          产出 states / sensors
                                   │
        ④ 的 reward() 读 states/sensors ──► 每条轨迹一个分数 ──► MPC 选最优
```

**关键心智模型**：XML 描述"世界长什么样、有哪些可测量"（静态结构），Python 描述"目标是什么、
怎么算好坏"（动态逻辑）。两者通过**名字**对接：XML 里命名的 joint / site / sensor，Python 用同名字符串索引。

---

## 1. MuJoCo 基础概念速览

写 XML 前，先建立对这几个标签的直觉：

| 标签 | 是什么 | 类比 |
|---|---|---|
| `<body>` | 一个刚体，可嵌套 | 一个"零件" |
| `<joint>` | 刚体相对父体的自由度 | 关节。`type="free"` = 6 自由度，能在空间自由移动 |
| `<geom>` | 几何体，负责**碰撞**和/或**视觉** | 零件的"形状"。一个 body 可有多个 geom |
| `<site>` | 无质量、无碰撞的"标记点/坐标系" | 贴在零件上的一个**记号**，用来测量或定位 |
| `<inertial>` | 质量、质心、转动惯量 | 零件的"重量属性" |
| `<sensor>` | 从仿真里读出某个量（位置/速度/朝向…） | "传感器"，输出喂给奖励函数 |
| `<asset>` | 资源声明：mesh、material、texture | "素材库" |
| `<default class="X">` | 一组 geom 的默认属性 | "样式表"，避免每个 geom 重复写属性 |

两个最容易混淆的点：

- **geom 既可以是碰撞体也可以是视觉体**。靠 `contype`/`conaffinity` 区分：都为 `0` → 不参与碰撞，纯显示；非 0 → 参与碰撞。约定上视觉体放 `group="2"`，碰撞体放 `group="3"`，在 viser 里可分别开关显示来调试。
- **site 没有物理作用**，它只是个"挂点"。奖励函数关心的"夹爪到物体的距离""物体朝向"，都是通过在物体/机器人上放 site，再用 sensor 测量 site 之间的关系得到的。

---

## 2. `barrel_defs.xml` —— 资产与样式声明

> 路径：`sumo/models/xml/objects/barrel/barrel_defs.xml`
> 职责：声明这个物体要用到的 **mesh / material**（`<asset>`）和 **样式类**（`<default>`）。
> 为什么单独一个文件：MuJoCo 要求 `<asset>`/`<default>`/`<worldbody>` 分区组织；把"素材声明"和
> "物体摆放"拆开，结构更清晰，也方便一个 mesh 被多个场景复用。纯原始几何体的物体（如 barbell）不需要 defs。

```xml
<mujoco model="barrel_defs">
  <default>
    <!-- 视觉样式：mesh 类型，不参与碰撞（contype/conaffinity=0），显示在 group 2 -->
    <default class="visual_barrel">
      <geom type="mesh" contype="0" conaffinity="0" group="2"/>
    </default>
    <!-- 碰撞样式：参与碰撞，priority 决定接触参数谁说了算，显示在 group 3 -->
    <default class="collision_barrel">
      <geom priority="6" group="3"/>
    </default>
  </default>

  <asset>
    <!-- 注意：这个 rgba 只对 MuJoCo 原生渲染器（如离屏录视频）或"mesh 自身不带贴图"时才生效。
         本例的 model.obj 自带贴图（model_baseColor.png），在 viser GUI 里它会被忽略——详见下方逐项说明。 -->
    <material name="barrel_material" rgba="0.15 0.35 0.75 1"/>
    <!-- 导入的 mesh 原始尺寸 ~2.0（沿局部 Y，即圆柱中轴）× ~1.30（横截面）。
         scale=0.45 → ~0.90m 高、~0.585m 直径（一个真实油桶大小）。 -->
    <mesh name="barrel_mesh" file="../../../meshes/barrel/model.obj" scale="0.45 0.45 0.45"/>
  </asset>
</mujoco>
```

逐项说明：

- **`<default class="visual_barrel">`**：定义一个"样式"。之后任何写 `class="visual_barrel"` 的 geom，
  自动获得 `type="mesh" contype=0 conaffinity=0 group=2` 这些属性，不用每个 geom 重复写。
- **`contype="0" conaffinity="0"`**：碰撞掩码。两个 geom 只有当 `(contype_A & conaffinity_B)` 非 0 才会碰撞。
  都设 0 → 这个 geom 永不碰撞 → 纯视觉。
- **`priority`**：当两个碰撞 geom 接触时，priority 高的一方决定接触的摩擦等参数。物体一般给 6，让它压过地面。
- **`<material name=... rgba=...>`**：一个颜色/材质。`rgba` 是红绿蓝 + 不透明度（0~1）。
  **但要特别注意它在不同渲染器下行为不同：**
  - **viser GUI（你日常 `pixi run sumo` 看到的画面）**：可视化器用 `trimesh.load(mesh_file)` 重新加载 mesh
    来渲染。对 `.obj` 文件，trimesh 会**自动连带读取同目录的 `.mtl`**，而 `.mtl` 里用 `map_Kd` 声明了贴图
    （这里是 `model_baseColor.png`）。判定逻辑（`judo/visualizers/model.py` 的 `add_mesh_from_file`
    + `has_material`）是：**只要 mesh 自带真实贴图，就直接用它的贴图，跳过这里的 `<material>`**。
    所以**如果你导入的模型自带贴图文件（`model.obj`+`model.mtl`+`*.png`，或带嵌入贴图的 glb/dae 等），
    这里改 `rgba` 在 GUI 里完全没效果**——画面颜色由那张贴图决定。想改 GUI 里的颜色，要么直接改贴图 PNG，
    要么去掉 mesh 自带贴图（删 `.mtl` 里的 `map_Kd` 行）让这里的 `rgba` 接管。
  - **MuJoCo 原生渲染器（离屏录视频、`mujoco.Renderer` 等）**：这里的 `<material>` **照常生效**。
    注意 MuJoCo 原生加载器**不会**自动读取 `.obj`/`.mtl` 里的贴图（实测编译后模型里只有场景自带的
    几张纹理，`model_baseColor.png` 并未被加载）。所以在原生渲染器下，默认就是这里 `rgba` 的纯色。
  - **想让贴图也作用于 MuJoCo 原生渲染器**，需要在 `<asset>` 里**显式声明** texture 和 material 并绑定
    （MuJoCo 不会替你从 `.mtl` 找）：
    ```xml
    <asset>
      <texture name="barrel_tex" type="2d" file="../../../meshes/barrel/model_baseColor.png"/>
      <material name="barrel_material" texture="barrel_tex"/>   <!-- 用贴图替代纯 rgba -->
      <mesh name="barrel_mesh" file="../../../meshes/barrel/model.obj" scale="0.45 0.45 0.45"/>
    </asset>
    ```
    前提是 mesh 自带 UV 贴图坐标（导入的 `model.obj` 已含 texcoord，MuJoCo 编译时会一并读入）；
    geom 仍写 `material="barrel_material"`，MuJoCo 原生渲染器就会按 UV 把 PNG 贴上去。
    这样两个渲染器看到的颜色才一致。
- **`<mesh name="barrel_mesh" file=... scale=...>`**：
  - `name` 是这个 mesh 在 XML 里的**引用名**（geom 用 `mesh="barrel_mesh"` 引用它）。
  - `file` 路径是**相对于本 XML 文件所在目录**的：`objects/barrel/` 往上三级到 `sumo/models/`，再进 `meshes/barrel/`。
  - `scale` 把原始 mesh 顶点坐标整体缩放。**导入的网格几乎总是要缩放**，因为建模软件的单位和 MuJoCo 的"米"不一致。

> ⚠️ **`scale` 必须和碰撞体尺寸对齐**。下一个文件里碰撞圆柱的半径/高，要按同一缩放后的真实尺寸来写，
> 否则"看到的"和"摸到的"不一致。

---

## 3. `barrel.xml` —— 物体本体

> 路径：`sumo/models/xml/objects/barrel/barrel.xml`
> 职责：把 mesh 包装成一个有**关节、质量、碰撞体、标记点**的真实刚体。

> 📐 本节出现了大量数字（`mass`、`diaginertia`、`quat`、`size`、`friction`、`scale` 等）。
> 它们**几乎都是算出来的，不是经验值**。每个数字的来历、计算公式、以及"改了会怎样"的小实验，
> 见文末 **[附录 A：barrel.xml 里每个数字的来历与改动实验](#附录-abarrelxml-里每个数字的来历与改动实验)**。

```xml
<mujoco model="barrel">
  <include file="barrel_defs.xml"/>          <!-- ① 先引入素材和样式 -->

  <worldbody>
    <body name="barrel" pos="0 0 0">          <!-- ② 刚体 -->
      <joint name="barrel_joint" type="free"/> <!-- ③ 自由关节：6 自由度 -->

      <!-- ④ 质量/惯量（必须显式写，原因见下） -->
      <inertial pos="0 0 0" mass="8.0" diaginertia="0.71 0.71 0.34"/>

      <!-- ⑤ 视觉：导入的 mesh，绕 X 转 90° 把躺着的圆柱立起来 -->
      <geom name="barrel_visual" class="visual_barrel" type="mesh" mesh="barrel_mesh"
            quat="0.7071 0.7071 0 0" material="barrel_material"/>

      <!-- ⑥ 碰撞：单个直立圆柱（圆柱是凸的，无需凸分解） -->
      <geom name="barrel_collision" class="collision_barrel" type="cylinder"
            size="0.29 0.45" friction="0.6 0.3 0.0001"/>

      <!-- ⑦ 必需的 site_object（见第 6 节契约 1） -->
      <site name="site_object" pos="0 0 0" size="0.01"/>
      <!-- ⑧ 任务用的追踪点 -->
      <site name="trace_barrel" pos="0 0 0" size="0.01"/>
    </body>
  </worldbody>
</mujoco>
```

逐项说明：

- **① `<include>`**：把 `barrel_defs.xml` 的内容并进来，于是下面能引用 `barrel_mesh`、`visual_barrel` 等名字。
- **② `<body name="barrel">`**：物体的根刚体。`name` 很重要——Python 里 `get_joint_position_start_index` 等
  会用到关节名，sensor 可能用到 body 名。
- **③ `<joint type="free">`**：自由关节，给这个 body **6 个自由度**（3 平移 + 3 旋转）。
  - 在状态向量里，它对应 **7 个 qpos 数**：`[x, y, z, qw, qx, qy, qz]`（位置 3 + 四元数 4）。
  - 对应 **6 个 qvel 数**：`[vx, vy, vz, wx, wy, wz]`。
  - 这就是 Python 里 `object_pose_start`（qpos 中的位置）和 `object_vel_start`（qvel 中的位置）索引的东西。
  - 没有这个关节，物体就被钉死在世界里、不能动。
- **④ `<inertial>`**：质量、质心 `pos`、对角转动惯量 `diaginertia`。
  - **为什么必须显式写**：如果省略，MuJoCo 会根据碰撞 geom 的 `density`（默认 1000 kg/m³）自动算质量。
    这个圆柱体积约 0.24 m³ → 自动算出 ~240 kg，机器人根本推不动。显式写 `mass="8.0"` 才合理。
  - `diaginertia` 这里按"实心圆柱 m=8, r=0.29, h=0.9"估算。一般物体不必精确，量级对即可。
- **⑤ 视觉 geom**：
  - `class="visual_barrel"` → 自动获得 mesh/不碰撞/group2。
  - `mesh="barrel_mesh"` 引用 defs 里声明的 mesh。
  - **`quat="0.7071 0.7071 0 0"`** 是关键：导入的桶圆柱中轴沿局部 **Y**，但 MuJoCo 是 **Z 朝上**，
    直接放桶会横躺。这个四元数表示"绕 X 轴旋转 90°"，把 Y→Z，让桶立起来。
    （四元数格式是 `[w, x, y, z]`；绕 X 转 90° → `w=cos45°≈0.707, x=sin45°≈0.707`。）
- **⑥ 碰撞 geom**：
  - 用一个 `type="cylinder"` **原始几何体**近似桶，而不是用碰撞 mesh。
    原因：mesh 碰撞昂贵且**必须是凸的**；圆柱本身是凸的、便宜、稳定，一个就够。
  - `size="0.29 0.45"`：圆柱的 `size` 是 `[半径, 半高]` → 半径 0.29m，半高 0.45m（全高 0.9m）。
    与视觉 mesh 缩放后的真实尺寸对齐。
  - 碰撞圆柱的默认轴是 **Z**，桶视觉已转正到 Z，所以碰撞圆柱**不需要** quat，默认就对。
  - `friction="0.6 0.3 0.0001"`：三个数是 `[滑动, 扭转, 滚动]` 摩擦。滚动摩擦给得极小（0.0001），
    这样桶被推倒躺下时能顺畅滚动。
- **⑦⑧ `<site>`**：两个标记点，都在桶心。`site_object` 是硬性要求（第 6 节），`trace_barrel` 给奖励/可视化用。

---

## 4. `spot_barrel_v1.xml` —— 场景装配

> 路径：`sumo/models/xml/spot_tasks/spot_barrel_v1.xml`
> 职责：把机器人 + 物体 + 地面 + 灯光 + 传感器**拼**成一个完整的物理世界。
> 它本身几乎不定义新东西，全靠 `<include>`。**include 的顺序和分区很关键**。

```xml
<mujoco model="spot_barrel_v1">
  <include file="../spot_components/default.xml"/>  <!-- ① 全局样式/编译选项 -->
  <include file="../spot_components/assets.xml"/>   <!-- ② 全局素材（地面纹理等） -->

  <worldbody>
    <light .../> <light .../>                       <!-- ③ 灯光 -->
    <geom name="ground" type="plane" .../>          <!-- ④ 地面 -->

    <body name="body" pos="0 0 0.7">                <!-- ⑤ 机器人本体 -->
      <include file="../spot_components/body.xml"/>  <!--   躯干 -->
      <include file="../spot_components/legs.xml"/>  <!--   四条腿 -->
      <include file="../spot_components/arm.xml"/>   <!--   机械臂+夹爪 -->
    </body>
  </worldbody>

  <include file="../objects/barrel/barrel.xml"/>    <!-- ⑥ 你的物体（自带 worldbody） -->

  <include file="../spot_components/actuator.xml"/> <!-- ⑦ 电机（驱动哪些关节） -->
  <include file="../spot_components/sensor.xml"/>   <!-- ⑧ 通用传感器（含 site_object 引用！） -->
  <include file="../spot_components/contact.xml"/>  <!-- ⑨ 碰撞规则 -->

  <sensor>                                          <!-- ⑩ 本任务专属传感器 -->
    <framepos name="base_position" objtype="site" objname="site_body"/>
    ...
    <framepos name="trace_barrel" objtype="site" objname="trace_barrel"/>
  </sensor>
</mujoco>
```

说明：

- **`spot_components/` 里其实只有 `sensor.xml` 是本仓库的**；`default/assets/body/legs/arm/actuator/contact`
  这些 `include` 在加载时会被一个解析器（`SpotBase._resolve_include_path`）映射到 judo 安装包里的
  `spot_primitive/` 目录——即 Spot 机器人的官方定义。所以你不用关心机器人本体，照抄这些 include 即可。
- **⑤ 机器人放在 `<worldbody>` 里**，物体（⑥）通过 include 带进来（它自己有 `<worldbody>`，MuJoCo 会合并）。
- **⑧ `spot_components/sensor.xml` 是"通用传感器"**，里面写死了对 `site_object` 的引用
  （比如"夹爪到 site_object 的向量""物体的 y 轴朝向"）。**这就是为什么每个物体都必须有 `site_object`**。
- **⑩ `<sensor>` 块是本任务自己加的传感器**。这里大部分（base/foot/com 等）是为了喂给 Spot 的运动策略，
  和 `spot_traffic_cone.xml` 保持一致照抄即可；真正和"桶"有关的只有 `trace_barrel`。
- **`framepos` / `framexaxis` 等是传感器类型**：
  - `framepos`：某个 site/body 的位置（可加 `reftype`/`refname` 得到相对另一个 site 的向量）。
  - `framexaxis`/`frameyaxis`/`framezaxis`：某个坐标系的某根轴在世界里的方向（用来测朝向）。

> 你写新场景时，95% 是照抄某个现成的 `spot_tasks/*.xml`，只改两处：第 ⑥ 行换成你的物体、第 ⑩ 段换成你物体的追踪/测量传感器。

---

## 5. `spot_barrel_push.py` —— 任务逻辑

> 路径：`sumo/tasks/spot/spot_barrel_push.py`
> 职责：定义动作空间、奖励、初始姿态、成功/失败判据。这是任务的"大脑"。

### 5.1 骨架

```python
class SpotBarrelPush(SpotBase[SpotBarrelPushConfig]):
    name = "spot_barrel_push"
    config_t = SpotBarrelPushConfig   # 关联的配置类
    config: SpotBarrelPushConfig

    def __init__(self, config=None):
        super().__init__(model_path=XML_PATH, use_arm=True, config=config)
        # ↑ use_arm=True：机械臂进入动作空间（当推杆）；不开 use_gripper（这版不抓取）

        # 按"名字"拿到各个量在状态向量里的起始下标：
        self.body_pose_start          = self.get_joint_position_start_index("base")
        self.object_pose_start        = self.get_joint_position_start_index("barrel_joint")
        self.object_vel_start         = self.get_joint_velocity_start_index("barrel_joint")
        self.end_effector_to_object_start = self.get_sensor_start_index("sensor_arm_link_fngr")
```

- **`SpotBase` 的开关**决定动作空间：`use_arm` / `use_gripper` / `use_legs` / `use_torso`。
  这版只 `use_arm=True`，所以控制维度 `nu = 3(底盘速度) + 7(机械臂) = 10`。
- **`get_*_start_index("名字")` 是 XML↔Python 的接口**：
  - `get_joint_position_start_index("barrel_joint")` → 桶的位姿在 `qpos` 里的起始下标。
    名字 `"barrel_joint"` 必须和 `barrel.xml` 里 `<joint name="barrel_joint">` 完全一致。
  - `get_sensor_start_index("sensor_arm_link_fngr")` → 该传感器在 `sensordata` 里的起始下标。
    `sensor_arm_link_fngr` 来自通用的 `spot_components/sensor.xml`，是"夹爪指尖到 site_object 的向量"。

### 5.2 状态向量布局（本任务实测值）

这是理解所有索引的关键。Spot + 一个自由物体时：

| 段 | qpos 下标 | qvel 下标 | 维度 | 含义 |
|---|---|---|---|---|
| base 位姿 | 0–6 | 0–5 | 7 / 6 | 机器人底盘 xyz + 四元数 |
| 四条腿 | 7–18 | 6–17 | 12 | 12 个腿关节 |
| 机械臂 | 19–25 | 18–24 | 7 | 7 个臂关节（含夹爪） |
| **物体（桶）** | **26–32** | **25–30** | 7 / 6 | 桶的 xyz + 四元数 / 线速度 + 角速度 |
| 合计 | `nq=33` | `nv=31` | | |

reward 拿到的 `states` 是 `qpos` 和 `qvel` **拼接**后的数组，宽度 `nq+nv=64`：

- `qpos` 占 `states[..., 0:33]`
- `qvel` 占 `states[..., 33:64]`，所以物体速度在拼接数组里的下标 = `33 + 25 = 58` ← 这就是 `object_vel_start=58` 的来历。

而 `success()` 里读的是裸 `data.qvel`（只有 31 长），所以要减回去：`object_vel_start - model.nq = 58 - 33 = 25`。

> 你**不需要手算这些下标**——`get_*_start_index("名字")` 会替你算。上表只是帮你理解它返回的数字是什么意思、
> 以及为什么 `success()` 里要 `- self.model.nq`。

### 5.3 reward()

奖励是**向量化**的：一次性对一批（batch）轨迹算分，返回 `(batch,)`。

```python
def reward(self, states, sensors, controls, system_metadata=None):
    qpos = states[..., : self.model.nq]                                   # 取出位置部分
    object_pos = qpos[..., self.object_pose_start : self.object_pose_start + 3]      # 桶的 xyz
    object_linear_velocity = states[..., self.object_vel_start : self.object_vel_start + 3]  # 桶的线速度
    end_effector_to_object = sensors[..., self.end_effector_to_object_start :
                                          self.end_effector_to_object_start + 3]     # 夹爪→桶 向量

    # 三项奖励（复用 spot_push.py 的辅助函数）：
    goal_reward            = goal_distance_reward(self.config, object_pos)                       # 桶离目标越近越好
    gripper_proximity      = gripper_distance_reward(self.config, np.linalg.norm(end_effector_to_object, axis=-1))  # 末端离桶越近越好
    object_velocity_pen    = object_linear_velocity_reward(self.config, object_linear_velocity)  # 别把桶撞飞

    return goal_reward + gripper_proximity + object_velocity_pen
```

要点：

- **维度约定**：`states` 形状 `(batch, horizon+1, nq+nv)`，`sensors` 是 `(batch, horizon, nsensordata)`，
  `controls` 是 `(batch, horizon, nu)`。所有索引都在最后一维上做切片，前面的 batch/时间维用 `.mean(-1)` 等聚合。
- **奖励是"越大越好"**，所以"距离"类要取负号（`-w * distance`）。`w_*` 权重在 Config 里，可在 GUI 里实时调。
- 三项的设计意图：① 把桶推向目标；② 鼓励末端贴近桶（否则机器人站着不动也"不扣分"）；③ 惩罚桶被打飞（稳定推进）。
- 想加新行为就加一项。比如"让桶滚得快"可以加 `+ w * object_angular_velocity`；"机器人省力"可以加 `- w * ||controls||`。

### 5.4 reset_pose / success / failure

```python
@property
def reset_pose(self):
    # 返回一整条 qpos（长度必须 == nq == 33）
    # 顺序：机器人位姿(7) + 腿(12) + 臂(7) + 物体位姿(7)
    ...
    object_pose = np.array([*object_xy, DEFAULT_BARREL_HEIGHT, 1, 0, 0, 0])  # 桶：xy + 高度 + 四元数(竖直)
    return np.array([*robot_pose, *LEGS_STANDING_POS, *self.reset_arm_pos, *object_pose])
```

- **`reset_pose` 的长度必须严格等于 `nq`**，顺序也必须和状态布局一致（机器人各段在前，物体在最后）。
  这里用了 `*self.reset_arm_pos`（基类提供的默认臂姿）和 `*LEGS_STANDING_POS`（站立腿姿）。
- 物体四元数 `[1,0,0,0]` = 不旋转 = 竖直。想让桶**躺下滚动**：把它改成 `[0.7071,0.7071,0,0]`，
  并把 `DEFAULT_BARREL_HEIGHT` 从 0.45（半高）改成 0.29（半径），让它躺在地上。
- `reset_pose` 里用了 `np.random`，所以每次复位机器人和桶的位置都不同——这让 MPC 学到的策略更鲁棒。

```python
def success(self, model, data, metadata=None):
    object_pos = data.qpos[self.object_pose_start : self.object_pose_start + 3]
    object_vel = data.qvel[self.object_vel_start - self.model.nq :
                           self.object_vel_start - self.model.nq + 3]   # 注意 - nq
    # 桶到目标足够近、且几乎静止 → 成功
    return (np.linalg.norm(object_pos - goal, ord=np.inf) < POS_TOL) and (np.linalg.norm(object_vel) < VEL_TOL)

def failure(self, model, data, metadata=None):
    return data.qpos[self.body_pose_start + 2] <= SPOT_FALLEN_THRESHOLD  # 机器人底盘太低 = 摔了
```

- `success`/`failure` 接收的是**单个** `MjData`（不是 batch），用真实的 `data.qpos`/`data.qvel`。
- 注意 `success` 里读 `data.qvel` 要用 `object_vel_start - model.nq`（因为 `data.qvel` 是裸的 31 长向量，
  而 `object_vel_start=58` 是"拼接数组"里的下标）。

### 5.5 Config

```python
@dataclass
class SpotBarrelPushConfig(SpotPushConfig):
    goal_position: np.ndarray = np_1d_field(
        np.array([0.0, 0.0, DEFAULT_BARREL_HEIGHT]),
        names=["x", "y", "z"], mins=[-5,-5,0], maxs=[5,5,1], steps=[0.1,0.1,0.05],
        vis_name="barrel_goal_position", xyz_vis_indices=[0,1,2], xyz_vis_defaults=[0,0,DEFAULT_BARREL_HEIGHT],
    )
    # 继承自 SpotPushConfig 的权重：w_goal / w_gripper_proximity / w_object_velocity
```

- Config 是 `@dataclass`，里面的字段会**自动出现在 viser GUI 的面板里**，可运行时拖动调参（权重、目标点）。
- `np_1d_field` 是带 GUI 元数据的数组字段：`mins/maxs/steps` 控制滑块范围，`xyz_vis_*` 让目标点在 3D 场景里
  显示成一个可拖动的标记。

---

## 6. 两层之间的"接口契约"

这两条是写任何新物体/任务都必须遵守的硬规则：

### 契约 1：物体必须有一个名叫 `site_object` 的 site

`spot_components/sensor.xml`（通用传感器）里写死了对 `site_object` 的引用：

```xml
<framepos  name="sensor_body"          ... reftype="site" refname="site_object"/>
<frameyaxis name="object_y_axis"       objtype="site" objname="site_object"/>
<framepos  name="sensor_arm_link_fngr" ... refname="site_object"/>
```

所以**任何**被放进 Spot 场景的物体，都必须定义 `site_object`，否则模型编译直接失败。
它代表"物体参考点"，夹爪到物体的距离、物体朝向都基于它算。

### 契约 2：XML 里的名字 = Python 里的索引 key

```
barrel.xml:            <joint name="barrel_joint">
spot_barrel_push.py:   get_joint_position_start_index("barrel_joint")   ← 字符串必须完全一致
```

加一个新可观测量的标准流程永远是：

1. 在 XML 的 `<sensor>` 里定义并**命名**一个传感器；
2. 在 py 的 `__init__` 里用 `get_sensor_start_index("那个名字")` 拿下标；
3. 在 `reward()` 里用那个下标切片读数据。

---

## 7. 维度参考表（本任务实测）

| 量 | 值 | 来源 |
|---|---|---|
| `nq`（位置维度） | 33 | base 7 + 腿 12 + 臂 7 + 桶 7 |
| `nv`（速度维度） | 31 | base 6 + 腿 12 + 臂 7 + 桶 6 |
| `model.nu`（电机数） | 19 | Spot 全身关节电机 |
| `task.nu`（控制维度） | 10 | 底盘速度 3 + 机械臂 7（`use_arm`） |
| `nsensordata` | 72 | 通用 + 任务传感器总输出长度 |
| `object_pose_start` | 26 | 桶位姿在 qpos 的起点 |
| `object_vel_start` | 58 | 桶速度在 (qpos+qvel) 拼接数组里的起点 = 33+25 |

> 验证方式：`pixi run python -c "import sumo.tasks; from sumo.tasks.spot.spot_barrel_push import SpotBarrelPush; t=SpotBarrelPush(); print(t.model.nq, t.model.nv, t.nu)"`

---

## 8. 改造清单：换新物体 / 新任务

**换一个新物体（如把桶换成箱子）**：

1. 把 mesh 放到 `meshes/<obj>/`。
2. 复制 `objects/barrel/` 成 `objects/<obj>/`，改 `_defs.xml` 的 mesh 路径/scale，改 `<obj>.xml` 的
   body 名、关节名、碰撞几何、`inertial`，**保留 `site_object`**。
3. 复制 `spot_tasks/spot_barrel_v1.xml` 成 `spot_<obj>_v1.xml`，改第 ⑥ 行 include 和第 ⑩ 段 trace 传感器。

**换一个新任务逻辑（如从"推"改成"滚到指定朝向"）**：

1. 复制 `spot_barrel_push.py`，改 `name`、`XML_PATH`、`get_*_start_index` 的关节/传感器名。
2. 改 `reward()`：增删奖励项（要新量就先在 XML 加 sensor，再按契约 2 取下标）。
3. 改 `reset_pose` / `success` / `failure` / `Config` 权重。
4. 在 `tasks/__init__.py` 三处注册：加进 `SPOT_TASK_NAMES`、`import`、`register_task(..., **_SPOT_REGISTRATION_KWARGS)`。
5. 冒烟测试：构造一次任务，确认 `nq == len(reset_pose)`、`reward` 返回 `(batch,)`。

---

## 9. 常见坑

- **物体不动 / 重得离谱**：忘了写 `<inertial>`，被默认 density 算成几百公斤。显式写 `mass`。
- **物体横躺 / 朝向不对**：导入 mesh 的"上方向"和 MuJoCo 的 Z 不一致，需要在视觉 geom 上加 `quat` 转正。
- **看到的和摸到的不一致**：视觉 mesh 的 `scale` 和碰撞几何的 `size` 没按同一真实尺寸对齐。
- **编译报错找不到 `site_object`**：违反契约 1，物体没定义 `site_object`。
- **Python 里 `KeyError` / 索引名找不到**：违反契约 2，XML 里的名字和 py 里的字符串不一致。
- **`reset_pose` 维度不匹配**：长度必须 == `nq`，且顺序是"机器人各段在前、物体在最后"。
- **碰撞 mesh 被填实**（凹形物体如带把手的桶）：MuJoCo 碰撞只认凸体，必须把凹形拆成多个凸块；
  规则形状（圆柱/箱）直接用原始几何体近似，不用 mesh 碰撞。

---

## 附录 A：barrel.xml 里每个数字的来历与改动实验

§3 的 `barrel.xml` 里几乎每个数字都不是拍脑袋，而是从"mesh 原始尺寸 + 你想要的真实尺寸 + 物理公式"
推出来的。本附录把每个数字讲透：**它从哪来、怎么算、改大改小会发生什么**。

> 先做一件事：量出 mesh 的原始包围盒。所有尺寸数字都从这里出发。
> ```bash
> pixi run python - <<'PY'
> xs=[];ys=[];zs=[]
> for line in open("sumo/models/meshes/barrel/model.obj"):
>     if line.startswith("v "):
>         _,x,y,z=line.split()[:4]; xs+=[float(x)];ys+=[float(y)];zs+=[float(z)]
> for n,a in (("X",xs),("Y",ys),("Z",zs)):
>     print(n,"size =",round(max(a)-min(a),4))
> PY
> ```
> 本例输出：`X size = 1.30`，`Y size = 2.00`，`Z size = 1.28`。
> X≈Z（≈1.30）是圆形横截面，Y（=2.00）最长 = 圆柱中轴。下面记 **`raw_diam = 1.30`、`raw_h = 2.00`**。

---

### A.1 `scale = "0.45 0.45 0.45"`（在 barrel_defs.xml）

**是算出来的，不是经验值。** 公式：

```
scale = 期望真实高度 / mesh 原始中轴长度
      = target_h / raw_h
      = 0.90 / 2.00
      = 0.45
```

这里 `target_h = 0.90 m` 是你的设计选择（标准油桶约 0.9 m 高）。定了它，scale 就唯一确定。

**为什么三个轴用同一个值（0.45 0.45 0.45）**：等比缩放，保持桶不变形。
验证一下直径是否合理：`raw_diam × scale = 1.30 × 0.45 = 0.585 m`，正好是真实油桶直径——说明这个 mesh
本身比例就对，所以"按高度算 scale"和"按直径算 scale"结果几乎一样（`0.58/1.30 ≈ 0.446`）。
如果某个 mesh 比例失真，你就得分轴用不同 scale 去校正。

> **改动实验**：把 scale 改成 `0.9 0.9 0.9` → 桶变成 1.8 m 高、1.17 m 直径的巨桶。
> 后果不仅是"变大"：碰撞圆柱的 `size` 没跟着变，于是**视觉和碰撞脱节**（看到大桶，但碰撞还是小桶，
> 机器人手会"穿模"或推不到）。所以改 scale **必须**同步改 A.4 的 `size`。

---

### A.2 `<body ... pos="0 0 0">` 与各 `<site pos="0 0 0">`

`pos` 是该元素相对父坐标系的位置 `[x, y, z]`（米）。

- `body pos="0 0 0"`：桶 body 的原点。真正的初始落点由 Python 的 `reset_pose` 决定（§5.4），
  所以这里写 0 即可，相当于"以桶心为原点建模"。
- `site pos="0 0 0"`：把标记点放在桶心。

> **改动实验**：把 `site_object` 改成 `pos="0 0 0.45"` → 参考点移到桶**顶面**。
> 于是"夹爪到 site_object 的距离""目标点对齐"全部以桶顶为基准计算，奖励行为随之改变
> （机器人会去够桶顶而不是桶心）。site 放哪，奖励就盯哪。

---

### A.3 `<inertial pos="0 0 0" mass="8.0" diaginertia="0.71 0.71 0.34">`

- **`pos="0 0 0"`**：质心位置。桶均匀，质心在几何中心。
  > 改成 `pos="0 0 0.4"`（质心抬到接近顶部）→ 桶变成"头重脚轻"，轻轻一碰就倾倒，很难稳定推动。
  > 真实空桶质心偏下，可写 `pos="0 0 -0.2"` 让它更"稳"。

- **`mass="8.0"`**：**这是设计选择**（你想让桶多重）。8 kg = 一个能被 Spot 推动的中等空桶。
  > 改成 `mass="80"` → 桶重如灌满液体，Spot 多半推不动、或推动时自己被反作用力带得打滑。
  > 改成 `mass="0.5"` → 太轻，一碰就飞，`object_velocity` 惩罚项会频繁触发。
  > **这是你调"任务难度"的主要旋钮之一。**

- **`diaginertia="0.71 0.71 0.34"`**：转动惯量，**由 mass 和尺寸算出**（实心圆柱公式，r=0.29, h=0.9）：
  ```
  Izz（绕中轴）   = ½·m·r²             = ½·8·0.29²            = 0.34
  Ixx = Iyy（横轴）= 1/12·m·(3r² + h²)  = 1/12·8·(3·0.29²+0.9²) = 0.71
  ```
  顺序是 `[Ixx, Iyy, Izz]`，绕对称轴（Z）的 `Izz` 最小。
  > 📖 **这两条公式的出处与推导**：维基百科 [List of moments of inertia](https://en.wikipedia.org/wiki/List_of_moments_of_inertia)
  > （查 "Solid cylinder" 一行，直接给出 `½mr²` 和 `1/12·m(3r²+h²)`）；推导与一般理论见
  > [Moment of inertia](https://en.wikipedia.org/wiki/Moment_of_inertia)。其它规则形状（球、箱、薄壳等）
  > 的惯量也都能在前一个列表页直接查到，套公式即可。
  > **为什么必须写**：省略 `<inertial>` 时 MuJoCo 用碰撞 geom 的 `density`（默认 1000 kg/m³）反推质量，
  > 0.24 m³ 的圆柱 → ~240 kg，机器人根本推不动。
  > **改动实验**：把 `diaginertia` 三个值都 ×10 → 桶"转起来很费劲"，被推时几乎不旋转、像被焊住的陀螺；
  > 都 ÷10 → 一推就疯狂打转。量级对就行，不必和 mass 精确自洽（但差太远会让动力学看着别扭）。

---

### A.4 视觉 geom 的 `quat="0.7071 0.7071 0 0"`

**精确算出来的**，不是凑的。四元数格式 `[w, x, y, z]`，"绕单位轴 **a** 转角 θ"对应：

```
quat = [cos(θ/2),  sin(θ/2)·aₓ,  sin(θ/2)·a_y,  sin(θ/2)·a_z]
```

> **四元数（quaternion）是什么、为什么这么定义**：它是一个 4 维数 `[w, x, y, z]`，用来表示 3D 旋转，
> 是欧拉角（绕 xyz 依次转的三个角）之外更稳的另一种表示——**没有"万向锁"奇异、便于插值、数值稳定**，
> 所以机器人/图形/物理引擎几乎都用它。这里 `[w, x, y, z]` 的具体形式（`cos(θ/2)` 配 `sin(θ/2)·轴`）
> 来自"单位四元数与 3D 旋转的对应关系"，不是随意约定，而是能让"四元数乘法 = 旋转复合"成立的唯一自然定义。
>
> 📖 **参考**：
> - 维基百科 [Quaternions and spatial rotation](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation)
>   —— 讲清"为什么 `cos(θ/2)/sin(θ/2)` 这么定义"以及它和旋转的对应（即上面的公式）。
> - 维基百科 [Quaternion](https://en.wikipedia.org/wiki/Quaternion) —— 四元数本身的代数定义与乘法。
> - 轴角 ↔ 四元数的转换：[Axis–angle representation](https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Unit_quaternions)。
> - MuJoCo 的约定（**`[w, x, y, z]` 顺序、单位四元数表示朝向**）见
>   [MuJoCo 文档 · Orientations](https://mujoco.readthedocs.io/en/stable/modeling.html#corientation)
>   （注意：不同库的分量顺序可能是 `[x, y, z, w]`，MuJoCo 用 `w` 在前）。

本例要把 mesh 的中轴从 **Y** 扳到 **Z**（MuJoCo 是 Z-up），即绕 **X 轴**转 90°：

```
θ=90°, a=(1,0,0)  →  [cos45°, sin45°, 0, 0]  →  [0.7071, 0.7071, 0, 0]
```

> **改动实验**：
> - 改成 `quat="1 0 0 0"`（不旋转）→ 桶**横躺**在地上（中轴沿 Y）。
> - 改成 `quat="0.7071 0 0.7071 0"`（绕 Y 转 90°）→ 对这个圆柱看不出区别（绕自身对称轴转）。
> - 想让它"斜躺 45°"：θ=45° 绕 X → `[cos22.5°, sin22.5°, 0, 0] = [0.924, 0.383, 0, 0]`。
>
> 注意：**只有视觉 geom 需要这个 quat**，因为 mesh 的"自带朝向"是歪的。碰撞圆柱（A.5）用 MuJoCo
> 原生 `cylinder`，其默认轴就是 Z，所以**不需要** quat。

---

### A.5 碰撞 geom 的 `size="0.29 0.45"` 与 `friction="0.6 0.3 0.0001"`

**`size="0.29 0.45"`** — `cylinder` 的 size 是 `[半径, 半高]`，由 A.1 的 scale 直接推出：

```
半径   = raw_diam·scale / 2 = 1.30·0.45 / 2 = 0.2925 ≈ 0.29
半高   = raw_h·scale   / 2 = 2.00·0.45 / 2 = 0.45
```

即与视觉 mesh 缩放后**完全同尺寸**。这是"看到的=摸到的"的保证。

> **改动实验**：把 `size` 改成 `0.15 0.45`（半径变小、半高不变）→ 碰撞体比视觉桶**瘦一圈**，
> 机器人手指会插进视觉桶壁里才碰到碰撞体（穿模）。改成 `0.29 0.9`（半高翻倍）→ 碰撞体比视觉桶高一倍，
> 桶会"悬空"或被一根看不见的柱子撑住。**改 scale 必同步改这里。**

**`friction="0.6 0.3 0.0001"`** — 三个数是 `[滑动, 扭转, 滚动]` 摩擦系数，**偏经验/调参**，量级有物理依据：

- **滑动 0.6**：中等。塑料/金属对地面的典型滑动摩擦在 0.3~0.8。越大越"推得动但难滑走"。
- **扭转 0.3**：抵抗绕接触法线自转的阻力。
- **滚动 0.0001**：**故意给极小**，让桶倒下后能顺畅滚动。

> **改动实验**：
> - 滚动摩擦改成 `1.0` → 桶倒地后几乎不滚，像粘在地上（适合"只推不滚"的任务）。
> - 滑动摩擦改成 `0.05` → 地面极滑，桶一碰就溜走、很难精确停在目标点。
> - 这三个值是你区分"推 vs 滚"任务手感的关键旋钮。

---

### A.6 `<site ... size="0.01">`

site 的 `size` 是它在 GUI 里**显示成的小球半径**（米），**纯视觉**，对物理和奖励**无任何影响**。
0.01 = 1 cm 的小点，肉眼可见又不挡视线。

> **改动实验**：改成 `size="0.1"` → GUI 里出现一个 10 cm 的大球，方便你 debug 看 site 在哪；
> 改成 `size="0.001"` → 几乎看不见。改它**不会**改变任何仿真结果。

---

### 小结：哪些是"算出来的"，哪些是"设计选择"

| 数字 | 类型 | 由什么决定 |
|---|---|---|
| `scale 0.45` | **算** | 期望高度 / mesh 原始高度 |
| 碰撞 `size 0.29 0.45` | **算** | raw 尺寸 × scale ÷ 2 |
| `quat 0.7071 0.7071 0 0` | **算** | 绕 X 转 90° 的四元数 |
| `diaginertia 0.71 0.71 0.34` | **算** | 实心圆柱公式（用 mass + size） |
| `mass 8.0` | **选** | 你想要的桶重（=任务难度旋钮） |
| `friction 0.6 0.3 0.0001` | **半选** | 量级有物理依据，具体值靠调"推/滚手感" |
| `pos 0 0 0` | **选** | 建模时把桶心设为原点 |
| `site size 0.01` | **选** | 纯显示大小，随意 |

一句话：**先定一个设计量（期望高度、期望质量），其余尺寸/惯量类数字都能用公式推出来；
只有摩擦和"难度"类参数需要靠跑仿真调手感。**

---

## 附录 B：进阶任务 —— 抓取 + 拖拽（`spot_bucket_drag`）

§1~9 的"推桶"任务里，机器人只是用手臂当推杆，不涉及真抓取。"抓住把手把桶拖走"要复杂得多，
体现在三处：**碰撞体更多**（凹形把手）、**site/sensor 更多**（要测夹爪—把手关系、物体朝向、抓取状态）、
**reward 更复杂**（要"检测"抓没抓住，并分阶段引导）。本附录就这三点展开，以实际的
`spot_bucket_drag`（带提梁的水桶）为例。参考文件：
[bucket.xml](../sumo/models/xml/objects/bucket/bucket.xml)、
[spot_bucket.xml](../sumo/models/xml/spot_tasks/spot_bucket.xml)、
[spot_bucket_drag.py](../sumo/tasks/spot/spot_bucket_drag.py)。

### B.1 建立更多碰撞体模型

**为什么需要多个**：MuJoCo 的碰撞检测**只认凸体**。桶身是凸的（一个 `cylinder` 搞定），但**提梁是凹形**
（中间有洞让手伸进去）。如果直接拿整块视觉 mesh 当碰撞体，MuJoCo 会取它的**凸包**，把洞填实，夹爪就插不进去。
解决办法是**凸分解**：用若干个凸的原始体（`box`/`capsule`/`cylinder`）拼出近似形状。

**我们实际走的工作流**（你以后接入任何带凹形特征的物体都可复用）：

1. **点云分析定位特征**——先量出 mesh 各处的形状，找出把手在哪、朝向、哪根轴是"上"：
   ```bash
   pixi run python - <<'PY'
   import numpy as np
   V=np.array([[*map(float,l.split()[1:4])] for l in open("sumo/models/meshes/bucket/model.obj") if l[:2]=="v "])
   for ax,name in [(1,"Y(疑似上轴)")]:
       lo,hi=V[:,ax].min(),V[:,ax].max(); edges=np.linspace(lo,hi,21)
       for i in range(20):
           s=V[(V[:,ax]>=edges[i])&(V[:,ax]<edges[i+1])]
           if len(s): print(f"{name} slice{i} z~{edges[i]:+.2f} 宽X={s[:,0].ptp():.2f} 厚Z={s[:,2].ptp():.2f}")
   PY
   ```
   桶身切片是"满圆盘"，把手切片是"X 宽、Z 薄"——由此判定提梁是 X-Y 平面内的半圆拱、上轴是 Y。

2. **坐标变换（关键、易错）**：导入 mesh 通常 Y-up，MuJoCo 是 Z-up，视觉 geom 上加了
   `quat="0.7071 0.7071 0 0"`（绕 X 转 90°，见 [附录 A.4](#a4-视觉-geom-的-quat07071-07071-0-0)）。
   **但碰撞原始体和 site 是直接写在最终 Z-up body 系里的，不会自动跟着 mesh 转**，所以你要先把
   "原始 mesh 坐标"换算到"body 坐标"再去摆。绕 X 转 90° 的映射是：
   ```
   bodyX =  rawX · s        （s = scale）
   bodyY = -rawZ · s
   bodyZ =  rawY · s
   ```
   于是把手（raw Y 0.55→1.0）落到 body z 0.12→0.21，桶身落到 body z −0.22→+0.12。

3. **摆原始体**（见 [bucket.xml](../sumo/models/xml/objects/bucket/bucket.xml)）：
   ```xml
   <!-- 桶身：单个凸 cylinder -->
   <geom name="bucket_body_collision" class="collision_bucket" type="cylinder"
         pos="0 0 -0.05" size="0.13 0.17"/>
   <!-- 提梁：3 根细 capsule 拼出拱形，顶杆是抓取目标 -->
   <geom name="handle_top"   class="collision_bucket" type="capsule" size="0.008" fromto="-0.07 0 0.21   0.07 0 0.21"/>
   <geom name="handle_left"  class="collision_bucket" type="capsule" size="0.008" fromto="-0.13 0 0.12  -0.07 0 0.21"/>
   <geom name="handle_right" class="collision_bucket" type="capsule" size="0.008" fromto="0.13 0 0.12   0.07 0 0.21"/>
   ```
   - **`capsule` 用 `fromto="x1 y1 z1  x2 y2 z2"`** 表示"从点1到点2的一段胶囊"，`size` 是半径。
     比用 `pos`+`quat`+长度 直观得多，拼折线/拱形首选。
   - 顶杆半径只有 **8mm**——必须够细，夹爪才能捏住、且 resistance 抓取检测才会触发（见 B.3）。

4. **验证对齐**（碰撞体看不见就是在瞎摆）：viser 默认**隐藏名字含 "collision" 的 geom**
   （见 §正文末"常见坑"上方的说明）。用我们建的离屏渲染脚本看碰撞 vs 视觉是否重合：
   ```bash
   MUJOCO_GL=egl pixi run python -m tools.render_collision spot_bucket_drag
   # 输出 out/collision_render/*.png（front/side/3q/top 四视角）
   ```

**引入方式**：这些碰撞 geom 和普通物体一样写在 `objects/bucket/bucket.xml` 的 `<body>` 里，套
`collision_bucket` class（在 `bucket_defs.xml` 定义）。**不需要额外 include 或重新编译**。

> **修改实验（最常用）**：跑完渲染脚本发现提梁碰撞拱比视觉把手**高了/宽了** → 只改三根 capsule 的 `fromto`。
> 比如想把拱降低 2cm、收窄到 ±0.10：
> ```xml
> <geom name="handle_top"  ... fromto="-0.06 0 0.19   0.06 0 0.19"/>   <!-- 顶杆 0.21→0.19，半宽 0.07→0.06 -->
> <geom name="handle_left" ... fromto="-0.10 0 0.12  -0.06 0 0.19"/>   <!-- 端点 0.13→0.10 -->
> <geom name="handle_right"... fromto="0.10 0 0.12   0.06 0 0.19"/>
> ```
> 同时记得把抓取 site `site_grasp_handle` 的 z 跟着改成 0.19（B.2）。改完 **Reset Task** 重载即可。

> **若要用真碰撞 mesh 而非原始体近似**：把把手在建模软件里**切成几块凸 mesh**导出，在 `bucket_defs.xml`
> 里为每块声明 `<mesh name="handle_piece_1" file="..."/>`，再为每块写一个 `<geom type="mesh" mesh="handle_piece_1" class="collision_bucket"/>`。
> （barrier 的 `crowd_barrier_defs.xml` 里注释掉的 29 块 `collision_*.obj` 就是这种做法。）原始体近似更便宜、更稳，优先用它。

### B.2 更多 site / trace / sensor

**三类 site 的角色**（site 本身无质量无碰撞，只是"挂点"，见 §1）：

| 类型 | 例子 | 作用 |
|---|---|---|
| **必需参考点** | `site_object` | 硬性要求（[契约 1](#契约-1物体必须有一个名叫-site_object-的-site)）；共享传感器测物体位置/朝向都靠它 |
| **功能 site** | `site_grasp_handle`、`site_approach_left/mid/right` | 给 reward 当"目标点"：抓哪里、机器人该站哪 |
| **trace site** | `trace_bucket`、`trace_grasp_handle` | 纯可视化，在 GUI 里画一个点/轨迹，帮你 debug，不影响物理 |

**sensor 类型速查**（sensor 把仿真里的量算出来喂给 reward）：

| sensor 标签 | 输出 | 典型用途 |
|---|---|---|
| `framepos` | 一个 site/body 的位置(3) | 物体在哪 |
| `framepos` + `reftype/refname` | **A 相对 B 的向量**(3) | **核心**：夹爪→把手 的向量，取范数=距离 |
| `framexaxis`/`frameyaxis`/`framezaxis` | 某坐标系一根轴在世界中的方向(3) | 测朝向/对齐（点积=夹角余弦） |
| `jointpos` | 一个关节的角度(1) | 读夹爪开合 `arm_f1x` 做抓取检测 |

**最关键的模式 = 相对向量传感器**。例如 [spot_bucket.xml](../sumo/models/xml/spot_tasks/spot_bucket.xml) 里：
```xml
<framepos name="sensor_gripper_to_grasp_handle"
          objtype="site" objname="site_arm_link_wr1"     <!-- A = 夹爪腕部 site -->
          reftype="site" refname="site_grasp_handle"/>   <!-- B = 把手抓取点 -->
```
它输出"从把手指向夹爪的向量"，reward 里 `np.linalg.norm(...)` 取范数就是**夹爪到把手的距离**——
这是"把夹爪吸向把手"奖励的数据来源。

**bucket 任务用到的 sensor 一览**（XML 定义 → py 读取）：

| sensor 名 | 含义 | reward 里用来 |
|---|---|---|
| `object_x/y/z_axis` | 桶的三根坐标轴朝向 | 判断桶有没有歪/倒 |
| `sensor_gripper_to_grasp_handle` | 夹爪→把手 向量 | 引导夹爪接近把手 |
| `sensor_gripper_x/y/z_axis` | 夹爪三根轴朝向 | 判断夹爪姿态对不对 |
| `sensor_torso_to_approach_*` | 躯干→站位点 向量 | 引导机器人走到桶旁 |
| `arm_f1x`（jointpos，经 `get_joint_position_start_index`） | 夹爪开合角 | 抓取检测（B.3） |

**设置流程永远是三步**（[契约 2](#契约-2xml-里的名字--python-里的索引-key)）：① XML `<sensor>` 里定义并命名 →
② py 的 `__init__` 里 `get_sensor_start_index("名字")` 拿下标 → ③ `reward()` 里按下标切片。

> **修改实验（完整加一个新观测量）**：假设你想加"夹爪离桶**底**多近"作为新信号。三步：
> 1. **XML**：在 `bucket.xml` 加一个底部 site，在 `spot_bucket.xml` 加相对向量 sensor：
>    ```xml
>    <!-- bucket.xml，body 内 -->
>    <site name="site_bucket_bottom" pos="0 0 -0.22" size="0.01"/>
>    <!-- spot_bucket.xml，<sensor> 内 -->
>    <framepos name="sensor_gripper_to_bottom" objtype="site" objname="site_arm_link_wr1"
>              reftype="site" refname="site_bucket_bottom"/>
>    ```
> 2. **py `__init__`**：`self.gripper_to_bottom_idx = self.get_sensor_start_index("sensor_gripper_to_bottom")`
> 3. **py `reward`**：`v = sensors[..., self.gripper_to_bottom_idx : self.gripper_to_bottom_idx+3]`，
>    然后 `dist = np.linalg.norm(v, axis=-1)`，按需加进总 reward。

### B.3 修改 reward —— 原理与做法

**原理**：reward 是若干**塑形项的加权和**，每一项是一个引导信号。函数是向量化的：输入
`states (B,T,nq+nv)`、`sensors (B,T,nsensordata)`、`controls (B,T,nu)`，输出每条候选轨迹一个分数 `(B,)`，
MPC 据此挑最优控制。**"越大越好"，所以"距离/误差"类都取负号**。

把 [spot_bucket_drag.py](../sumo/tasks/spot/spot_bucket_drag.py) 的 reward 按职责**解剖**成五组，
你改 reward 时就是在这五组里增删/调权：

| 组 | 项 | 作用 |
|---|---|---|
| **① 终极目标** | `goal_reward` | 桶离目标点越近越好 |
| **② 接近引导** | `gripper_to_grasp_proximity` `approach_site_proximity` | 没有它们，机器人不知道"先走过去、把夹爪凑到把手" |
| **③ 抓取检测（核心）** | `grasp_quality_reward` `false_grasp_penalty` | 判断"真抓住了没"，并奖励抓住、惩罚空抓 |
| **④ 姿态对齐** | `gripper_orientation` `object_orientation` | 夹爪朝向对、桶保持直立 |
| **⑤ 约束/惩罚** | `fence` `fallen` `object_velocity` `controls` | 别出界、别摔、别把桶撞飞、省力 |

**③ 是这套任务的精髓 —— "阻力式抓取检测"**。没有专门的"是否抓住"传感器，它用一个巧思推断：
```python
gripper_joint_pos = qpos[..., self.gripper_joint_idx]   # 夹爪实际开合角
gripper_joint_cmd = controls[..., 9]                    # 给夹爪的“闭合”指令
position_error = gripper_joint_pos - gripper_joint_cmd   # 实际 - 指令

has_resistance  = position_error > self.config.resistance_threshold   # 命令闭合却合不拢 = 被东西挡住
is_fully_closed = np.abs(gripper_joint_pos - GRIPPER_CLOSED_POS) < tol # 完全合拢 = 手里是空的
is_grasping     = has_resistance & ~is_fully_closed       # 有阻力且没合死 → 抓住了！
is_false_grasp  = is_fully_closed & (gripper_joint_cmd < -0.5)  # 命令抓但合死了 = 空抓
```
直觉：**你让夹爪闭合，但它合不拢（实际比指令开），说明中间夹着把手**——这就判定为"抓住"。
`grasp_quality_reward = w * (is_grasping * grasp_quality)` 奖励真抓，`false_grasp_penalty` 惩罚对空气抓。

**设计原则 = 课程式塑形**：把"拖到目标"拆成`接近→对准→抓住→拖`几个阶段，每阶段一个奖励项接力引导。
**权重决定优先级**——`w_gripper_to_grasp_proximity` 大，机器人先学"凑近把手"；`w_grasp_quality` 大，
它更看重"抓牢"；`w_goal` 大，它急着往目标冲（可能还没抓稳就拖）。调这些权重就是在调"先学什么"。

> **修改实验 1（加"抓住才给拖拽分"的耦合）**：当前 `goal_reward` 不管有没有抓住都在算，机器人可能
> 用身体顶着桶走。把它改成"只有抓住时，靠近目标才奖励"：
> ```python
> # 需要把 is_grasping 从 (B,T) 聚合到与 goal 一致；这里用 per-timestep 耦合
> object_to_goal = np.linalg.norm(object_pos - np.array(self.config.goal_position)[None,None], axis=-1)  # (B,T)
> coupled_goal = -self.config.w_goal * (is_grasping * (-object_to_goal)).mean(-1)  # 抓住时才奖励变近
> # 用 coupled_goal 替换原 goal_reward 加入总和
> ```
>
> **修改实验 2（从"拖"改成"提"）**：加一个"抓住后桶离地越高越好"的项：
> ```python
> object_height = object_pos[..., 2]                      # (B,T)
> lift_reward = self.config.w_lift * (is_grasping * object_height).mean(-1)
> # 并在 Config 加 w_lift: float = 200.0
> ```
> 注意把 `object_fallen` 的判定放宽（提起时桶本来就离地），否则会和提升项打架。

**调参在哪**：以上所有 `w_*` 都是 `Config` 字段，会出现在 GUI 的 **Task 标签页**，可**实时拖动**
（见正文"参数调节"一节）。改 reward 的**结构**（增删项）要改 py 并重启；改**权重**直接在 GUI 调。

> 经验：本任务用 **MPPI 优化器**比 CEM 更容易稳定抓取（采样分布更适合这种窄成功区间的接触任务）。
> 在 GUI 的 optimizer 下拉里切换即可。
