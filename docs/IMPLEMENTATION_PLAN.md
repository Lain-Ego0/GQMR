# GQMR 四足运动重定向工具实施计划

> 状态：核心架构冻结，当前执行顺序已重排
> 计划版本：3.0
> 最后更新：2026-08-12
> 维护规则：修改已冻结的架构、协议或首发范围时，必须在 PR 中说明兼容性和工期影响。

## 0. 当前执行目标与优先级

阶段 0～7 记录从零建设产品基础的历史路线。当前开发不再以“增加机器人型号、输入源或导出格式”为主线，而是完成以下真实数据处理闭环：

```text
真实动作导入
  → 自适应预处理
  → 受约束重定向
  → 问题定位
  → 局部修复
  → 修复前后验收
  → 质量门禁
  → 导出
```

功能优先级重新定义为：

| 优先级 | 范围 | 原则 |
|---|---|---|
| P0 核心 | 局部修复、诊断到修复联动、质量门禁 | 直接决定真实数据能否完成闭环 |
| P1 重要 | 精简诊断曲线、自动质量模式 | 提高调参效率和批量处理能力 |
| P2 后续 | 动力学筛选、新机器人、新导出格式 | 在运动学闭环稳定后再扩展 |

当前已完成的基础包括：真实动作自适应预处理、根平移/根旋转/关节联合优化、问题帧时间轴、区间重算以及局部接触/地面重估。下一项唯一 P0 主任务是完整局部修复引擎。

## 1. 目标和首发边界

GQMR 用于把动物三维关键点或 MuJoCo 四足运动重定向到目标四足机器人，完成检查、编辑和质量评估，并输出 AMP 训练数据。

完整产品工作流：

```text
关键点文件 ──────────────────────┐
动物视频 → 2D姿态 → 3D重建 ─────┼→ AnimalMotion → 重定向 → RobotMotion → 编辑/质检 → AMP
MuJoCo状态 → 语义骨架适配 ───────┘                         │
                                                            └→ MuJoCo回放
```

首个可发布版本的明确边界：

- 首发机器人是 Unitree Go2；B2 用作第二模型和通用性验证。
- 首发输入是当前 27 点三维动物数据、通用 CSV/JSON/NPZ 关键点和统一 `AnimalMotion`。
- 首发训练目标是 Isaac Lab `v2.3.2` 的 AMP `MotionLoader` NPZ 结构；为 Go2 提供项目侧适配器和往返测试。
- DeepMimic JSON 仅作为历史兼容导出器，不作为内部标准格式。
- 输出定义为“运动学 AMP 参考轨迹”。质量报告包含动力学回放指标，但不承诺轨迹无需控制器即可稳定执行。
- 视频阶段接入 DeepLabCut、SLEAP 已推理结果和自定义后端，不负责从零标注或训练通用动物三维模型。
- 首版仅支持自由根节点加 12 个标量 hinge/slide DOF 的四足机器人。ball joint、闭链、轮腿和耦合关节不属于 v1 通用导入范围。

不再作为首版承诺：

- A1、Laikago 和 Vision60。它们在当前仓库中只有依赖 `pybullet_data` 的历史路径，没有可随项目合法、稳定分发的 MuJoCo 资产。
- A1、Laikago、Vision60 的旧配置和输出只用于人工理解迁移算法；找到许可证清晰且通过验证的 MJCF 后，才能重新列入支持矩阵。
- 多人/多动物跟踪、云端推理、远程控制仿真和 Windows/macOS 正式发布。

## 2. 已冻结架构决策

### 2.1 运行环境

| 项目 | 决策 |
|---|---|
| 首发平台 | Ubuntu 24.04 LTS x86-64 |
| Python | 3.12 |
| 包管理 | `pyproject.toml` + `uv.lock`；库声明兼容范围，本地发布验收使用锁文件 |
| GUI | PySide6 6.11 系列，不使用 PyQt6，避免 MIT 项目分发时引入 GPL 约束 |
| 仿真 | MuJoCo Python 3.11.0；唯一模型、FK、Jacobian、接触、仿真和渲染后端 |
| 数值计算 | NumPy 2.5 系列、SciPy 1.18 系列 |
| 视频 | PyAV + 系统 FFmpeg |
| IPC | ZeroMQ ROUTER/DEALER，多段消息；协议见 `STREAM_PROTOCOL_V1.md` |
| 内部数据 | NPZ + 无 pickle 的 JSON manifest；协议见 `MOTION_SCHEMA_V1.md` |
| 大型数据集 | v1.1 再增加 HDF5，不在核心接口中直接暴露 HDF5 对象 |
| 配置验证 | YAML + Pydantic v2；YAML 只保存声明式数据，不允许构造 Python 对象 |
| 测试 | pytest、pytest-qt、Hypothesis；无界面测试使用 EGL，OSMesa 作为回退 |

依赖版本以提交到仓库的 `uv.lock` 为发布事实来源。上述系列版本是当前实施基线，升级必须通过完整测试矩阵。

### 2.2 坐标和旋转

- 统一使用右手坐标系：X 向前，Y 向左，Z 向上，长度单位为米，时间单位为秒。
- 内部和磁盘协议四元数统一使用 `[w, x, y, z]`。
- 选择 `wxyz` 是因为 MuJoCo 自由关节和首个 Isaac Lab AMP 目标都使用该顺序。
- SciPy Rotation 等使用 `xyzw` 的边界必须通过具名转换函数，不允许手写切片转换。
- 腿顺序固定为 `FL、FR、RL、RR`。
- 机器人 DOF 顺序由机器人配置中的 `dof_order` 明确给出，不采用 MJCF 声明顺序、`qpos` 偏移或 actuator 顺序作为业务顺序。
- 根线速度和根角速度在核心 `RobotMotion` 中固定为世界坐标；导出器可以显式转换为根局部坐标。
- 所有旋转重采样使用最短弧 SLERP；禁止对四元数四个分量直接做普通线性插值或滤波。

### 2.3 机器人资产

首发资产来自：

- 仓库：`https://github.com/unitreerobotics/unitree_mujoco`
- 固定提交：`ae6a8403e272733e9996ef59990880330496177f`
- 许可证：BSD-3-Clause
- 首发模型：`unitree_robots/go2`
- 第二验证模型：`unitree_robots/b2`

当前技术验证结果：

| 模型 | MuJoCo 验证版本 | nq | nv | nu | 足端表达 |
|---|---:|---:|---:|---:|---|
| Go2 | 3.8.0 | 19 | 18 | 12 | `FL_foot/FR_foot/RL_foot/RR_foot` body |
| B2 | 3.8.0 | 19 | 18 | 12 | calf body + 配置中的局部足端偏移 |

Go2 机器人配置的规范字段和 DOF 顺序固定如下；实际文件中的 `model_sha256` 由资产安装器填写和校验：

```yaml
schema_version: 1
id: unitree-go2
model: unitree_robots/go2/scene.xml
model_sha256: auto
root_joint: $root
base_body: base_link
dof_order:
  - FL_hip_joint
  - FL_thigh_joint
  - FL_calf_joint
  - FR_hip_joint
  - FR_thigh_joint
  - FR_calf_joint
  - RL_hip_joint
  - RL_thigh_joint
  - RL_calf_joint
  - RR_hip_joint
  - RR_thigh_joint
  - RR_calf_joint
feet:
  FL: {body: FL_foot, local_position: [0.0, 0.0, 0.0], contact_geom: FL}
  FR: {body: FR_foot, local_position: [0.0, 0.0, 0.0], contact_geom: FR}
  RL: {body: RL_foot, local_position: [0.0, 0.0, 0.0], contact_geom: RL}
  RR: {body: RR_foot, local_position: [0.0, 0.0, 0.0], contact_geom: RR}
default_root_position: [0.0, 0.0, 0.27]
default_root_rotation: [1.0, 0.0, 0.0, 0.0]
default_dof_position: [0.0, 0.9, -1.8, 0.0, 0.9, -1.8,
                       0.0, 0.9, -1.8, 0.0, 0.9, -1.8]
```

B2 使用相同 `dof_order` 命名模式，默认根高度 `0.52 m`，每条腿默认 DOF 为 `[0.0, 0.8, -1.6]`。四足端分别绑定 `FL_calf/FR_calf/RL_calf/RR_calf`，局部足端位置固定为 `[0.0, 0.0, -0.35]`；接触通过 calf body 所属碰撞 geom 与地面接触聚合，不依赖缺失的 foot body/site。以上数值进入 FK golden test。

> 2026-08-11 实施修正：上游 B2 `home` 的 calf `-2.84 rad` 超出固定模型硬限位 `[-2.82, -0.43] rad`，且即使夹到 `-2.80 rad` 也属于折叠姿态，只剩 `0.02 rad` 运动余量，无法用于重定向。产品默认姿态因此改为有充足关节余量的站立姿态 `[0.0, 0.8, -1.6]`。该修正不改变 schema 或模型资产 hash，但会改变 B2 默认姿态和机器人配置 hash。

资产不直接复制整个上游仓库，仅将固定提交中 Go2/B2 所需 XML、mesh 和 BSD LICENSE 纳入仓库 `assets/`，并继续用 SHA-256 manifest 逐文件校验。`gqmr assets install/pack/unpack` 保留用于修复、自定义资产根和离线部署。

完整策略见 [ASSET_AND_LICENSE_POLICY.md](ASSET_AND_LICENSE_POLICY.md)。

### 2.4 AMP 目标

第一训练目标固定为 Isaac Lab `v2.3.2`：

```text
source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/motions/motion_loader.py
```

导出器名称：`isaaclab_amp_v232`。输出必须包含：

- `fps`
- `dof_names`
- `body_names`
- `dof_positions`
- `dof_velocities`
- `body_positions`
- `body_rotations`，顺序 `wxyz`
- `body_linear_velocities`
- `body_angular_velocities`

GQMR 不修改 Isaac Lab 内置 humanoid 环境来冒充 Go2。仓库将提供一个最小 Go2 AMP 参考加载适配器，用原版 `MotionLoader` 读取输出，并验证名称重排、随机时间采样和 MuJoCo 往返回放。

### 2.5 GUI、并发和渲染

- GUI 主线程只运行 Qt 事件、输入和纹理提交。
- MuJoCo 渲染在单独渲染线程中完成；OpenGL context 只在创建它的线程使用，渲染线程输出 RGB 帧给 Qt。
- 视频解码使用线程；模型推理和长时间优化使用 `multiprocessing` 的 `spawn` 模式。
- 每个求解/仿真工作进程拥有自己的 `MjData`；不跨进程共享 MuJoCo 指针。
- 大数组优先通过内存映射缓存文件传输。共享内存环形缓冲只用于流式采集实现，不作为普通任务系统的默认方式。
- 任务取消采用协作检查点；两秒未退出时终止工作进程。输出先写临时文件，成功后原子替换，取消任务不得留下可被误认为有效的输出。

### 2.6 插件边界

只有以下两类公开插件：

- `gqmr.pose_backends`：视频/图像姿态后端。
- `gqmr.exporters`：外部训练格式导出器。

使用 Python entry points 发现插件，API 版本为 `1`。MuJoCo 机器人后端、核心文件源和求解器不插件化，避免在唯一后端已经确定的情况下引入无用抽象。

第三方插件是受信任 Python 代码，不是安全沙箱。GUI 必须显示插件包名、版本和来源；默认在独立进程运行并支持超时、取消和资源清理。

姿态插件 API v1：

```python
class PoseBackendV1(Protocol):
    api_version: Literal[1]
    def describe(self) -> PoseBackendInfo: ...
    def load(self, config: dict) -> None: ...
    def infer(self, batch: VideoFrameBatch, cancel: CancelToken) -> KeypointBatch: ...
    def close(self) -> None: ...
```

`PoseBackendInfo` 必须声明包版本、支持的关键点 schema、2D/3D、单/多动物、batch 范围、设备和输出坐标。`KeypointBatch` 必须包含 frame PTS、位置、置信度、实例 ID 和坐标描述。

导出插件 API v1：

```python
class ExporterV1(Protocol):
    api_version: Literal[1]
    def describe(self) -> ExporterInfo: ...
    def validate(self, motion: RobotMotion, config: dict) -> ValidationReport: ...
    def export(self, motion: RobotMotion, destination: Path, config: dict,
               cancel: CancelToken) -> ExportReport: ...
```

## 3. 核心数据和工程格式

正式协议见 [MOTION_SCHEMA_V1.md](MOTION_SCHEMA_V1.md)，核心要求如下：

- schema ID 和语义版本必填。
- 时间戳严格递增，不允许重复或 NaN。
- 关键点名、DOF 名和模型 SHA-256 必填，业务层禁止依赖数字 ID。
- 每帧保存有效掩码和求解状态；失败帧不得静默替换后当作有效帧导出。
- 接触使用 `[0,1]` 概率，最终导出阈值默认 `0.5`，同时保存检测来源。
- NPZ 使用 `np.load(..., allow_pickle=False)`；元数据是 UTF-8 JSON 字节，不保存任意 Python 对象。
- 机器人磁盘数据保存世界系根速度；任何局部系转换均由导出器显式记录。

`.gqmr` 工程文件定义为 ZIP64 容器：

```text
project.gqmr
├── project.json       # schema、资源引用、参数、软件版本
├── edits.json         # 非破坏性编辑命令和稳定 UUID
├── thumbnails/        # 小型预览，可重新生成
└── embedded/          # 仅 portable pack 时包含用户选择嵌入的资源
```

`project.json` v1 的顶层字段固定为：

```json
{
  "schema_id": "gqmr.project",
  "schema_version": "1.0",
  "project_id": "uuid",
  "created_at": "RFC3339",
  "updated_at": "RFC3339",
  "gqmr_version": "...",
  "resources": {},
  "active_animal_motion": "resource-uuid",
  "robot": {"config_id": "unitree-go2", "model_sha256": "..."},
  "mapping": {"config_id": "dog27-to-go2", "sha256": "..."},
  "retarget": {},
  "timeline": {"start": 0.0, "end": 1.0, "loop": false},
  "export_presets": [],
  "ui_state": {}
}
```

`resources` 中每项必须含稳定 UUID、URI、媒体类型、大小、mtime、SHA-256 和是否嵌入。算法结果不直接塞进 `project.json`，只作为具名资源引用。

规则：

- 工程内部路径使用 POSIX 相对 URI；外部文件保存规范化路径、大小、mtime 和 SHA-256。
- 普通保存只保存引用；“打包工程”才嵌入输入和模型资产。
- 缓存键由输入 hash、插件版本、配置规范化 JSON 和 GQMR 算法版本共同计算。
- 保存流程为同目录临时文件、`fsync`、原子替换，并保留一个 `.bak`。
- 打开旧 schema 时只允许显式迁移，不允许就地覆盖原文件。
- 编辑命令引用时间戳和帧 UUID，不引用易因重采样失效的数组下标。

## 4. 推荐代码结构

```text
GQMR/
├── pyproject.toml
├── uv.lock
├── src/gqmr/
│   ├── core/               # motion、schema、coordinates、resampling、validation
│   ├── assets/             # 固定版本下载、hash、许可证和缓存
│   ├── skeletons/          # 动物骨架定义与语义验证
│   ├── robots/             # MuJoCo模型、配置、名称解析、FK/Jacobian/接触
│   ├── retarget/           # 初始化、目标函数、在线和窗口求解器
│   ├── sources/
│   │   ├── files/
│   │   ├── video/
│   │   └── mujoco_stream/
│   ├── editing/            # 非破坏编辑、撤销重做、接触和循环
│   ├── exporters/          # canonical、DeepMimic、Isaac Lab v2.3.2
│   ├── jobs/               # 进程任务、取消、进度和临时产物
│   ├── rendering/          # MuJoCo离屏渲染线程
│   ├── ui/
│   └── cli/
├── configs/
│   ├── skeletons/dog_27.yaml
│   ├── robots/unitree_go2.yaml
│   ├── robots/unitree_b2.yaml
│   └── mappings/dog27_to_go2.yaml
├── docs/
├── tests/
│   ├── synthetic/          # MIT许可的合成测试数据
│   ├── golden/
│   └── integration/
└── tools/
```

## 5. 重定向器定义

### 5.1 快速模式

快速模式用于交互预览和在线处理：

1. 用骨盆、颈部、肩和髋估计根位置和姿态。
2. 检测退化几何；关键向量长度低于 `1e-6 m` 或叉积范数低于 `1e-6` 时，使用上一可靠朝向并标记降级。
3. 根据身体尺度把动物髋足向量映射到机器人工作空间。
4. 使用 MuJoCo site/body Jacobian 和阻尼最小二乘求解 12 DOF。
5. 每帧热启动，硬裁剪关节限位，记录残差和状态。

### 5.2 高质量模式

采用 0.5～2.0 秒滑动窗口，目标为：

```text
L = keypoint + root + velocity + acceleration + contact_lock
  + joint_limit + rest_pose + penetration + self_collision
```

- 置信度直接调节关键点残差权重。
- 关节限位是硬边界，软惩罚只用于远离边界。
- 接触锁定由接触概率和人工标注共同决定。
- 地面穿透默认启用，自碰撞在高质量模式默认启用。
- 失败帧保留 `solver_status` 和残差，不使用无标记的上一帧复制。
- 自动修复只允许处理连续不超过 3 帧的失败段；更长失败段必须由用户确认或从导出区间排除。

本阶段只保证运动学质量。动力学报告执行 PD 跟踪回放，报告跌倒时间、根误差、足端滑移和力矩峰值，但不把“完全动态可执行”作为 v1 重定向求解约束。

## 6. 量化验收基线

详细门槛见 [ACCEPTANCE_CRITERIA.md](ACCEPTANCE_CRITERIA.md)。性能参考机固定为当前开发机：

- Ubuntu 24.04.3 LTS
- Intel Core i7-12650H，16 逻辑 CPU
- 16 GiB RAM
- CPU 模式完成核心性能测试；GPU 只用于视频插件测试

关键发布门槛：

| 项目 | 门槛 |
|---|---:|
| 坐标/四元数往返误差 | 最大绝对误差 `< 1e-6` |
| Go2/B2 FK golden 位置误差 | `< 1e-5 m` |
| Jacobian 有限差分误差 | 相对误差 `< 1e-4` |
| 关节硬限位违规 | `0` 帧 |
| 快速模式有效帧比例 | `>= 99.5%` |
| 有效帧足端目标 RMSE | `<= 0.03 m` |
| 在线快速模式 | p95 `<= 25 ms/帧` |
| 离线快速模式 | `>= 100 帧/秒` |
| 高质量模式 | `>= 10 帧/秒` |
| GUI 输入响应 | p95 `< 100 ms` |
| 任务取消 | `< 2 s`，且无有效输出残留 |
| 流式采集 | 200 Hz、30 分钟、丢帧 `< 0.1%` |
| 源仿真降速 | `< 5%` |
| Isaac Lab 往返 | 名称一致，位置 `< 1e-5 m`，旋转 `< 1e-5 rad` |

如果某项不适用于输入数据，报告必须写出跳过原因，不允许显示为通过。

## 7. 分阶段实施

### 阶段 0：工程和资产基础，1 周

任务：

- 建立 `pyproject.toml`、`uv.lock`、`src/` 和本地自动验证入口。
- 实现 Unitree 固定提交资产下载、hash、许可证安装和离线包。
- 建立 Go2/B2 配置；B2 使用 calf body + 局部足端偏移。
- 固化 dog-27 骨架名称、父子关系、左右对称和历史关键点索引。
- 实现 schema v1、坐标转换和合成测试模型。

出口条件：

- 干净环境单命令安装并获取 Go2；无 PyBullet 运行时依赖。
- Go2/B2 加载、名称解析、FK 和 Jacobian 自动测试通过。
- wheel/sdist 不包含 CC BY-NC 动作数据。

### 阶段 1：CLI 纵向闭环，3 周

任务：

- 实现旧 27 点读取器和 `AnimalMotion` 验证器。
- 实现根姿态退化处理、尺度估计、快速 IK 和状态记录。
- 实现 `RobotMotion`、MuJoCo 回放和质量报告。
- 实现 canonical NPZ、DeepMimic 和 `isaaclab_amp_v232` 导出器。
- 提供：

```bash
gqmr assets install unitree-go2
gqmr inspect input.txt
gqmr convert input.txt --skeleton dog-27 --output animal.npz
gqmr retarget animal.npz --robot unitree-go2 --output robot.npz
gqmr validate robot.npz --robot unitree-go2
gqmr play robot.npz --robot unitree-go2
gqmr export robot.npz --format isaaclab_amp_v232 --output amp.npz
```

出口条件：

- 一段 pace、一段 trot 和一段 turn 完成从旧输入到 Isaac Lab NPZ 的自动闭环。
- Isaac Lab 原版 `MotionLoader` 可以加载和随机插值采样输出。
- 所有数值门槛通过，失败帧和原因可追踪。

### 阶段 2：机器人和映射泛化，2 周

任务：

- 配置支持 body/site/geom 目标和局部足端偏移。
- 加载时验证自由根、12 个标量 DOF、关节顺序、限位和腿语义。
- B2 完整接入并通过与 Go2 相同的测试。
- 实现四足结构候选识别，但自动识别只生成建议配置，用户确认后才能保存。
- 添加机器人导入检查 CLI，不在本阶段制作 GUI 向导。

出口条件：

- Go2 和 B2 无源码修改切换。
- 一个符合 v1 约束的新 12 DOF MJCF 可仅通过配置接入。
- 不符合 v1 的 ball/闭链/轮腿模型在加载阶段明确拒绝并说明原因。

### 阶段 3：GUI MVP，3 周

范围：项目、导入、预览、参数、任务和基础时间范围，不包含完整编辑器。

任务：

- PySide6 主窗口、项目模型和原子保存。
- 源骨架与目标机器人并排/叠加预览。
- 播放、暂停、逐帧、循环和基础区间选择。
- 后台重定向、进度、取消和错误详情。
- canonical/DeepMimic/Isaac Lab 导出。
- 恢复上次未完成任务时只显示诊断，不自动续跑。

出口条件：

- “导入现有数据→Go2 重定向→预览→Isaac Lab 导出”闭环。
- GUI、取消、退出和渲染满足量化门槛。
- 崩溃或强制退出后工程本体不损坏。

发布 `v0.1.0`。

### 阶段 4：MuJoCo 流式接入，2 周

任务：

- 本地加载、暂停、单步、录制和回放。
- 实现 `STREAM_PROTOCOL_V1` 的 publisher/client、握手、序列号、信用窗口、GAP 和重连。
- `qpos/qvel` 先转换为具名 RobotMotion，再按映射转换为 AnimalMotion；不把任意机器人 qpos 直接当动物关键点。
- 默认只绑定 `127.0.0.1`；非本机地址必须显式开启 CurveZMQ。

出口条件：

- 200 Hz、30 分钟采集门槛通过。
- 模型 hash 或关节顺序不匹配时拒绝录制。
- 网络断开和缓冲溢出在时间轴上形成明确 GAP，不伪造连续数据。

发布 `v0.2.0`。

### 阶段 5：视频和姿态插件，4 周

接入顺序：

1. DeepLabCut 已推理文件。
2. SLEAP 已推理文件。
3. 通用 2D/3D CSV、JSON、NPZ。
4. 自定义 Python 推理插件。
5. 多目三角化。
6. 单目三维提升仅作为实验功能。

插件 API v1 必须报告能力、关键点 schema、坐标、批大小、设备、版本，并支持进度、取消和关闭。

出口条件：

- 2D 结果可以与视频精确时间戳对齐并逐帧检查。
- 已标定双目数据可以三角化并输出重投影误差。
- 缓存因输入、模型、插件版本或配置变化而正确失效。
- 单目结果界面明确标记“尺度/深度不确定”，不得伪装成可靠绝对三维数据。

发布 `v0.3.0`。

### 阶段 6：高质量求解和编辑器，4 周

任务：

- 实现滑动窗口高质量求解器。
- 接触检测、足端锁定、地面穿透和自碰撞。
- 裁剪、拆分、拼接、变速、重采样、滤波、根变换和循环处理。
- 接触轨道、失败帧轨道和误差曲线。
- 基于命令模式的撤销重做；大数组编辑使用共享不可变缓存和增量参数。
- PD 动力学回放质量报告。

出口条件：

- walk/trot/pace 的循环首尾根旋转误差 `< 1 degree`，关节最大差 `< 0.03 rad`。
- 接触段平均足端滑移相对快速模式降低至少 `50%`。
- 连续 100 次撤销/重做后数据 hash 与预期一致。

发布 `v0.4.0`。

### 阶段 7：发布加固，3 周

任务：

- 完成长时稳定性、损坏输入、错误模型和资源不足测试。
- Ubuntu 24.04 wheel、源码包和安装说明。
- headless 自动测试、GUI smoke test、性能基线和资产缓存。
- 生成第三方许可证清单、SBOM 和示例工程。
- 对 schema、插件 API 和 stream protocol 做兼容性测试。

出口条件：

- 所有 P0/P1 缺陷关闭。
- 安装、快速入门和 AMP 往返验证在干净 Ubuntu 24.04 环境通过。
- 发布包不包含未授权或非商业限制数据。

发布 `v1.0.0`。

## 8. 工期和里程碑

| 里程碑 | 范围 | 时间 | 累计 |
|---|---|---:|---:|
| M0 | 工程、资产、schema | 1 周 | 1 周 |
| M1 | CLI 和 Go2 AMP 纵向闭环 | 3 周 | 4 周 |
| M2 | B2 和通用映射 | 2 周 | 6 周 |
| M3 | GUI MVP | 3 周 | 9 周 |
| M4 | MuJoCo 流式接入 | 2 周 | 11 周 |
| M5 | 视频和姿态插件 | 4 周 | 15 周 |
| M6 | 高质量求解和编辑器 | 4 周 | 19 周 |
| M7 | 发布加固 | 3 周 | 22 周 |

单人基准工期为 22 周，计划区间为 20～24 周。区间不包含从零标注或训练动物姿态模型，也不包含 Windows/macOS 正式支持。

必须在 M1、M3、M5 后各保留一次继续/缩减范围评审。若落后超过两周，优先缩减单目三维、自动机器人识别和内置推理，不削弱 schema、测试、资产许可证或 AMP 往返验证。

## 9. 测试和完成定义

测试从阶段 0 开始，不集中到最后：

- 单元：坐标、四元数、SLERP、差分、重采样、schema 和配置。
- 性质测试：随机合法旋转、时间戳和关节范围。
- 数值测试：解析 FK、有限差分 Jacobian、IK 残差。
- 回归：MIT 合成数据为正式 golden；CC BY-NC 历史数据只在本地开发运行。
- 集成：输入到 canonical、Isaac Lab 加载器和 MuJoCo 往返回放。
- GUI：工程打开、播放、取消、崩溃恢复和退出。
- 性能：在线/离线求解、渲染、流式采集、长时间内存。

功能只有同时满足以下条件才算完成：

- 自动化测试覆盖正常路径、错误路径和取消路径。
- GUI 与 CLI 调用同一核心实现。
- schema、单位、坐标、名称顺序和版本可机器验证。
- 输出是原子写入的，失败和取消不会留下有效假象。
- 错误包含资源、阶段、字段和建议操作。
- 对应量化门槛通过，或有书面批准的限期豁免。
- 新增资产和依赖已完成许可证记录。

## 10. 已完成的前三个迭代（历史）

### 迭代 1：资产和 schema

- [x] 建立项目包、真实 `uv.lock` 和 Ubuntu 24.04 本地自动验证流程。
- [x] 实现固定提交 Unitree 资产安装器和 manifest。
- [x] 建立 Go2/B2 配置、加载和 FK/Jacobian 测试。
- [x] 实现 `AnimalMotion v1`、`RobotMotion v1` 和安全 NPZ I/O。
- [x] 将 27 点数据转换为具名 dog-27 配置。
- [x] 建立 MIT 合成测试骨架和动作。

逐项实现证据、当前环境差异和待解决问题见 [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)。

### 迭代 2：Go2 CLI 重定向

- [x] 实现坐标适配、根姿态退化处理和尺度估计。
- [x] 实现 MuJoCo Jacobian DLS IK、限位和逐帧状态。
- [x] 实现 canonical NPZ、运动学回放和质量报告。
- [x] 完成 walk/pace/trot/turn 的 Go2/B2 量化本地迁移对照。

### 迭代 3：AMP 往返和 B2

- [x] 实现 `isaaclab_amp_v232` 导出器。
- [x] 使用 Isaac Lab v2.3.2 原版 MotionLoader 加载、1000 点采样和名称重排。
- [x] 从导出结果恢复 MuJoCo Go2/B2 姿态并比较 FK。
- [x] 接入 B2 的 calf body + 足端局部偏移配置。
- [x] 冻结 CLI v0.1 前的核心 schema、机器人、重定向和导出接口。

## 11. 当前产品化工作计划

### 11.1 实施顺序

1. 完整局部修复引擎。
2. 诊断到修复的一键工作流。
3. 统一质量门禁和修复前后对比。
4. 精简但实用的诊断曲线。
5. 自动质量模式。
6. 动力学可执行性筛选。
7. 新机器人型号和新导出格式。

顺序中的后一项不得以“更容易展示”为理由提前，除非前一项已达到出口条件。

### 11.2 迭代 A：完整局部修复引擎（P0）

目标：使任意连续问题片段可在不改动区间外轨迹的前提下，使用显式参数修复并可可靠撤销。

任务：

- 建立可序列化的 `LocalRepairConfig`，不将 GUI 控件值直接传入求解器。
- 实现局部根高度偏置，并在选区两端使用光滑权重过渡。
- 实现动作幅度缩放，分别控制根位移、根倾斜和肢体目标。
- 实现局部平滑强度，只修改选区内部，不在边界产生速度或角速度跳变。
- 支持 `FL/FR/RL/RR` 单足或多足强制锁定、解锁和保持自动接触三种模式。
- 支持选区内重新估计接触和地面，但不覆盖区间外的人工编辑。
- 只重算选中区间，区间外的根位姿、关节、接触和求解状态逐值不变。
- 区间两端同时校验位置、旋转、关节、线速度和角速度连续性。
- 将修复作为非破坏性命令进入工程编辑历史，支持撤销、重做和重现。

出口条件：

- 选区外所有运动学数组和接触概率逐字节不变。
- 边界根位置跳变 `< 2 mm`，根旋转跳变 `< 0.5 degree`，关节跳变 `< 0.02 rad`。
- 启用指定足锁定后，该足选区内接触滑移比修复前降低 `>= 50%`，且不新增穿地或自碰撞。
- 根高度、幅度和平滑参数均有数值边界、单元测试和元数据记录。
- 同一输入和配置重复修复产生相同的内容 hash。

### 11.3 迭代 B：诊断到修复的一键工作流（P0）

目标：用户从“看到问题”到“开始修复”不需手工搜索范围或猜测参数。

任务：

- 把相邻问题帧合并为问题片段，保留主要类型和严重程度。
- 点击问题帧时提供“选中当前问题片段”，自动加入可配置的前后缓冲帧。
- 根据问题类型给出保守修复建议：滑移→足端锁定，穿地→根高度，限位→降低幅度，抖动→加强平滑。
- 修复前必须显示将修改的帧区间、参数和预期解决的诊断类型。
- 提供“上一问题片段/下一问题片段”和“只显示未解决问题”。

出口条件：

- 任意诊断帧可在两次用户操作内变成局部修复选区。
- 自动选区覆盖该连续问题片段的所有帧，且不吞并中间的长时正常片段。
- 系统不自动执行修复；用户确认前不改动作数据。

### 11.4 迭代 C：统一质量门禁与前后对比（P0）

目标：修复不以“看起来好一些”为完成标准，而以可比较指标和导出门禁为依据。

任务：

- 建立统一 `QualityGateReport`，同时包含全片与选区指标。
- 指标至少包含：有效帧比例、残差 RMSE/P95/最大值、限位接近、穿地、自碰撞、接触滑移和根倾斜。
- 每次局部修复生成 before/after/delta 对比，并标识改善、不变和退化。
- 将门禁结果分为 `pass/warn/fail/not_applicable`，每个失败项包含帧索引和建议操作。
- 导出默认拒绝 `fail`；显式强制导出时必须在输出元数据中记录未通过项。
- 门禁阈值作为具名预设保存，不散落在 GUI 或导出器中。

出口条件：

- 修复后若核心指标恶化超过阈值，修复结果标记为退化，不会被 GUI 默认接受。
- 同一 `RobotMotion` 在 CLI、GUI 和导出前检查得到相同门禁结果。
- 每个 `fail` 都能定位到具体问题帧或连续片段。

### 11.5 迭代 D：精简诊断曲线（P1）

只实现能直接支持修复决策的视图：

- 足端目标与实际位置连线，接触足使用独立颜色。
- 根高度修正和 roll/pitch/yaw 修正曲线。
- 最大关节限位接近度、最大足端滑移和残差曲线。
- 曲线与主时间轴、选区和问题片段双向联动。

不在本迭代实现任意拖拽仪表盘、自定义曲线编辑器或通用数据可视化系统。

### 11.6 迭代 E：自动质量模式（P1）

目标：根据预处理动作标签和接触特征选择经验证的参数预设，而不是发明一套无法解释的连续自调参系统。

首批预设：

- 慢走：强接触、强平滑。
- 奔跑：允许腾空、降低平滑。
- 急转：保留 yaw，加强 roll/pitch 稳定。
- 蹲伏：允许较大根高度修正。
- 周期动作：启用首尾闭环保护。
- 非周期动作：启用完整根联合优化。

每次自动选择必须记录触发特征、预设名称和最终参数，用户可覆盖。

### 11.7 迭代 F：动力学可执行性筛选（P2）

在运动学门禁稳定后，再将现有 PD 诊断扩展为筛选器：

- 质心投影与支撑多边形余量。
- 接触冲击和足端着地速度。
- 关节速度、加速度和力矩峰值。
- PD 跟踪中的跌倒、根误差和跟踪失稳。
- 为不稳定片段生成减速或降幅建议，默认不自动修改原动作。

### 11.8 迭代 G：覆盖面扩展（P2）

只在迭代 A～F 的冻结验收集上执行：

- 新增机器人必须同时通过局部修复、诊断、质量门禁和动力学报告测试。
- 新导出格式必须消费通过门禁的统一 `RobotMotion`，不得在导出器中隐式修复动作。
- 不为单一导出目标修改核心运动语义。

## 12. 当前迭代 A 的任务分解

### A1：核心数据与命令

- `LocalRepairConfig` 和参数验证。
- `LocalRepairResult`，包含修复动作、修复区间、实际应用参数和求解诊断。
- 工程编辑命令序列化、撤销和重做。

### A2：目标修正

- 根高度偏置。
- 根位移/根倾斜/肢体幅度缩放。
- 局部接触重估和四足锁定状态。
- 修复窗内平滑强度。

### A3：区间求解与衔接

- 将修正后目标传入根姿态/关节联合求解器。
- 根据选区时长自动决定前后缓冲帧数。
- 对位置与旋转使用光滑权重，不对四元数分量线性混合。
- 重新计算选区内及必要边界的速度、状态和残差。

### A4：验证与 GUI 最小接入

- 单元测试：每个参数的边界、无效组合和决定性。
- 集成测试：选区外不变、边界连续、足端锁定收益和无新增碰撞。
- GUI 提供简化修复面板，参数默认保守，应用前显示选区和参数摘要。
- 对 walk02 和固定合成回归集报告修复前后数据。

迭代 A 不包含质量门禁、高级曲线面板、自动执行修复、动力学优化或新导出格式。
