# GQMR 实施状态

> 状态日期：2026-08-11  
> 当前代码版本：`0.0.1`  
> 作用：记录实际代码完成度、验证证据和实施中发现的问题；冻结的产品决策仍以 `IMPLEMENTATION_PLAN.md` 和各 v1 协议为准。

## 1. 本次已实现

### 1.1 工程骨架

- 建立 `pyproject.toml`、`src/gqmr`、`tests` 和 `gqmr` console script。
- 包构建明确只包含 `src/gqmr`，历史 CC BY-NC 动作目录不进入 wheel 配置。
- CLI 已提供：

```text
gqmr inspect <motion.npz>
gqmr validate <motion.npz> [--model-sha256 SHA256 | --robot ROBOT]
gqmr assets install|status|pack|unpack ...
gqmr robots inspect <robot>
```

- CLI 成功和失败结果均为 JSON；协议错误返回退出码 `2`，不向普通用户输出 Python traceback。

### 1.2 数值基础

- 具名 `wxyz_to_xyzw`、`xyzw_to_wxyz`，禁止业务代码手写四元数切片。
- 四元数归一化、序列符号连续化、旋转矩阵往返、无符号测地角。
- 最短弧 SLERP，拒绝时间范围外的隐式外推。
- 非等间隔时间轴的位置/标量速度估计。
- 基于 SO(3) log map 的世界系根角速度估计。

### 1.3 Motion Schema v1

- 实现 `AnimalMotion` 和 `RobotMotion` 不可变数据对象。
- 验证 schema 固定字段要求：时间戳、名称唯一性、数组形状、概率范围、有效掩码、求解状态、残差、四元数单位范数和 metadata 固定语义。
- `solver_status` 为 `UNREACHABLE`、`MISSING_INPUT` 或 `NUMERICAL_ERROR` 的帧禁止标为有效；`INTERPOLATED` 保留文档规定的人工修复语义。
- RobotMotion 可在加载时绑定调用方给出的 `model_sha256`，不匹配立即拒绝。

### 1.4 安全 NPZ I/O

- 加载固定使用 `allow_pickle=False`。
- 加载前检查 ZIP/NPZ 成员数量、成员路径、重复成员名、加密成员、未压缩总大小和异常压缩比。
- 严格拒绝 object dtype、非 Unicode 名称、错误 endian/dtype、缺失字段、额外字段和非 C-contiguous 数组。
- `metadata_json` 只接受 UTF-8 `uint8[M]` 严格 JSON，拒绝 NaN/Infinity、重复 JSON key 和非对象顶层值。
- 保存统一输出 little-endian float、C contiguous 数组和稳定排序 JSON。
- RobotMotion 保存时修正相邻有限四元数的符号跳变。
- 保存使用同目录临时文件、文件 `fsync`、原子替换和目录 `fsync`；失败时清理临时文件。

### 1.5 Unitree 资产供应链

- 内置 Go2/B2 固定提交逐文件 manifest，官方归档 SHA-256 为 `824a51b228c317348866180b1214ed736621d2163006d682156d54b6a55da711`。
- Go2 只安装许可证、2 个 XML 和 16 个 OBJ，共 19 个文件、28,427,057 bytes。
- B2 只安装许可证、2 个 XML 和 31 个 mesh，共 34 个文件、31,573,113 bytes。
- 预览图、terrain scene、height field 和其他上游仓库内容不会进入默认资产缓存或离线包。
- 实现 `assets install/status/pack/unpack`，支持显式缓存目录、预下载归档、损坏缓存修复和离线部署。
- tar.gz 安装拒绝错误归档 hash、绝对路径、`..`、重复成员、链接/特殊文件、超量成员和异常解压体积，只提取 manifest 声明的文件。
- 离线 ZIP64 包再次校验固定内置 manifest、成员集合、逐文件 hash、路径、文件类型、压缩比和解压总量。
- 安装状态会报告来源提交、许可证、精确大小、`model_sha256`、缺失、损坏和意外文件。
- `model_sha256` 定义为 XML/mesh 文件清单的确定性聚合 hash；许可证保留在逐文件完整性校验中，但不改变机器人模型身份。完整算法见 `ASSET_MANIFEST_V1.md`。

### 1.6 机器人配置与 MuJoCo 绑定

- 使用 Pydantic v2 严格验证内置 Go2/B2 YAML；拒绝未知字段、重复 YAML key、不安全模型路径、错误 hash、重复 DOF、非单位根四元数和非有限默认值。
- `$root` 解析为模型中唯一 free joint，因此同时支持 Go2 的无名 free joint和 B2 的 `floating_base_joint`，业务代码不依赖上游根关节名称差异。
- 加载时要求可信资产状态、配置 `model_sha256` 和 manifest 完全一致。
- 验证模型恰好包含一个自由根、12 个具名标量 hinge/slide DOF、完整硬限位、`base_body`、足端 body/局部点和接触 geom。
- 业务 DOF 顺序只来自配置；已用故意打乱 MJCF 声明顺序的合成模型证明 qpos/qvel 地址不会冒充业务顺序。
- 提供根姿态/12 DOF 写入、body/足端 FK、单足/四足 Jacobian 和碰撞 geom 聚合 API。
- Go2 使用具名 `FL/FR/RL/RR` contact geom；B2 按 calf body 聚合所有启用接触的 collision geom。
- `gqmr robots inspect` 输出模型维度、配置/model hash、业务 DOF、qpos/qvel 地址、关节限位和足端绑定。
- `gqmr validate --robot` 同时验证可信资产、机器人配置、模型 hash 和 RobotMotion DOF 规范顺序。
- 详细机器语义见 `ROBOT_CONFIG_V1.md`。

## 2. 自动验证结果

当前环境：Python 3.12.3、MuJoCo 3.8.0、NumPy 1.26.4、SciPy 1.11.4、Pydantic 2.11.7、pytest 7.4.4。Pydantic/platformdirs 仅为本机验证安装在被忽略的 `.deps`。

```text
36 passed
python -m compileall: passed
CLI version smoke test: gqmr 0.0.1
```

与验收 ID 的对应关系：

| ID | 当前状态 | 自动证据 |
|---|---|---|
| NUM-001 | 已实现并通过当前环境测试 | 100,000 个随机单位四元数 `wxyz↔xyzw` 往返 |
| NUM-002 | 已实现并通过当前环境测试 | 10,000 个随机旋转的矩阵/四元数测地角误差门槛 |
| NUM-003 | 已实现并通过当前环境测试 | 单位范数、相邻符号连续、短弧中点 |
| NUM-004 | 已实现并通过当前环境测试 | Go2 FK 最大误差 `0 m`；B2 `5.55e-17 m`，独立解析链对照 |
| NUM-005 | 已实现并通过当前环境测试 | Go2/B2 各 100 个合法姿态，最大相对误差分别 `2.81e-10`、`2.76e-10` |
| NUM-006 | 核心算法已实现并通过当前环境测试 | 非等间隔正弦线速度和恒角速度 RMSE `<1%` |
| DAT-001 | 核心加载器已实现并通过 | object dtype、非法 JSON、重复 NPZ 名和重复 JSON key 均拒绝 |
| DAT-002 | 已实现并通过 | 重复、倒序、NaN、非零起点时间轴均拒绝 |
| DAT-003 | 核心验证路径已完成 | `validate --robot` 强制校验可信资产、配置、模型 hash 和 DOF 顺序；回放/导出命令尚未实现 |
| DAT-004 | 部分完成 | 数据层阻止严重失败状态成为有效帧；导出器尚未实现 |

真实上游集成验证：

| 项目 | 结果 |
|---|---|
| 固定归档下载与归档 SHA-256 | 通过 |
| Go2 逐文件安装和状态复验 | 通过，`model_sha256=48baeb791c25c3fdaca0163c614145ade0e29d710ee9fcce9d8a5f551e3ca2e1` |
| B2 逐文件安装和状态复验 | 通过，`model_sha256=2ebeb90cb3cee67b4ae37e719244454b854719db126d9394ed89d3f0c9ec76e5` |
| Go2 离线 pack/unpack | 通过，解包后逐文件复验通过 |
| MuJoCo 加载 | Go2/B2 均通过；`nq=19`、`nv=18`、`nu=12` |
| 默认姿态 FK golden | Go2 最大误差 `0 m`；B2 最大误差 `5.55e-17 m` |
| Jacobian 中心有限差分 | 每个模型 100 个合法姿态；最大相对误差 `<2.81e-10` |

这里的“通过”只代表当前开发环境的自动测试。正式发布结论仍需在冻结依赖和参考机上运行完整验收，不提前替代发布验收报告。

## 3. 实施中发现的问题

### ISSUE-001：发布锁文件尚不能生成

- 当前机器没有 `uv`，也没有 `hatchling`。
- `pyproject.toml` 已按冻结基线声明 Python 3.12、NumPy 2.5 和 SciPy 1.18，但本轮只能使用系统已有 NumPy 1.26/SciPy 1.11 做兼容性测试。
- 影响：`uv.lock`、wheel/sdist 构建和“干净环境单命令安装”尚不能声明完成。
- 处理：下一步在具备 `uv` 和包索引访问的构建环境生成锁文件，安装冻结版本后重跑全部测试；不得手工伪造 lock。

### ISSUE-002：模型绑定还缺可信资产来源（已解决）

- 固定提交资产安装器、Go2/B2 逐文件 manifest 和可信 `model_sha256` 已落地。
- Go2/B2 Pydantic YAML、配置 hash、MuJoCo 名称解析和 `validate --robot` 已接入同一可信链。
- 回放和导出命令实现后仍需复用该入口，不允许各自绕过模型绑定。

### ISSUE-003：当前没有可发布的机器人闭环

- 仓库中的 `motion_imitation` 是算法迁移参考，且历史动作数据为 CC BY-NC；它不能充当正式 CI 测试集或发布示例。
- 资产、模型加载、FK 和 Jacobian 已有可信证据；快速 IK、MuJoCo 轨迹回放和 AMP 往返仍未实现。
- 处理：建立 MIT 合成骨架/动作后开始重定向纵向闭环。

### ISSUE-004：本机 MuJoCo 版本不是冻结发布基线

- 本机可用 MuJoCo 为 3.8.0；冻结运行环境指定 MuJoCo 3.11 系列。
- 本轮真实 Go2/B2 加载结果与计划中的技术验证数据一致，但只能算资产兼容性预验证。
- `pyproject.toml` 已声明 `mujoco>=3.11,<3.12`。生成 `uv.lock` 后必须在 3.11 系列重跑加载、FK、Jacobian、接触和 AMP 往返，才能形成发布证据。

### ISSUE-005：B2 冻结默认姿态违反硬限位（已解决）

- 原计划每条 B2 腿默认值为 `[0.0, 1.28, -2.84] rad`，但固定上游 MJCF 的 calf 硬限位为 `[-2.82, -0.43] rad`。
- 这会使模型初始化立即违反 RET-001，并被严格加载器正确拒绝。
- 默认 calf 已修正为 `-2.80 rad`，保留 `0.02 rad` 安全余量；schema 和资产 `model_sha256` 不变，B2 默认姿态和 `robot_config_sha256` 改变。

### ISSUE-006：当前 Python 缺少 venv/pip

- 系统 Python 3.12 没有 `ensurepip`、`pip` 或可用的 `python3.12-venv`，无法创建常规开发虚拟环境。
- 本轮只为验证从 PyPI 下载固定 wheel、逐个核对官方 SHA-256 后解压到被忽略的 `.deps`，没有修改系统或用户级 Python。
- 该目录不是发布物，也不替代 `uv.lock`；干净构建环境仍必须通过 `uv sync` 安装正式锁定依赖。

## 4. 下一步顺序

1. 固化 dog-27 骨架语义、历史索引和严格读取器。
2. 生成 MIT 合成 walk/trot/pace/turn，作为正式 CI 测试集。
3. 实现动物根姿态退化处理、尺度估计和状态码。
4. 在已验证的 MuJoCo Jacobian 上实现 Go2 快速 DLS IK。
