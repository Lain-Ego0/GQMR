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
gqmr validate <motion.npz> [--model-sha256 SHA256]
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

## 2. 自动验证结果

当前环境：Python 3.12.3、NumPy 1.26.4、SciPy 1.11.4、pytest 7.4.4。

```text
21 passed
python -m compileall: passed
CLI version smoke test: gqmr 0.0.1
```

与验收 ID 的对应关系：

| ID | 当前状态 | 自动证据 |
|---|---|---|
| NUM-001 | 已实现并通过当前环境测试 | 100,000 个随机单位四元数 `wxyz↔xyzw` 往返 |
| NUM-002 | 已实现并通过当前环境测试 | 10,000 个随机旋转的矩阵/四元数测地角误差门槛 |
| NUM-003 | 已实现并通过当前环境测试 | 单位范数、相邻符号连续、短弧中点 |
| NUM-006 | 核心算法已实现并通过当前环境测试 | 非等间隔正弦线速度和恒角速度 RMSE `<1%` |
| DAT-001 | 核心加载器已实现并通过 | object dtype、非法 JSON、重复 NPZ 名和重复 JSON key 均拒绝 |
| DAT-002 | 已实现并通过 | 重复、倒序、NaN、非零起点时间轴均拒绝 |
| DAT-003 | 部分完成 | 调用方提供预期模型 hash 时强制匹配；资产配置自动绑定待实现 |
| DAT-004 | 部分完成 | 数据层阻止严重失败状态成为有效帧；导出器尚未实现 |

这里的“通过”只代表当前开发环境的自动测试。正式发布结论仍需在冻结依赖和参考机上运行完整验收，不提前替代发布验收报告。

## 3. 实施中发现的问题

### ISSUE-001：发布锁文件尚不能生成

- 当前机器没有 `uv`，也没有 `hatchling`。
- `pyproject.toml` 已按冻结基线声明 Python 3.12、NumPy 2.5 和 SciPy 1.18，但本轮只能使用系统已有 NumPy 1.26/SciPy 1.11 做兼容性测试。
- 影响：`uv.lock`、wheel/sdist 构建和“干净环境单命令安装”尚不能声明完成。
- 处理：下一步在具备 `uv` 和包索引访问的构建环境生成锁文件，安装冻结版本后重跑全部测试；不得手工伪造 lock。

### ISSUE-002：模型绑定还缺可信资产来源

- 数据加载器已支持预期模型 hash 校验，但 Unitree 固定提交资产安装器、逐文件 manifest 和机器人配置尚未落地。
- 影响：CLI 目前只能接收显式 `--model-sha256`，还不能通过 `--robot unitree-go2` 自动解析可信 hash；因此 DAT-003 只部分完成。
- 处理：优先实现资产 manifest、安装/状态命令及 Go2/B2 配置，再把 `validate --robot` 接到同一校验路径。

### ISSUE-003：当前没有可发布的机器人闭环

- 仓库中的 `motion_imitation` 是算法迁移参考，且历史动作数据为 CC BY-NC；它不能充当正式 CI 测试集或发布示例。
- 影响：FK/Jacobian、快速 IK、MuJoCo 回放和 AMP 往返尚无可信发布证据。
- 处理：建立 MIT 合成骨架/动作和固定上游 Unitree 资产后，再开始重定向纵向闭环。

## 4. 下一步顺序

1. 实现 Unitree 固定提交资产 manifest、安装、状态、pack/unpack 安全校验。
2. 建立 Go2/B2 YAML 配置和 MuJoCo 名称解析，完成独立 FK/Jacobian golden。
3. 固化 dog-27 骨架与旧 27 点读取器，生成 MIT 合成 walk/trot/pace/turn。
4. 在上述可信底座上实现 Go2 快速 IK，而不是直接依赖历史 PyBullet 路径。
