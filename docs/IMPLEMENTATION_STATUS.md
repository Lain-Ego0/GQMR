# GQMR 实施状态

> 状态日期：2026-08-11  
> 当前代码版本：`0.0.1`  
> 冻结环境：Python 3.12.3、MuJoCo 3.11.0、NumPy 2.5.2、SciPy 1.18.0、PySide6 6.11.1  
> 作用：记录仓库实际完成度、验证证据和仍需外部环境执行的发布审计。

## 1. 已完成的产品闭环

### 1.1 工程、依赖和资产

- `pyproject.toml` 和真实 `uv.lock` 已提交，锁定 33 个包。
- GitHub Actions 在 Ubuntu 24.04 执行 frozen sync、compileall、pytest、wheel/sdist 构建和限制数据扫描。
- Unitree Go2/B2 使用固定提交 `ae6a8403e272733e9996ef59990880330496177f`，实现归档/逐文件 hash、安全安装、状态复验和离线 pack/unpack。
- wheel 和 sdist 都排除 `motion_imitation` 与 CC BY-NC `*_joint_pos.txt`；该检查已固化到 CI。
- 已生成第三方许可证清单和 CycloneDX SBOM，GUI 包含 LGPL/Qt “关于”入口。

### 1.2 Motion Schema v1 与安全 I/O

- `AnimalMotion` / `RobotMotion` 严格验证时间轴、名称、单位、坐标、wxyz 四元数、接触概率、求解状态和模型 hash。
- NPZ 加载固定 `allow_pickle=False`，拒绝 object dtype、额外字段、重复 ZIP/JSON key、路径、加密成员、异常压缩比、非有限 JSON 和错误 dtype/endian。
- 输出使用同目录临时文件、`fsync` 和原子替换；规范内容可生成确定性 SHA-256。
- 数值层包含 wxyz/xyzw、矩阵往返、短弧 SLERP、SO(3) 角速度和非等间时间差分。

### 1.3 dog-27、通用姿态与视频

- dog-27 已固化 27 点名称、父子拓扑、对称、根标志、肢体链和历史索引；详见 `DOG27_PROTOCOL_V1.md`。
- 历史 81 列 TXT 严格读取并执行 `Rz(0.47*pi) @ Rx(0.5*pi)`，保留 CC-BY-NC 来源标记而不将数据进入发布物。
- 独立生成 MIT walk/trot/pace/turn，作为 CI、演示和量化验收数据。
- 支持通用长表 CSV、严格 JSON/NPZ、DeepLabCut CSV 和 SLEAP 多轨 CSV。
- `KeypointBatch` 支持 2D/3D、多实例、置信度、有效掩码和坐标描述；已实现标定相机 DLT 三角化与重投影误差。
- PyAV 解码保留原始 PTS/time base，关键点可按容差与视频精确对齐。
- 姿态插件 API v1 和 entry point 发现已实现，长任务可通过 `spawn` 子进程运行和强制清理。

### 1.4 机器人泛化与 MuJoCo

- Go2/B2 Pydantic YAML 配置、`$root` 解析、业务 DOF 顺序、限位、body/局部足端和 contact geom 绑定已完成。
- FK、body pose、单足/四足 Jacobian、碰撞/地面穿透统计和 actuator 绑定都使用同一 `RobotModel`。
- `robots suggest` 对任意 MJCF 生成只读候选 DOF/足端报告，ball/非 12 DOF/无唯一 free root 会说明拒绝原因。
- 用户 MJCF 目录可计算确定性 hash，通过外部 YAML 无源码修改加载；拒绝 symlink、超量文件和 hash 不匹配。

### 1.5 重定向、回放和编辑

- 动物根姿态估计支持缺失/退化几何，同时估计躯干、肢长和接触概率。
- 快速模式使用热启动 MuJoCo Jacobian DLS，硬裁剪限位并保留逐帧残差/状态。
- 高质量模式在滑动窗中执行接触锁定、平滑正则、地面最小高度和自碰撞回退。
- MuJoCo FK 质量报告包含限位、残差、接触滑移、地面穿透和自碰撞；PD 回放额外报告跌倒时间、根/关节跟踪误差和峰值控制量，且明确标记为诊断而非稳定性承诺。
- 非破坏性编辑包含 trim、变速、重采样、根变换、接触覆盖、SO(3) 平滑滤波、对齐拼接和循环闭合。
- 撤销/重做基于稳定 UUID 命令重放；100 次 undo/redo 后的运动 hash 与预期完全一致。

### 1.6 导出与 Isaac Lab

- canonical NPZ、DeepMimic root xyz + root xyzw + scalar DOF JSON 已实现。
- Isaac Lab `v2.3.2` AMP 导出使用 Hermite/SLERP 重采样，从固定 MuJoCo 模型重算全具名 body 位姿和世界系速度。
- 严格兼容加载器验证字段/形状/dtype/四元数；Go2/B2 逐帧恢复 MuJoCo 的 body 位置和姿态误差 `<1e-5`。
- 已直接下载并执行 Isaac Lab v2.3.2 原版 `MotionLoader`：61 帧、12 DOF、17 body、1000 个随机时间点全部有限，DOF/body 名称重排通过。
- DOF/body 线速度和 body 角速度与重新计算值的 RMSE 满足 `<1%`。

### 1.7 `.gqmr` 工程与 GUI

- `project.json` / `edits.json` 实现冻结 v1 顶层字段、UUID 资源、SHA-256、时间线和编辑命令。
- ZIP64 项目加载拒绝路径穿越、重复/加密/意外成员、超量和异常压缩；嵌入资源会复验 size/hash。
- 普通保存保留一份 `.bak`，portable pack 嵌入选中资源，GUI 可直接打开嵌入工程。
- PySide6 GUI 实现导入/合成、Go2/B2 选择、快速/高质量后台任务、取消、动物/机器人时间线预览、质量日志、编辑/撤销/重做、工程和三种导出。
- 已生成可展示截图 `docs/images/gqmr-gui.png`。

### 1.8 MuJoCo Stream Protocol v1

- ROUTER/DEALER `HELLO/WELCOME/FRAME/ACK/GAP/HEARTBEAT/ERROR/BYE` 已实现，数组是 little-endian 原始 multipart。
- publisher 的仿真线程路径只复制进有界 ring，ZeroMQ socket 由专用线程持有；每个客户独立 credit 和序列状态。
- recorder 验证模型 hash、完整 qpos/qvel layout、shape/dtype/session/seq，乱序或无 GAP 跳号立即拒绝。
- ring 溢出生成明确 GAP，并保存到 `RobotMotion.metadata.retarget_config`；不伪造连续时间轴。
- 默认只允许 loopback；非本机地址必须配置 40-byte Z85 CurveZMQ server/client key。

## 2. CLI 与可展示结果

主要命令：

```text
gqmr inspect / validate / convert / synthetic
gqmr assets install|status|pack|unpack
gqmr robots inspect|suggest|hash-assets|inspect-config
gqmr retarget --mode fast|high-quality
gqmr play [--dynamics]
gqmr export --format canonical|deepmimic|isaaclab_amp_v232
gqmr edit trim|time-scale|root-transform|contact|resample|filter|splice|loop
gqmr project new|info|add|pack
gqmr stream record
gqmr pose inspect|convert|triangulate
gqmr gui
```

`examples/demo` 已包含 MIT 合成 trot 的：

- dog-27 AnimalMotion
- Go2 高质量 RobotMotion
- Isaac Lab AMP NPZ
- DeepMimic JSON
- 运动学 + PD 动力学质量报告
- 嵌入 animal/robot 资源的 portable `.gqmr` 工程

## 3. 自动验证和量化证据

冻结环境命令：

```bash
env -u PYTHONPATH QT_QPA_PLATFORM=offscreen MUJOCO_GL=egl \
GQMR_TEST_ASSET_CACHE=<verified-cache> uv run --frozen pytest -q
```

结果：

```text
98 passed
python -m compileall: passed
wheel + sdist: passed
clean temporary venv wheel install: passed
```

真实资产和数值结果：

| 项目 | 结果 |
|---|---|
| Go2/B2 加载 | `nq=19, nv=18, nu=12` |
| Go2 FK golden | 最大误差 `0 m` |
| B2 FK golden | 最大误差 `5.55e-17 m` |
| Jacobian | 各 100 个姿态，最大相对误差 `<2.81e-10` |
| 快速重定向 | Go2/B2 × walk/trot/pace/turn，有效帧 100%，RMSE `<=0.03 m`，限位违规 0 |
| 高质量接触 | walk/trot/pace 滑移降至快速模式的约 26%–32%，自碰撞帧 0 |
| 离线快速模式 | `1367 frames/s` |
| 高质量模式 | `112 frames/s` |
| 200 Hz 短时流式基线 | 2 s / 400 帧 / 0 GAP / `199.8 Hz` |
| Isaac Lab 原版 Loader | 1000 随机采样全有限，名称重排通过 |
| AMP MuJoCo 往返 | Go2/B2 body 位置/旋转 `<1e-5` |
| 100 次 undo/redo | 最终内容 hash 完全一致 |

## 4. 实施中发现并已解决的问题

- B2 calf 原冻结默认值 `-2.84 rad` 超出上游 `-2.82 rad` 限位，已改为 `-2.80 rad`。
- 初始环境没有 pip/venv/uv，已用验证 wheel 完成开发验证，随后生成真实 `uv.lock` 并在 frozen venv 重跑全部测试。
- 原本机 MuJoCo 3.8.0 只能做预验证；现已在冻结 MuJoCo 3.11.0 完成全矩阵。
- 首次 sdist 构建仍包含 CC BY-NC 历史数据；已添加独立 sdist exclude 并在 CI 同时扫描 wheel/sdist。
- Isaac Lab 轻量兼容加载器的采样细节与原版不完全相同；已使用固定 v2.3.2 原文件 + CPU Torch 直接执行验证，不再用“字段似乎相同”代替验收。

## 5. 发布前仍需执行的长时/外部审计

以下项目不是当前功能缺口，但在宣布 `v1.0.0` 发布验收前不得省略：

1. 在参考机运行 `tools/benchmark_release.py --stream-seconds 1800`，完成 200 Hz / 30 分钟、内存增长、延迟和源仿真降速审计。
2. 在真实桌面 EGL 和参考 GPU 上运行 GUI 长时播放/取消，以及第三方 DeepLabCut/SLEAP 实际数据集验收。
3. 使用正式发布签名、完整许可证文本归档和包签名流程生成 `v1.0.0` 产物。

单目三维提升仍按冻结计划保持为实验能力，GQMR 不内置或伪称提供可靠的通用独眼动物三维模型。
