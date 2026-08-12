# GQMR 第三方软件与分发注意事项

> 基线：`uv.lock`，2026-08-11 生成  
> 本表是发布工程摘要，完整条款以各包内置的 license 文件为准。

| 组件 | 锁定版本 | 许可证 | 用途 |
|---|---:|---|---|
| NumPy | 2.5.2 | BSD-3-Clause | 数组和线性代数 |
| SciPy | 1.18.0 | BSD-3-Clause | 旋转、插值、优化和滤波 |
| MuJoCo | 3.11.0 | Apache-2.0 | 机器人模型、FK/Jacobian、仿真和渲染 |
| Pydantic | 2.13.4 | MIT | 配置与工程 schema |
| PyYAML | 6.0.3 | MIT | 声明式 YAML |
| platformdirs | 4.11.2 | MIT | 资产缓存路径 |
| PyZMQ | 27.1.0 | BSD-3-Clause | MuJoCo Stream Protocol v1 |
| PyAV | 16.1.0 | BSD-3-Clause | 视频解码与 PTS |
| PySide6 / Qt | 6.11.1 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only | GUI |
| shiboken6 | 6.11.1 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only | PySide6 绑定运行时 |

GQMR 选择按 LGPL-3.0 条件使用 PySide6/Qt，不将 Qt 静态链接进 GQMR。发布包必须：

- 保留 Qt/PySide6/shiboken6 的版权和 LGPL 文本。
- 保持 Qt 为可替换的动态库，不禁止用户调试其 LGPL 组件。
- 在 GUI “关于”对话框和发布文档中提供许可证入口。

Unitree Go2/A2/B2 来自固定的 `unitree_mujoco` 提交；Unitree A1/Go1 和 ANYmal C 来自固定的 MuJoCo Menagerie 提交；Lite3 来自固定的 Deep Robotics 官方模型仓库提交。机器人资产与上游 `LICENSE` 一同分发，均按 BSD-3-Clause 使用。

`motion_imitation/retarget_motion/data` 中的历史动作为 CC BY-NC，已同时从 wheel 和 sdist 排除；本地发布扫描会检查任何 `_joint_pos.txt` 都未进入发布物。
