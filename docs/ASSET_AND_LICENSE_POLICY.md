# GQMR 资产和许可证策略

> 状态：冻结

## 1. Unitree 机器人资产

首发资产来源固定为：

```text
Repository: https://github.com/unitreerobotics/unitree_mujoco
Commit:     ae6a8403e272733e9996ef59990880330496177f
License:    BSD-3-Clause
Models:     unitree_robots/go2, unitree_robots/b2
```

规则：

- 不跟随 `main` 自动升级。
- 下载固定提交的归档，解包前防止绝对路径和 `..` 路径穿越。
- 只安装模型所需 XML、mesh、纹理和上游 LICENSE。
- 仓库维护一份逐文件 SHA-256 manifest；任一文件不匹配即拒绝加载。
- 缓存目录通过 `platformdirs` 决定，不硬编码用户 home。
- `gqmr assets status` 显示来源、提交、许可证、大小、hash 和验证状态。
- `gqmr assets pack` 生成带 manifest 和许可证的离线包；`unpack` 重复校验。
- 发布包和 UI 不得暗示 Unitree 对 GQMR 的认可或背书。

升级上游资产必须：

1. 更新提交和 manifest。
2. 重新完成 Go2/B2 加载、FK、Jacobian、接触和 AMP 往返测试。
3. 人工检查上游许可证和模型结构变化。
4. 在变更日志记录关节、body、geom、惯量和碰撞变化。

## 2. 旧机器人

A1、Laikago、Vision60 当前只存在指向 `pybullet_data` 的历史配置。v1 不下载、不分发、不承诺支持它们。

重新加入支持矩阵的条件：

- 找到许可证允许项目使用和分发的 MJCF/URDF 及全部 mesh。
- 固定来源提交和逐文件 hash。
- 通过与 Go2 相同的自动验收。
- 明确模型是官方、社区还是项目转换版本。

## 3. 动作数据

`motion_imitation/retarget_motion/data` 的动作数据采用 CC BY-NC 4.0：

- 允许本地研究、算法迁移对照和非商业开发。
- 不打入 wheel、sdist、示例工程或商业训练数据发布物。
- CI 正式测试使用项目生成的 MIT 合成动作，不依赖该目录。
- 任何报告使用这些数据时保留来源和 Attribution-NonCommercial 声明。

历史 `motion_imitation` 代码采用 Apache-2.0。迁移期间保留原许可证；删除历史代码前，确认已迁移代码的来源说明和 NOTICE 要求。

## 4. GUI 和依赖

- GUI 使用 PySide6，按 LGPLv3/GPLv3/商业三许可中的 LGPL 路径进行动态链接分发。
- 不修改或静态链接 Qt 二进制。
- 安装和关于页面提供 Qt/PySide6 许可证入口。
- Python 依赖在发布时生成 SPDX SBOM 和第三方许可证清单。
- 新增依赖不得只在代码审查中口头确认许可证，必须进入机器可读清单。

## 5. 用户内容和插件

- 用户导入的视频、模型和动作默认只在本机处理，不上传网络。
- 工程打包前显示将要嵌入的外部资产清单。
- 第三方姿态插件是受信任代码，安装和首次启用时显示包名、版本和来源。
- NPZ 一律禁用 pickle；YAML 使用 safe loader；ZIP 工程解包防止路径穿越和压缩炸弹。

