# GQMR
General Quadruped Motion Retargeting

![GQMR GUI](docs/images/gqmr-gui.png)

项目统一使用 MuJoCo 作为运动学、动力学、仿真和渲染后端，不使用 PyBullet。

项目的完整实施路线见 [GQMR 四足快速运动重定向工具实施计划](docs/IMPLEMENTATION_PLAN.md)。

当前已实现 Motion Schema v1、安全 NPZ I/O、Unitree 可信资产、dog-27 输入、Go2/B2 快速与高质量重定向、MuJoCo 回放、AMP/DeepMimic 导出、`.gqmr` 工程、PySide6 GUI、流式录制、通用/DLC/SLEAP 姿态文件和多目三角化。实际完成度、验证结果与已知问题见 [实施状态](docs/IMPLEMENTATION_STATUS.md)。

GUI 会直接渲染 Go2/B2 的 MuJoCo mesh，并随时间轴更新机器人姿态；在模型区域拖动可旋转视角，滚轮可缩放。

使用已提交的锁文件安装：

```bash
uv sync --frozen --extra test
uv run gqmr --version
```

最短演示闭环：

```bash
uv run gqmr assets install unitree-go2
uv run gqmr synthetic trot --duration 2 --output trot.animal.npz
uv run gqmr retarget trot.animal.npz --robot unitree-go2 \
  --mode high-quality --output trot.go2.npz
uv run gqmr play trot.go2.npz --robot unitree-go2 --dynamics
uv run gqmr export trot.go2.npz --robot unitree-go2 \
  --format isaaclab_amp_v232 --output trot.amp.npz
uv run gqmr gui
```

仓库中的 [`examples/demo`](examples/demo) 包含由 MIT 合成 trot 生成的 AnimalMotion、Go2 RobotMotion、Isaac Lab AMP、DeepMimic JSON、质量报告和 portable `.gqmr` 工程，不包含历史 CC BY-NC 动作。

开发验证：

```bash
env -u PYTHONPATH QT_QPA_PLATFORM=offscreen MUJOCO_GL=egl \
  GQMR_TEST_ASSET_CACHE="$HOME/.cache/gqmr" uv run --frozen pytest -q
PYTHONPATH=src python3 -m gqmr.cli.main --version
PYTHONPATH=src python3 -m gqmr.cli.main validate motion.robot.npz \
  --model-sha256 <sha256>
PYTHONPATH=src python3 -m gqmr robots inspect unitree-go2
PYTHONPATH=src python3 -m gqmr validate motion.robot.npz --robot unitree-go2
```

当前阶段不启用 CI/CD，仓库中的自动流水线配置已移除；发布前按
[实施状态](docs/IMPLEMENTATION_STATUS.md)记录的冻结环境、本地构建、包扫描和干净环境演示步骤验收。
本轮可展示流程和量化结果见 [展示与验收说明](docs/DEMO_ACCEPTANCE.md)。

Unitree 资产命令：

```bash
PYTHONPATH=src python -m gqmr assets status
PYTHONPATH=src python -m gqmr assets install unitree-go2
PYTHONPATH=src python -m gqmr assets pack unitree-go2 go2.gqmr-assets
PYTHONPATH=src python -m gqmr assets unpack go2.gqmr-assets
```
