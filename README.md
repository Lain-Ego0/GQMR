# GQMR
General Quadruped Motion Retargeting

项目统一使用 MuJoCo 作为运动学、动力学、仿真和渲染后端，不使用 PyBullet。

项目的完整实施路线见 [GQMR 四足快速运动重定向工具实施计划](docs/IMPLEMENTATION_PLAN.md)。

当前已实现 Motion Schema v1、安全 NPZ I/O、Unitree 可信资产、dog-27 输入、Go2/B2 快速与高质量重定向、MuJoCo 回放、AMP/DeepMimic 导出、`.gqmr` 工程、PySide6 GUI、流式录制、通用/DLC/SLEAP 姿态文件和多目三角化。实际完成度、验证结果与已知问题见 [实施状态](docs/IMPLEMENTATION_STATUS.md)。

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

开发验证：

```bash
pytest -q
PYTHONPATH=src python3 -m gqmr.cli.main --version
PYTHONPATH=src python3 -m gqmr.cli.main validate motion.robot.npz \
  --model-sha256 <sha256>
PYTHONPATH=src python3 -m gqmr robots inspect unitree-go2
PYTHONPATH=src python3 -m gqmr validate motion.robot.npz --robot unitree-go2
```

Unitree 资产命令：

```bash
PYTHONPATH=src python -m gqmr assets status
PYTHONPATH=src python -m gqmr assets install unitree-go2
PYTHONPATH=src python -m gqmr assets pack unitree-go2 go2.gqmr-assets
PYTHONPATH=src python -m gqmr assets unpack go2.gqmr-assets
```
