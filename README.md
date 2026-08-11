# GQMR
General Quadruped Motion Retargeting

项目统一使用 MuJoCo 作为运动学、动力学、仿真和渲染后端，不使用 PyBullet。

项目的完整实施路线见 [GQMR 四足快速运动重定向工具实施计划](docs/IMPLEMENTATION_PLAN.md)。

当前已开始实现 Motion Schema v1 核心数据层、四元数/速度数值工具和安全 NPZ I/O。实际完成度、验证结果与已知问题见 [实施状态](docs/IMPLEMENTATION_STATUS.md)。

当前开发环境可直接运行：

```bash
pytest -q
PYTHONPATH=src python -m gqmr.cli.main --version
PYTHONPATH=src python -m gqmr.cli.main validate motion.robot.npz \
  --model-sha256 <sha256>
```

Unitree 资产命令：

```bash
PYTHONPATH=src python -m gqmr assets status
PYTHONPATH=src python -m gqmr assets install unitree-go2
PYTHONPATH=src python -m gqmr assets pack unitree-go2 go2.gqmr-assets
PYTHONPATH=src python -m gqmr assets unpack go2.gqmr-assets
```
