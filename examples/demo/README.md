# GQMR 完整闭环示例

本目录全部运动数据由 GQMR 的 MIT 合成 `trot` 生成器产生，不包含 `motion_imitation` 中的 CC BY-NC 历史动作。

| 文件 | 用途 |
|---|---|
| `trot.animal.npz` | dog-27 `AnimalMotion v1` |
| `trot.go2.npz` | Unitree Go2 高质量接触锁定 `RobotMotion v1` |
| `trot.amp.npz` | Isaac Lab v2.3.2 AMP 导出 |
| `trot.deepmimic.json` | DeepMimic 历史兼容帧布局 |
| `quality.json` | MuJoCo 运动学与 PD 动力学诊断 |
| `trot-portable.gqmr` | 嵌入 animal/robot 资源的 portable 工程 |

重新验证：

```bash
uv run gqmr validate examples/demo/trot.go2.npz --robot unitree-go2
uv run gqmr play examples/demo/trot.go2.npz --robot unitree-go2 --dynamics
uv run gqmr project info examples/demo/trot-portable.gqmr
```
