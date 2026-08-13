# GQMR dog-mmpose backend

这是 GQMR 的可选犬类视频 2D 姿态后端。它通过 Ultralytics YOLO 检测狗，
再用 MMPose 的动物 Top-down 关键点模型处理狗框，并保留原视频 PTS。

该插件不把 PyTorch/MMCV 等 GPU 依赖加入 GQMR 核心包。先按 MMPose 官方安装说明安装
与当前 CUDA 匹配的 PyTorch、MMEngine、完整 MMCV、MMPose、MMDetection 和
Ultralytics，然后安装插件。Python 3.12 环境需要官方 MMCV 2.2 CUDA 轮子，
插件已对 MMDetection 3.2 的导入期版本上限做了进程内兼容：

```bash
python -m pip install -e plugins/gqmr-dog-mmpose
gqmr pose backends
```

配置示例：

```json
{
  "pose_model": "/absolute/path/to/ap10k_pose_config.py",
  "pose_weights": "/absolute/path/to/ap10k_pose_weights.pth",
  "detector_model": "/absolute/path/to/yolo11n.pt",
  "detector_category_ids": [16],
  "detector_confidence": 0.25,
  "device": "cuda:0",
  "score_threshold": 0.15
}
```

`detector_category_ids` 是检测模型的零起始类别索引；对常见 COCO 检测器，
`16` 对应狗。如果换用其他数据集的检测器，必须同步修改。

运行：

```bash
gqmr pose video dog.mp4 --backend dog-mmpose \
  --config dog-mmpose.json --batch-size 16 --output dog.2d.npz
gqmr pose inspect dog.2d.npz --format generic-npz
```

当前后端输出模型原生的 AP-10K/APT-36K/AnimalPose 2D 点，不会伪造 DOG27 的不可见
中间关节，也不进行单目 3D 提升。后续的视觉骨架映射与时序 3D 模型应消费此 NPZ。
