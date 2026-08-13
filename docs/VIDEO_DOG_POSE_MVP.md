# 犬类视频动作提取 MVP

## 当前已完成

GQMR 已提供真实视频到姿态结果的可运行入口：

```text
PyAV 增量解码 -> PoseBackendV1 -> KeypointBatch -> generic keypoint NPZ
```

- 保留视频原始 PTS/time base，不使用估算帧率替代时间戳。
- 长视频按批解码；姿态模型在隔离子进程内只加载一次。
- 验证每个输入帧都有且只有一个输出时间戳。
- 验证各批次关键点名称、实例、维度和坐标系保持不变。
- 最后一批不足模型最小 batch 时复制末帧补齐，推理后裁掉补帧结果。
- 输出记录视频 SHA-256、文件大小、截取范围、后端版本、配置 hash 和批次信息。
- 提供可选 `gqmr-dog-mmpose` 后端，支持 MMPose 动物 2D 姿态模型。

## 命令

桌面 GUI 也已提供同样的功能。启动 `gqmr gui` 后，在左侧“狗视频动作提取”中：

1. 确认后端为 `MMPose dog 2D`。
2. 选择本机模型 JSON 配置。
3. 选择狗视频。
4. 点击“提取 2D 关键点”并选择 NPZ 保存位置。
5. 完成后查看带关键点覆盖的首帧预览。

下面的 CLI 入口保留给自动化和批处理。

列出当前环境安装的后端及能力：

```bash
gqmr pose backends
```

执行视频推理：

```bash
gqmr pose video input.mp4 \
  --backend dog-mmpose \
  --config dog-mmpose.json \
  --batch-size 16 \
  --output input.2d.npz
```

可以通过 `--start`、`--end`、`--max-frames` 限定测试片段，通过 `--timeout`
强制终止失去响应的模型进程。

## 当前边界

本阶段结果仍是图像坐标系 2D 关键点，不能直接转换为 `AnimalMotion`。下一阶段需要：

1. 定义稳定的视觉犬类骨架并映射 AP-10K 等数据集的关键点。
2. 加入跨帧缺失点修复、左右肢身份约束和低置信度诊断。
3. 训练时序 2D 到相对 3D 提升模型。
4. 用骨长、地面和足接触优化生成视觉可信的 DOG27 动画。

绝对米制尺度、遮挡后的真实深度以及运动相机下的世界根轨迹仍属于不确定量，界面和
输出元数据必须继续明确标识。
