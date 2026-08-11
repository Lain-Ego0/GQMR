# dog-27 骨架与历史输入协议 v1

> 状态：已实现  
> 日期：2026-08-11

dog-27 将 AI4Animation 历史犬类文本动作的 81 个逗号分隔浮点数解释为 27 个 xyz 点。该协议只固化从历史导入代码和本地数据验证中能够确认的语义；源文件不含关键点名。

## 索引表

| 索引 | 名称 | 父点 | 说明 |
|---:|---|---|---|
| 0 | `pelvis` | — | 已确认核心根标志 |
| 1 | `pelvis_duplicate` | `pelvis` | 已检查历史片段中与索引 0 完全相同 |
| 2 | `spine` | `pelvis` | 描述性链名 |
| 3 | `neck` | `spine` | 已确认核心根标志 |
| 4 | `head` | `neck` | 描述性链名 |
| 5 | `muzzle` | `head` | 描述性链名 |
| 6–10 | `left_shoulder` → `left_front_toe` | 前肢链 | 肩 6 和趾 10 已确认，中间名为描述性语义 |
| 11–15 | `right_shoulder` → `right_front_toe` | 前肢链 | 肩 11 和趾 15 已确认，中间名为描述性语义 |
| 16–19 | `left_hip` → `left_hind_toe` | 后肢链 | 髋 16 和趾 19 已确认，中间名为描述性语义 |
| 20–23 | `right_hip` → `right_hind_toe` | 后肢链 | 髋 20 和趾 23 已确认，中间名为描述性语义 |
| 24–26 | `tail_base` → `tail_tip` | 尾部链 | 描述性链名 |

完整的机器可读拓扑、左右对称对、根标志和 `FL/FR/RL/RR` 肢体链位于 `src/gqmr/configs/skeletons/dog-27.yaml`。配置加载后使用规范 JSON 计算 SHA-256，并写入 `AnimalMotion.metadata.skeleton_sha256`。

## 坐标转换

历史文本点先执行：

```text
Rz(0.47 * pi) @ Rx(0.5 * pi)
```

转换后进入 GQMR 右手世界坐标：X 向前、Y 向左、Z 向上。读取器不再乘任何特定机器人比例。

## 安全与许可证

- 每行必须恰好包含 81 个有限数值；空文件、NaN/Infinity、非数值和非法帧范围立即拒绝。
- 单文件安全上限为 2 GiB。
- 历史数据保持 `CC-BY-NC-4.0` 来源标记，不进入 GQMR 发布包或正式测试产物。
- GQMR 的 walk/trot/pace/turn 合成动作由代码独立生成，来源标记为 MIT，用于 CI 和演示。
