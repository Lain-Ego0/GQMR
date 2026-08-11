# GQMR 展示与验收说明

> 验收日期：2026-08-11  
> 版本：`0.0.1`  
> 结论：当前产品闭环可展示；正式 `v1.0.0` 仍保留 30 分钟流式、真实桌面/GPU 和签名发布审计。

## 1. 五分钟展示路径

```bash
uv sync --frozen
uv run gqmr assets install unitree-go2
uv run gqmr synthetic trot --duration 0.5 --fps 20 --output trot.animal.npz
uv run gqmr retarget trot.animal.npz --robot unitree-go2 \
  --mode high-quality --output trot.go2.npz
uv run gqmr play trot.go2.npz --robot unitree-go2 --dynamics
uv run gqmr export trot.go2.npz --robot unitree-go2 \
  --format isaaclab_amp_v232 --output trot.amp.npz
uv run gqmr gui
```

无需现场生成时，可直接使用 [`examples/demo`](../examples/demo) 中已提交的 MIT 合成演示资产和 portable 工程。

## 2. 本轮验收证据

| 项目 | 结果 |
|---|---|
| 冻结环境完整测试 | `103 passed`，Go2/B2 真实资产测试无跳过 |
| 静态编译 | `src`、`tests`、`tools` 全部通过 |
| 高质量演示重定向 | 121/121 有效帧，足端目标 RMSE `0.000196 m`，接触滑移 `0.0133 m/s` |
| 周期与高度 | 根高首尾 `0.270 m`；同相位关节最大误差 `0.0021 rad` |
| 动力学演示 | 1000 步 / 2 秒完成，无跌倒时间；结果明确标记为诊断 PD 跟踪 |
| 导出 | DeepMimic JSON、Isaac Lab AMP NPZ 均成功 |
| portable 工程 | 两个资源均嵌入；原位保存、重复导入去重、再次打包通过 |
| GUI | Go2/B2 MuJoCo mesh、时间轴姿态、拖动旋转、滚轮缩放及离屏事件循环通过 |
| 200 Hz 流式 | 60 秒、12000/12000 帧、0 GAP、`199.93 Hz` |
| 峰值内存 | 60 秒基线最大 RSS `372.5 MiB` |

## 3. 发布物

wheel 和 sdist 均不包含 `motion_imitation`、`*_joint_pos.txt` 或 CI/CD 配置。CycloneDX 1.5 SBOM 由 clean wheel 环境生成，包含 GQMR 根组件和 22 个运行时依赖组件。最终归档的 SHA-256 在每次重建后的验收输出中记录，避免在被哈希的 sdist 文档中写入自引用摘要。

## 4. 当前边界

- 当前阶段按产品决策不启用 CI/CD，自动流水线配置已从仓库移除；以上检查通过本地冻结验收执行。
- 60 秒测试是适度长时基线，不替代正式发布前的 30 分钟、200 Hz 审计。
- 真实桌面 EGL/GPU、长时 GUI 取消和第三方 DeepLabCut/SLEAP 数据仍需在参考机执行。
- 单目三维提升保持实验定位，不作为可靠通用能力声明。
