# 本地第三方动作数据

第三方真实动物动作统一放在仓库根目录的 `external_data/`。该目录已被
`.gitignore` 排除，不会进入 Git、wheel、sdist 或 portable 工程。仓库只记录来源、
许可限制和本地安装方式。

## 当前本地状态

运行：

```bash
./tools/install_external_motion_data.sh
```

脚本会把仓库已有的 AI4Animation dog-27 动作复制到：

```text
external_data/ai4animation-dog27/
```

该数据共 15 个片段、13,772 帧，包括 walk、run、turn、trot 和 pace。动作数据采用
CC BY-NC 4.0，只能用于符合许可证的非商业用途，不能随 GQMR 的 MIT 发布物分发。

来源：<https://github.com/sebastianstarke/AI4Animation>

## 推荐外部数据集

### PFERD

- 官方数据：<https://doi.org/10.7910/DVN/2EXONE>
- 论文：<https://doi.org/10.1038/s41597-024-03312-1>
- 建议目录：`external_data/pferd/`
- 优先下载 `C3D_DATA`、`FBX_DATA` 和 `MODEL_DATA`，无需下载全部原始视频。
- 动作包括走、快步、跑步、后退、侧移、转圈、步态切换、后腿站立、踢腿、跳跃、坐下和躺下。
- Dataverse 当前标注为自定义数据条款；下载和再分发前必须阅读其最新条款。

Harvard Dataverse 当前可能要求浏览器完成 WAF 验证。脚本不会规避该验证；在浏览器
下载后将文件放入上述目录即可。

### RGBD-Dog

- 申请与说明：<https://github.com/CAMERA-Bath/RGBD-Dog>
- 建议目录：`external_data/rgbd-dog/`
- 提供 3D marker、BVH、骨骼旋转和同步视频，包含行走、小跑、跨杆、跳杆及上下约
  30 cm 平台。
- 必须签署 University of Bath 数据协议；仅限学术研究、禁止商业使用和第三方转发。

因此不能由 GQMR 自动下载，也不能提交到仓库。取得授权后，将收到的数据放入建议
目录即可。

### AcinoSet

- 项目：<https://github.com/African-Robotics-Unit/AcinoSet>
- 建议目录：`external_data/acinoset/`
- 提供野外猎豹高速多视角视频、2D keypoint、相机标定和 FTE 3D 轨迹。
- 上游 GitHub 仓库未提供明确的数据许可证文件；在作者确认许可前，仅用于本地研究，
  不随 GQMR 发布或转发。

## 验证本地文件

安装脚本会生成 `external_data/LOCAL_DATASETS.json`，其中记录本地文件的 SHA-256、
大小、帧数、来源和许可状态。可重复运行脚本刷新并校验 AI4Animation 副本。

这些数据不属于 GQMR 的 MIT 许可内容。使用者须自行遵守每个数据集的最新条款、
署名和用途限制。
