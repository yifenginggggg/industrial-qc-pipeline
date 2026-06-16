# 工业产品视觉质检流水线

这是一个面向 `KolektorSDD2` 数据集的工业表面缺陷视觉质检项目，采用 `YOLO11` 完成缺陷二分类检测，采用 `U-Net` 完成像素级缺陷分割。

本仓库实现了一套完整的多阶段工业视觉检测流程：

1. `YOLO11` 先判断图像是否存在缺陷。
2. 如果检测到缺陷，则根据检测框生成一个或多个 ROI 区域。
3. `U-Net` 在原图 ROI 裁剪区域上执行缺陷分割。
4. 分割结果经过后处理、缺陷量化、置信度融合和人工复检规则判断。
5. 最终输出结构化 JSON、掩码图、叠加可视化图，便于分析和展示。

## 功能概览

- 两阶段检测流程：`YOLO11 -> ROI -> U-Net`
- 检测门控分割：只有检测到缺陷时才进入分割阶段
- 基于原图 ROI 裁剪的分割推理，而不是对 YOLO 生成的掩码图做分割
- 支持为 ROI 场景重新构建分割训练集并重训练 U-Net
- 支持分割后处理：
  - 开运算
  - 闭运算
  - 小连通域过滤
- 支持缺陷量化：
  - 缺陷数量
  - 总缺陷面积
  - 面积占比
  - 单缺陷包围框
  - 质心坐标
  - 长宽估计
- 支持 YOLO 置信度与分割置信度融合
- 支持人工复检机制与复检原因输出
- 支持多缺陷统计
- 支持 YOLO 与 U-Net 训练接入 W&B
- 支持 ROI 与非 ROI 的消融对比评估

## 流程说明

```text
输入图像
  -> YOLO11 缺陷检测
  -> 若未检测到缺陷：直接输出 no_defect
  -> 若检测到缺陷：
       根据检测框扩展 ROI
       从原图裁剪 ROI
       在每个 ROI 上运行 U-Net
       将 ROI 概率图回填到整图
       做掩码后处理
       汇总缺陷统计信息
       导出 JSON、掩码图和可视化结果
```

## 仓库结构

```text
industrial_qc_pipeline/
  data/
    raw/                   # 原始数据集或 smoke 数据
    processed/             # YOLO / U-Net / ROI 转换结果与清单文件
  outputs/                 # 训练结果、权重、推理输出、评估结果
  requirements/
  scripts/
    prepare_dataset.py
    prepare_roi_dataset.py
    train_yolo.py
    train_unet.py
    predict_pipeline.py
    evaluate_roi_vs_full.py
    setup_env.sh
    with_dns.sh
  src/industrial_qc/
  tests/
```

## 当前已实现能力

当前代码已经实现了以下关键模块：

- ROI 框生成与合并：[src/industrial_qc/pipeline.py](src/industrial_qc/pipeline.py#L56)
- 分割后处理：[src/industrial_qc/pipeline.py](src/industrial_qc/pipeline.py#L69)
- 置信度融合、缺陷量化、人工复检、多缺陷汇总：[src/industrial_qc/pipeline.py](src/industrial_qc/pipeline.py#L91)
- 端到端推理入口：[scripts/predict_pipeline.py](scripts/predict_pipeline.py#L31)
- ROI 数据集构建：[scripts/prepare_roi_dataset.py](scripts/prepare_roi_dataset.py)
- ROI 与整图分割对比评估：[scripts/evaluate_roi_vs_full.py](scripts/evaluate_roi_vs_full.py)

## 环境配置

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/train.txt
```

也可以使用辅助脚本：

```bash
bash scripts/setup_env.sh
```

如果是离线服务器并且已经准备了本地 wheel 包目录：

```bash
bash scripts/setup_env.sh /path/to/wheelhouse
```

如果远程服务器本地 DNS 有问题，但外网仍可访问，可以通过下面的方式运行命令：

```bash
bash scripts/with_dns.sh python -c "import socket; print(socket.gethostbyname('wandb.ai'))"
bash scripts/with_dns.sh bash scripts/setup_env.sh
```

## 数据集

本项目使用的数据集：

- [Kolektor Surface-Defect Dataset 2](https://datasetninja.com/kolektor-surface-defect-dataset-2)

原始数据默认放置位置：

```text
data/raw/KolektorSDD2/
```

数据准备脚本会将其转换为：

- YOLO 检测格式：`data/processed/yolo_ksdd2/`
- U-Net 分割清单：`data/processed/unet_ksdd2/`

## 数据准备

准备真实数据集：

```bash
python scripts/prepare_dataset.py \
  --raw-dir data/raw/KolektorSDD2 \
  --output-dir data/processed
```

如果只想先验证通路，可以生成一个很小的 smoke 数据集：

```bash
python scripts/prepare_dataset.py --smoke-run
```

## 模型训练

### 训练 YOLO11

```bash
python scripts/train_yolo.py \
  --data data/processed/yolo_ksdd2/dataset.yaml \
  --weights yolo11n.pt \
  --epochs 30 \
  --batch 16 \
  --imgsz 640 \
  --workers 4 \
  --device 0 \
  --run-name ksdd2_yolo11n_e30 \
  --wandb-key "$WANDB_API_KEY"
```

输出位置：

- 权重保存在 `outputs/yolo/train/<run-name>/weights/`

### 训练整图版 U-Net

```bash
python scripts/train_unet.py \
  --train-manifest data/processed/unet_ksdd2/train.txt \
  --val-manifest data/processed/unet_ksdd2/val.txt \
  --epochs 30 \
  --batch-size 16 \
  --image-size 256 \
  --num-workers 4 \
  --device cuda:0 \
  --run-name ksdd2_unet_e30 \
  --wandb-key "$WANDB_API_KEY"
```

输出位置：

- 最优权重保存在 `outputs/unet/<run-name>/best.pt`

### 构建 ROI 分割训练集

这一步会把整图分割数据，转换为基于缺陷区域的 ROI 裁剪样本。负样本保留为整图。

```bash
python scripts/prepare_roi_dataset.py \
  --train-manifest data/processed/unet_ksdd2/train.txt \
  --val-manifest data/processed/unet_ksdd2/val.txt \
  --test-manifest data/processed/unet_ksdd2/test.txt \
  --output-dir data/processed/unet_ksdd2_roi \
  --roi-padding 0.15 \
  --roi-min-size 32
```

### 训练 ROI 重训版 U-Net

```bash
python scripts/train_unet.py \
  --train-manifest data/processed/unet_ksdd2_roi/train.txt \
  --val-manifest data/processed/unet_ksdd2_roi/val.txt \
  --epochs 30 \
  --batch-size 16 \
  --image-size 256 \
  --num-workers 4 \
  --device cuda:0 \
  --run-name ksdd2_unet_roi_e30 \
  --wandb-key "$WANDB_API_KEY"
```

## 推理入口

单张图像的完整 pipeline 入口如下：

```bash
python scripts/predict_pipeline.py \
  --image path/to/image.png \
  --yolo-weights outputs/yolo/train/ksdd2_yolo11n_e30/weights/best.pt \
  --unet-weights outputs/unet/ksdd2_unet_roi_e30/best.pt \
  --conf-threshold 0.10 \
  --image-size 256 \
  --mask-threshold 0.5 \
  --roi-padding 0.15 \
  --roi-min-size 32 \
  --post-min-area 25 \
  --opening-iterations 1 \
  --closing-iterations 1 \
  --fuse-yolo-weight 0.6 \
  --review-threshold 0.45 \
  --device cuda:0 \
  --output-dir outputs/predictions/example_case
```

### 当前 pipeline 的核心逻辑

- YOLO 先决定是否进入分割阶段。
- 如果 YOLO 没有检测到缺陷，则直接输出 `no_defect`。
- 如果 YOLO 检测到缺陷，则根据检测框生成 ROI。
- U-Net 对原图裁剪得到的 ROI 区域做分割。
- ROI 概率图会被映射回原图尺寸后再统一后处理。
- 本项目不是对 YOLO 输出的 mask 图做分割，而是对原图的 ROI 裁剪区域做分割。

## 推理输出内容

`predict_pipeline.py` 会在输出目录下生成：

- `detections.png`
- `roi_overlay.png`
- `probability_map.png`
- `raw_mask.png`
- `mask.png`
- `segmentation_overlay.png`
- `result.json`

其中 `result.json` 会包含类似下面的字段：

```json
{
  "decision": "defect",
  "review_status": "manual_review",
  "review_reasons": ["low_fused_confidence"],
  "fused_confidence": 0.62,
  "has_multiple_defects": true,
  "mask_summary": {
    "defect_count": 2,
    "total_defect_area_pixels": 1843,
    "total_defect_area_ratio": 0.021
  },
  "defects": [
    {
      "defect_id": 1,
      "bbox_xyxy": [42, 105, 89, 166],
      "centroid_xy": [65.4, 133.2],
      "area_pixels": 920,
      "length_pixels": 61,
      "width_pixels": 47,
      "yolo_confidence": 0.81,
      "segmentation_confidence": 0.73,
      "fused_confidence": 0.78,
      "review_reasons": []
    }
  ]
}
```

## ROI 消融对比评估

如果想比较以下三种策略：

- 整图版 U-Net
- 只做 ROI 推理但不重训
- ROI 重训练后的 U-Net

可以运行：

```bash
python scripts/evaluate_roi_vs_full.py \
  --manifest data/processed/unet_ksdd2/test.txt \
  --yolo-weights outputs/yolo/train/ksdd2_yolo11n_e30/weights/best.pt \
  --full-unet-weights outputs/unet/ksdd2_unet_e30/best.pt \
  --roi-unet-weights outputs/unet/ksdd2_unet_roi_e30/best.pt \
  --legacy-roi-unet-weights outputs/unet/ksdd2_unet_e30/best.pt \
  --conf-threshold 0.10 \
  --device cuda:1 \
  --output outputs/roi_retrain_comparison.json
```

输出位置：

- 对比结果保存在 `outputs/roi_retrain_comparison.json`

## 最近一次已验证实验结果

最近一次 ROI 重训练对比评估运行于 `2026年6月15日`，测试集统计如下：

- 测试集总数：`1004`
- 正样本数：`110`
- 在 YOLO 置信度阈值 `0.10` 下被检测到的正样本数：`89`

### 整图版 U-Net vs ROI 重训版 U-Net

在全部正样本上：

- 整图版 U-Net：`Dice = 0.5727`，`IoU = 0.4745`
- ROI 重训版 U-Net：`Dice = 0.6013`，`IoU = 0.4982`

在被 YOLO 成功检出的正样本上：

- 整图版 U-Net：`Dice = 0.7079`，`IoU = 0.5864`
- ROI 重训版 U-Net：`Dice = 0.7431`，`IoU = 0.6158`

按样本逐一比较：

- ROI 重训版更好：`52` 张
- 整图版更好：`36` 张
- 基本持平：`22` 张

### 只做 ROI 推理但不重训的旧方案

在全部正样本上：

- 整图版 U-Net：`Dice = 0.5727`
- 未重训 ROI 推理：`Dice = 0.5414`

这个结果说明了一个很重要的结论：

- 仅仅做 ROI 裁剪并不足以带来提升
- 必须基于 ROI 数据重新训练 U-Net，ROI 方案才真正有效

## W&B

可以通过环境变量 `WANDB_API_KEY` 或训练脚本参数 `--wandb-key` 接入 W&B。

- YOLO 训练在登录成功后会启用 Ultralytics 的 W&B 记录
- U-Net 训练会记录每个 epoch 的 `train_loss`、`val_loss`、`precision`、`recall`、`dice`、`iou`

## 测试

运行本地测试：

```bash
pytest tests -q
```

当前已验证状态：

- `13 passed`

## 说明

- YOLO 与 U-Net 的 train / val / test 划分来自同一套数据准备流程，因此划分是对齐的。
- ROI U-Net 的训练集来自同一原始划分，只是在分割阶段把样本转换为了 ROI 裁剪形式。
- 该仓库既适合远程服务器训练，也适合公开展示和课程作业提交。
