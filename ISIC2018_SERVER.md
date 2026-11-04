# 在服务器上运行 ISIC2018 Setting I

本适配保留 BSP 模型、损失及默认 SGD 参数，引入 FoB_SAM 的 Setting I 数据流程。
原仓库固定为提交 `40d0e23d75f81f07bb7f34cb641db51ef490d025`：
https://github.com/primebo1/FoB_SAM/tree/40d0e23d75f81f07bb7f34cb641db51ef490d025

## 1. 上传项目和数据

把整个项目上传服务器，保留以下结构。无需修改任何 Windows 绝对路径：

```text
BSP-final/
├── config.py
├── train.py
├── test.py
├── checkpoints/
│   └── deeplabv3_resnet101_coco-586e9e4e.pth
├── data/
│   ├── ISIC2018/
│   │   ├── train/
│   │   │   ├── images/ISIC_XXXXXXX.jpg
│   │   │   └── masks/ISIC_XXXXXXX_segmentation.png
│   │   ├── val/                    # 可保留，Setting I 不读取
│   │   └── test/                   # 可保留，Setting I 不读取
│   └── isic/
│       ├── class_id.csv
│       ├── prepare_setting1_dataset.py
│       └── ISIC_setting_1/
│           ├── isic_in_5_folds.json
│           └── superpixels/        # 运行生成脚本后自动创建
├── dataloaders/
└── scripts/
```

现有 `D:\Skin-DS\ISIC2018` 目录在服务器上放到项目的 `data/ISIC2018` 即可。
也支持在数据根目录直接放原始的 `ISIC2018_Task1-2_Training_Input` 和
`ISIC2018_Task1_Training_GroundTruth`，或 FoB_SAM 的 `images/gt` 布局。
设置 `ISIC_DATA_ROOT` 可覆盖根目录；相对路径均相对于项目根目录解析。
`ISIC_BASE_PATH` 可覆盖五折清单和伪标签所在目录，默认无需设置。

## 2. 检查环境和数据

在服务器激活你原来运行 BSP 的 Python/CUDA 环境，然后进入项目：

```bash
cd /你的服务器路径/BSP-final
python -c "import torch, torchvision, cv2, skimage, sacred, SimpleITK; print(torch.__version__, torchvision.__version__, skimage.__version__); print('CUDA:', torch.cuda.is_available())"
python data/isic/prepare_setting1_dataset.py --check-only
```

本适配新增流程需要 scikit-image、opencv-python、Pillow、tqdm；模型仍需要项目
README 中的 PyTorch、torchvision、Sacred、SimpleITK、SciPy 等依赖。
本机未安装或替换任何环境。原论文 README 使用 `scikit-image==0.18.3`；
若要求数值级复现，应在兼容的 Python 环境使用相同版本。生成记录会保存当前
scikit-image 版本，不把跨版本输出自动视为完全相同。

编码器权重需要自行放入上述 `checkpoints` 路径，也可设置：

```bash
export BSP_PRETRAINED_WEIGHTS=/你的权重路径/deeplabv3_resnet101_coco-586e9e4e.pth
```

只复用 FoB_SAM 的数据流程，不引入 SAM 模型，因此无需下载 SAM 权重。

## 3. 生成原论文的 SLIC 伪标签

```bash
python data/isic/prepare_setting1_dataset.py
```

使用原始分辨率 RGB，调用原论文相同的 `slic(image, n_segments=5,
compactness=15, start_label=1)`，保存 uint8 `ISIC_XXXXXXX_mask.png`。
图像和真实掩码均只读。缓存目录另存 `generation.json`，记录生成参数、库版本和
源图像 SHA256；重复运行可复用已完成的图像。源数据或库版本变化时会提示显式
使用 `--overwrite`，仅重建生成的伪标签。

五折清单覆盖 2594 张图，每折 519/519/519/519/518 张。可以一次对全部图像
生成伪标签；每次训练仅加载其余四折的伪标签，测试读取留出折的真实掩码。
无需复制三份病种图片，也无需先运行 `split.py`。该脚本保留为可选的病种目录整理工具。

## 4. 开始训练

```bash
# 默认第 1 折，50,000 次更新；先检查伪标签和编码器权重。
bash scripts/train_isic_setting1.sh

# 按顺序训练五折（每折独立重新初始化模型）。
bash scripts/train_isic_setting1.sh 1 2 3 4 5
```

也可直接使用 Python：

```bash
python train.py with isic_setting1 eval_fold=1
```

启动脚本支持 `PYTHON`、`CUDA_VISIBLE_DEVICES`、`NUM_WORKERS`、`N_STEPS`，例如：

```bash
CUDA_VISIBLE_DEVICES=0 NUM_WORKERS=8 bash scripts/train_isic_setting1.sh 1
```

权重保存在 `exps_on_ISIC2018/CoW_train_isic_setting1_cv1/<运行编号>/snapshots/50000.pth`。
每 5000 步保存一次，非整周期结束也保存最终权重。ISIC 分支按真实 DataLoader
长度循环到恰好 `n_steps` 次更新，避免直接替换加载器后实际步数变少。
保持 BSP 原有的 SGD、学习率和损失权重；这是数据协议对齐，未把 FoB 的 Adam
优化器及提示点损失移植进 BSP。

## 5. 测试

```bash
bash scripts/test_isic_setting1.sh 1 exps_on_ISIC2018/CoW_train_isic_setting1_cv1/1/snapshots/50000.pth
```

将路径中的运行编号 `1` 替换为实际训练运行编号，折号与权重必须对应。
沿用 FoB_SAM 的同病种内排序配对：索引 i 的图作为 support，下一张作为 query，
末张回到首张；support/query 互不相同。每折分别报告 melanoma、nevus、
seborrheic_keratosis 的 Dice/IoU，并报告按图像数加权的整体平均和病种宏平均。
结果写入 `results/ISIC2018/CoW_test_isic_setting1_cv1/<运行编号>/isic_results.json`，
其中记录逐图指标和支持/查询 ID；预测图存入同目录的 `interm_preds/<病种>/`。
不套用 MRI 连续切片、三段分块或 TTA；模型直接使用 BSP 的二分类预测。

## 保留与显式差异

- 训练沿用原加载器的同图双增强、颜色扰动、翻转、仿射、3×3 掩码腐蚀和归一化；
  测试沿用原来的图像缩放、二值化及配对方式。
- 默认 `isic_superpixel_mode=upstream`：保留原代码对区域编号的线性缩放，以及
  `gt[1 - fg] = 0` 的整数数组索引行为，便于核对已发布实现。
- 可通过 `python train.py with isic_setting1 isic_superpixel_mode=corrected` 显式选择
  最近邻区域编号缩放和 `gt[fg == 0] = 0` 布尔索引。此模式是修正版实验，不能声称
  与上游加载器逐像素完全相同；运行配置和测试结果均记录模式。
- 常量图像的 min-max 分母增加保护；无效路径、空数据、非法折号提前报错。
- `transform.py` 只补充旧版 torchvision 的 `affine` 参数兼容分支，保留相同插值。
- 五折 JSON 和类别 CSV 保持原始字节；上游原文件及修改前的 BSP 入口分别保存在
  `output/isic2018_integration/upstream/` 和 `output/isic2018_integration/before/`。
  FoB_SAM 许可证保留于 `data/isic/FoB_SAM_LICENSE.txt`。
