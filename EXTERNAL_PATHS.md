# 通过训练命令指定外部数据和权重

本版分支为 `ablation-external-paths`。五组模块开关与旧消融版相同，
本次改变资源路径及工程目录，不改变模型计算和训练损失。
项目中不包含数据集、数据划分文件、预训练权重或实验输出。

旧版消融已固定保存为 GitHub 标签 `ablation-before-external-paths`（提交 `03a5ae5`），
原 `ablation` 分支也保留。完整正常版仍在 `main`。

## 获取新版本

```bash
git clone --branch ablation-external-paths https://github.com/aihongjian777-crypto/BSP-cvpr.git BSP-ablation-external
cd BSP-ablation-external
```

建议存储结构：

```text
/workspace/BSP-ablation-external/                 源码、脚本、文档、许可
/datasets/CHAOST2/                               原有数据目录及内部结构
/pretrained/deeplabv3_resnet101_coco-586e9e4e.pth  编码器预训练权重
/experiments/BSP/                                训练产生的日志和模型
```

数据目录内部布局沿用原数据加载器的要求。`data_root` 指向原来配置中 `data_dir`
对应的那一层目录；不要为了配合本版移动、重新划分或复制原数据。
预处理 Python 脚本在 `scripts/isic`，上游许可在 `LICENSES`。
ResNet 网络源码仍然保留在 `models/encoder.py`。

## 直接使用训练命令

以第 2 折、种子 1234、仅 SEPC 为例，下列命令与之前的训练参数一致，
增加了 `data_root` 和 `pretrained_weights`。把示例路径替换成机器上的实际路径：

```bash
python train.py with ablation_variant=sepc_only mode=train dataset=CHAOST2 eval_fold=2 n_steps=50000 max_iters_per_load=5000 lr_step_gamma=0.9 use_gt=False exclude_label=None n_way=1 n_shot=1 n_query=1 batch_size=1 num_workers=4 seed=1234 save_snapshot_every=5000 data_root=/datasets/CHAOST2 pretrained_weights=/pretrained/deeplabv3_resnet101_coco-586e9e4e.pth path.log_dir=/experiments/BSP
```

通过修改 `ablation_variant` 选择实验组：

| 值 | SEPC | QRR | SWPA |
|---|---|---|---|
| full | ✓ | ✓ | ✓ |
| sepc_only | ✓ | — | — |
| wo_sepc | — | ✓ | ✓ |
| wo_qrr | ✓ | — | ✓ |
| wo_swpa | ✓ | ✓ | — |

路径包含空格时，应把整个赋值参数用引号括起，例如：

```bash
python train.py with ablation_variant=full "data_root=D:/Medical Data/CHAOST2" "pretrained_weights=D:/Model Weights/resnet101.pth"
```

使用与原实验相同来源的预训练权重，避免把 ImageNet ResNet 与 DeepLab COCO 权重
在不同实验组之间混用。训练不再接受 `deeplabv3`、`resnet101` 作为文件名别名，
也不依赖 `BSP_PRETRAINED_WEIGHTS` 环境变量。未提供数据或权重、路径不存在时，
会在进入 CUDA/构建模型之前报出具体路径错误，不会悄悄随机初始化训练。

## 使用辅助启动器

```bash
python scripts/run_ablation.py train --variant sepc_only --dataset CHAOST2 --fold 2 --seed 1234 --workers 4 --data-root /datasets/CHAOST2 --pretrained-weights /pretrained/deeplabv3_resnet101_coco-586e9e4e.pth
```

`--variant all` 顺序训练五组；`--dry-run` 只显示命令。
辅助启动器仍使用原 `config.py` 默认训练参数；需要完整控制参数时使用上面的 Sacred 命令。

## 测试时使用训练好的权重

测试无需再提供编码器预训练权重，但必须指定数据和该实验组训练出的模型：

```bash
python test.py with ablation_variant=sepc_only mode=test dataset=CHAOST2 eval_fold=2 seed=1234 supp_idx=1 num_workers=4 data_root=/datasets/CHAOST2 reload_model_path=/experiments/BSP/CoW_train_CHAOST2_cv2_sepc_only_seed1234/1/snapshots/50000.pth path.log_dir=/experiments/BSP-results
```

实际运行编号可能不是 `1`，请选对应运行生成的权重。权重旁的 `ablation.json`
也应保留，测试时会核对实验组、数据集、折号和种子。

## ISIC 额外路径

ISIC 还需外部协议目录，存放五折 JSON 与 `superpixels/`，示例：

```bash
python scripts/isic/prepare_setting1_dataset.py --data-root /datasets/ISIC2018 --base-path /datasets/ISIC_setting_1 --check-only
python train.py with isic_setting1 ablation_variant=full eval_fold=1 data_root=/datasets/ISIC2018 isic_setting_1_base_path=/datasets/ISIC_setting_1 pretrained_weights=/pretrained/deeplabv3_resnet101_coco-586e9e4e.pth
```

Setting I 训练直接从五折 JSON 读取类别；`class_id.csv` 仅为可选分类整理脚本所需，
使用该脚本时通过 `--class-csv` 指定，详见 `ISIC2018_SERVER.md`。

## 回到已保存的旧版

在新目录读取旧版，不会改动当前工作目录：

```bash
git clone --branch ablation-before-external-paths https://github.com/aihongjian777-crypto/BSP-cvpr.git BSP-ablation-saved
```

标签检出为固定提交，便于复现。模型结果仍需各组独立训练后获得；
代码检查与路径检查不等于已经完成训练实验。
