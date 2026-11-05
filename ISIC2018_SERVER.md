# ISIC2018 Setting I 外部路径运行

本版保留 BSP 模型和 FoB_SAM Setting I 数据协议，仅将所有数据路径改为命令传入。
预处理源码位于 `scripts/isic`，数据和协议文件不放在代码目录。
上游固定提交为 `40d0e23d75f81f07bb7f34cb641db51ef490d025`。

## 外部目录

```text
/datasets/ISIC2018/train/images/ISIC_XXXXXXX.jpg
/datasets/ISIC2018/train/masks/ISIC_XXXXXXX_segmentation.png
/datasets/ISIC_setting_1/isic_in_5_folds.json
/datasets/ISIC_setting_1/superpixels/                  生成的 SLIC 伪标签
/pretrained/deeplabv3_resnet101_coco-586e9e4e.pth
```

图像还支持原始 ISIC Training_Input/Training_GroundTruth 目录名，或 images/gt 布局。
五折 JSON 需要自行从原数据协议准备到外部协议目录；可复用旧版已有文件，
不要更改类别或折划分。当前 Setting I 直接从 JSON 读取类别，不要求分类 CSV。

## 检查与生成伪标签

```bash
python scripts/isic/prepare_setting1_dataset.py --data-root /datasets/ISIC2018 --base-path /datasets/ISIC_setting_1 --check-only
python scripts/isic/prepare_setting1_dataset.py --data-root /datasets/ISIC2018 --base-path /datasets/ISIC_setting_1
```

图像和真实掩码只读，生成的伪标签和 `generation.json` 写到外部协议目录。
沿用原 SLIC 参数 `n_segments=5, compactness=15, start_label=1`。
已有缓存的版本、参数或源文件变化时，仍要求显式 `--overwrite` 才能重建。

可选的按病种整理脚本需要额外 CSV，并使用独立输出目录：

```bash
python scripts/isic/split.py --data-root /datasets/ISIC2018 --class-csv /datasets/metadata/class_id.csv --output-root /datasets/ISIC_categorized
```

## 训练与测试

```bash
python train.py with isic_setting1 ablation_variant=full eval_fold=1 seed=1234 data_root=/datasets/ISIC2018 isic_setting_1_base_path=/datasets/ISIC_setting_1 pretrained_weights=/pretrained/deeplabv3_resnet101_coco-586e9e4e.pth path.log_dir=/experiments/BSP
python test.py with isic_setting1 mode=test ablation_variant=full eval_fold=1 seed=1234 data_root=/datasets/ISIC2018 isic_setting_1_base_path=/datasets/ISIC_setting_1 reload_model_path=/experiments/BSP/CoW_train_isic_setting1_cv1_full_seed1234/1/snapshots/50000.pth path.log_dir=/experiments/BSP-results
```

运行编号和权重路径按实际训练输出调整；保留同目录 `ablation.json`。
辅助启动器同样支持 `--data-root`、`--pretrained-weights` 和 `--isic-base-path`。
折号为 1 至 5。测试沿用同病种内排序配对，不添加 MRI 分块或 TTA。
结果仍输出 `isic_results.json`，分别报告图像加权平均与类别宏平均。
此图像折协议不等同于未见类别评测。

默认 `isic_superpixel_mode=upstream` 沿用原标签处理；`corrected` 是显式协议变体。
本版不需要 SAM 权重，也不改用 SAM 模型或其优化器。
FoB_SAM 许可保留于 `LICENSES/FoB_SAM_LICENSE.txt`。
