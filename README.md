# BSP规范版

本版分支为 `bsp-standard`，命名整理基于 `ablation-external-paths`。
项目只包含源码、脚本、说明和许可，
数据集、划分清单、预训练权重与实验输出均独立存放。
训练命令用 `data_root=... pretrained_weights=...` 指定资源，无需修改源码。

旧版五组消融固定保存在标签
[`ablation-before-external-paths`](https://github.com/aihongjian777-crypto/BSP-cvpr/tree/ablation-before-external-paths)，
原 `ablation` 分支和正常版 `main` 仍保留。

## 开始实验

模块名称与函数对应关系见 [规范版说明](BSP_STANDARD.md)。
来源信息见 [NOTICE](NOTICE.md)，原许可文件保留。

```bash
git clone --branch bsp-standard https://github.com/aihongjian777-crypto/BSP-cvpr.git BSP-ablation-external
cd BSP-ablation-external
python train.py with ablation_variant=sepc_only mode=train dataset=CHAOST2 eval_fold=2 n_steps=50000 max_iters_per_load=5000 lr_step_gamma=0.9 use_gt=False exclude_label=None n_way=1 n_shot=1 n_query=1 batch_size=1 num_workers=4 seed=1234 save_snapshot_every=5000 data_root=/datasets/CHAOST2 pretrained_weights=/pretrained/deeplabv3_resnet101_coco-586e9e4e.pth path.log_dir=/experiments/BSP
```

把路径替换成实际位置。`ablation_variant` 可取 `full`、`sepc_only`、`wo_sepc`、
`wo_qrr`、`wo_swpa`。正式比较时，每组使用相同数据协议、预训练来源和训练预算，独立训练。

- [外部路径与完整训练测试命令](EXTERNAL_PATHS.md)
- [模块关闭定义与五组实验表](ABLATION.md)
- [ISIC Setting I 外部数据准备](ISIC2018_SERVER.md)
- [版本保存与恢复](VERSION_CONTROL.md)

## 环境和数据

沿用原项目可运行的 Python/CUDA 环境，包括 PyTorch、torchvision、numpy、
OpenCV、Sacred、SimpleITK、SciPy、scikit-image、Pillow、tqdm 等。
本次修改不安装或更换训练依赖，也不修改网络计算及损失系数。
MRI/CT 数据根目录仍需保持原加载器要求的图像、标注和伪标签布局。
ISIC 预处理源码位于 `scripts/isic`，数据与元数据目录通过参数指定。
模型检查点旁保存 `ablation.json`，测试必须使用相同实验组、折号和种子。

## 来源说明

本项目包含对 CoW 和 DIFD 的继承与适配：

- [CoW](https://github.com/jcjiang99/CoW)
- [DIFD](https://github.com/zmcheng9/DIFD)
- 原数据流程参考 [SSL-ALPNet](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation)
  和 [ADNet](https://github.com/sha168/ADNet)。
- ISIC Setting I 数据流程适配自 FoB_SAM 固定提交
  `40d0e23d75f81f07bb7f34cb641db51ef490d025`，许可保留在
  [LICENSES/FoB_SAM_LICENSE.txt](LICENSES/FoB_SAM_LICENSE.txt)。
