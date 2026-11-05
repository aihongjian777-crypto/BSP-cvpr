# BSP 三模块消融实验

本分支从完整模型提交 `ebab703` 创建，用五组重新训练的模型验证 SEPC、QRR、SWPA。
正常版保留在 `main`，本分支为 `ablation`。下表只定义实验，未填写或虚构实验分数。

| 配置 | SEPC | QRR | SWPA | 各类别 Dice | Avg |
|---|---|---|---|---|---|
| full | ✓ | ✓ | ✓ | 待训练测试 | 待训练测试 |
| wo_sepc | — | ✓ | ✓ | 待训练测试 | 待训练测试 |
| wo_qrr | ✓ | — | ✓ | 待训练测试 | 待训练测试 |
| wo_swpa | ✓ | ✓ | — | 待训练测试 | 待训练测试 |
| sepc_only | ✓ | — | — | 待训练测试 | 待训练测试 |

## 模块关闭的准确含义

- `wo_sepc`：取消区域库参与支持预测，改为全局前景、背景上下文；
  将 FN/FP 位置采样替换为同类均匀随机采样。保留基础 MLP、边界和全局原型，
  以及全局上下文支持预测的 SSP 监督。前景保持 50 个基础原型加 50 个随机像素，
  背景保持 500 加 100，类别为空时按原有空类规则回退。
  这是对 **SEPC 的区域上下文与误差引导机制** 的替换消融，
  不是移除 SEPC 中全部共用原型构建操作。论文表注应写明此定义。
  支持侧、训练反向分支均使用这一规则。
- `wo_qrr`：跳过查询响应池化、融合系数和残差更新，归一化的支持原型直接进入聚合。
- `wo_swpa`：取消位置相关的原型 Softmax 权重，使用全部非全局原型的均匀平均；
  同时均匀平均缩放余弦响应。保留查询特征、`2C+1` 解码器输入及分类层。
- `full`：三个模块全部开启，保留原始完整模型运算、损失系数及参数结构。
- `sepc_only`：SEPC 正常开启，同时关闭 QRR、SWPA；原型不做查询残差更新，
  后续使用均匀聚合。保留分割解码器和原有训练损失，作为可训练的基础对照。

边界损失、SSP、反向对齐和原型判别项仍使用原训练流程。
继承的空查询种子 `continue` 行为没有在此分支中另行修正：备用种子仍为空时，
该任务的 SSP 项也会被跳过。应记录这一实现事实，不应声称只跳过对齐和判别项。
CT 专用的 `train_ct.py/test_ct.py` 未接入开关；本分支使用 `train.py/test.py`。

## 获取与环境

新目录克隆：

```bash
git clone --branch ablation https://github.com/aihongjian777-crypto/BSP-cvpr.git BSP-ablation
cd BSP-ablation
```

沿用原项目的 PyTorch、torchvision、Sacred、OpenCV、SimpleITK 等运行环境。
数据、权重和实验结果不在仓库中，需要自行准备。编码器预训练权重可设置：

```bash
export BSP_PRETRAINED_WEIGHTS=/你的权重目录/deeplabv3_resnet101_coco-586e9e4e.pth
```

本地独立工作目录默认没有数据和权重。通过 `--data-root` 指向原数据；
ISIC 还需要准备 `data/isic/class_id.csv`、五折 JSON 和伪标签，详见 `ISIC2018_SERVER.md`。

## 训练五组

以下以 CHAOST2 第 0 折、随机种子 2025 为例。五组会顺序训练，每组独立初始化：

```bash
python scripts/run_ablation.py train --variant all --dataset CHAOST2 --fold 0 --seed 2025 --data-root /数据/CHAOST2
```

先查看将执行的命令：

```bash
python scripts/run_ablation.py train --variant all --dataset CHAOST2 --fold 0 --dry-run
```

单独训练去 QRR：

```bash
python scripts/run_ablation.py train --variant wo_qrr --dataset CHAOST2 --fold 0 --seed 2025 --data-root /数据/CHAOST2
```

默认训练 50000 步；`--steps`、`--workers` 可调整。该启动器沿用 `config.py` 的
优化器和学习率调度默认值，不等同于旧 shell 脚本中可能覆盖过的参数。
需要自定义完整设置时，可直接使用 Sacred 命令，且五组设置必须一致：

```bash
python train.py with ablation_wo_qrr dataset=CHAOST2 eval_fold=0 seed=2025
```

仅 SEPC 可直接通过命令选择，不需要手工修改模型代码：

```bash
python scripts/run_ablation.py train --variant sepc_only --dataset CHAOST2 --fold 0 --seed 2025 --data-root /数据/CHAOST2
# 等价的 Sacred 配置名，也可写成 ablation_variant=sepc_only
python train.py with ablation_sepc_only dataset=CHAOST2 eval_fold=0 seed=2025
```

配置名统一决定三个模块的开关；测试时也必须选择对应配置，并提供该组自己的权重。

MRI/CT 折号为 0 至 4，ISIC 为 1 至 5。ISIC 示例：

```bash
python scripts/run_ablation.py train --variant all --dataset isic --fold 1 --data-root /数据/ISIC2018
```

结果目录包含配置、种子和折号，例如：

```text
exps_on_ABLATION/CoW_train_CHAOST2_cv0_wo_qrr_seed2025/<运行编号>/snapshots/
  50000.pth
  ablation.json
```

测试时必须使用该组自己训练的权重。`ablation.json` 与权重一起保留，
测试会核对模块、数据集、折号和训练种子。没有元数据的旧权重仅允许作为 `full`
运行，并提示其来源未验证；不能把正常版权重临时关闭模块当作正式消融结果。

## 测试与表格

```bash
python scripts/run_ablation.py test --variant wo_qrr --dataset CHAOST2 --fold 0 --seed 2025 --support-index 1 --checkpoint /该组/snapshots/50000.pth --data-root /数据/CHAOST2
```

其余四组同理，分别指定对应的配置和权重。
MRI/CT 输出 `ablation_results.json`，ISIC 保留 `isic_results.json` 及其原评价协议。
MRI/CT 的结果可汇总为模块勾选表 CSV：

```bash
python scripts/summarize_ablation.py /完整组/ablation_results.json /去SEPC组/ablation_results.json /去QRR组/ablation_results.json /去SWPA组/ablation_results.json /仅SEPC组/ablation_results.json --output results/ablation/table.csv
```

可同时传入多折、多种子的结果文件，但五组必须有完全匹配的折、种子和支持索引。
重复运行不会自动挑最好分数；遇到重复键会报错，需要明确选择实验文件。
表格以百分数显示 Dice，各类别对各次评价等权平均，`Avg` 为类别宏平均。
ISIC 的图像加权平均与类别宏平均应区分；此 CSV 工具不混合处理 ISIC 结果。

## 检查

```bash
python -m unittest discover -s tests -v
```

测试包括五组查询路径的数值与梯度、SEPC 替代采样、检查点配置核对和汇总匹配。
不安装 PyTorch/OpenCV 时只执行不依赖模型的检查，数值检查会明确跳过。
完整开启应再与正常版使用同一权重、输入和随机状态进行回归核对。
这些检查不代替真实数据上的训练测试。正式论文结果需要五组分别重新训练、
使用一致的数据协议和预算，并按患者/体积统计，避免将相关切片作为独立样本。
