# BSP规范版

本版分支为 `bsp-standard`，基于外部资源路径版提交 `99832de`。
修改范围为文件名、类名、函数名、实验显示名、注释与空白格式；
不改变模型结构、张量计算、参数初始化顺序、损失、采样规则、训练预算及五组消融定义。

## 代码名称

| 规范名称 | 位置 | 职责 |
|---|---|---|
| BSPModel | models/bsp.py | 主模型及 SEPC、QRR、SWPA 调度 |
| SEPCDecoder、SEPCPrototypeMLP | models/sepc.py | 支持预测解码器、基础原型压缩 |
| sepc_region_contexts | models/bsp.py | 区域上下文库 |
| sepc_self_prediction | models/bsp.py | 支持自预测 |
| sepc_foreground_prototypes、sepc_background_prototypes | models/bsp.py | 前景、背景原型构建 |
| qrr_fusion_factor | models/bsp.py | 查询细化的融合系数 |
| QRR_SWPA | models/bsp.py | 查询残差细化和空间加权聚合 |
| SWPADecoder | models/swpa.py | 聚合后的分割解码器 |

QRR 与 SWPA 原本就在同一个函数中，本次保留这一结构，统一命名为 `QRR_SWPA`。
未为了名称创建新计算层或拆分计算流程。CT 专用模型命名为 `models/bsp_ct.py`，
对应共用组件为 `models/sepc_ct.py`；CT 专用入口的原有能力范围不变。

本次验证：37 个 Python 文件在名称映射、移除说明字符串后可执行语法树一致；
13 项模型/路径测试通过；五组配置的查询与支持预测路径在固定 CPU 输入下与旧版
逐位一致。Python 注释及历史注释代码已清理，脚本执行所需的 shebang 保留。
这些检查不替代真实数据训练，也不表示新增训练结果。

## 名称对照

| 旧名称 | 新名称 |
|---|---|
| models/cow.py | models/bsp.py |
| models/cow_ct.py | models/bsp_ct.py |
| models/modules.py | models/sepc.py |
| models/module_ct.py | models/sepc_ct.py |
| models/Decoders.py | models/swpa.py |
| FewShotSeg | BSPModel |
| getSelfPred | sepc_self_prediction |
| IFR_FD | QRR_SWPA |
| get_fuse_factor | qrr_fusion_factor |
| Experiment 名称 CoW | BSP |

原 `state_dict` 内的参数属性名保持不变，包括历史上的 `ifr_decoder_*`、
`supp_decoder`、`mlp1` 等，因此现有权重键名兼容。类名和函数名不进入这些键名。
当前保存方式为 `state_dict`；自行序列化的完整 Python 模型对象不在此兼容承诺内。

## 命令与输出

训练、测试入口仍然为 `train.py`、`test.py`，五组参数值保持不变：
`full`、`sepc_only`、`wo_sepc`、`wo_qrr`、`wo_swpa`。
外部数据和权重仍由 `data_root`、`pretrained_weights` 指定。

```bash
git clone --branch bsp-standard https://github.com/aihongjian777-crypto/BSP-cvpr.git BSP-standard
cd BSP-standard
python train.py with ablation_variant=sepc_only mode=train dataset=CHAOST2 eval_fold=2 n_steps=50000 max_iters_per_load=5000 lr_step_gamma=0.9 use_gt=False exclude_label=None n_way=1 n_shot=1 n_query=1 batch_size=1 num_workers=4 seed=1234 save_snapshot_every=5000 data_root=/datasets/CHAOST2 pretrained_weights=/pretrained/deeplabv3_resnet101_coco-586e9e4e.pth path.log_dir=/experiments/BSP
```

新日志目录使用 `BSP_train_...`，不再以旧实验名称开头。旧权重路径按实际位置传入即可，
无需重命名或移动已有权重。新的数据路径说明见 `EXTERNAL_PATHS.md`。

## 来源与许可

本次名称整理不改变代码的继承关系和贡献归属：CoW、DIFD、ADNet、SSL-ALPNet
及 FoB_SAM 的相关来源仍在 README 与许可文件中保留。
模型辅助函数采用 BSP 术语命名，不代表将继承算法重新宣称为原创。
`LICENSES` 下的许可文本不作删改。
