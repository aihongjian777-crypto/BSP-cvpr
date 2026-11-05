#!/usr/bin/env python
"""
For evaluation
Extended from ADNet code by Hansen et al.
"""
import shutil
import json
from pathlib import Path
import SimpleITK as sitk
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from models.cow import FewShotSeg
from dataloaders.datasets import TestDataset
from dataloaders.dataset_specifics import *
from utils import *
from config import ex
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# def fourier_augmentation(img, L=0.01):
#     """
#     傅里叶振幅扰动 (兼容 PyTorch < 1.7 的旧版本写法)
#     Args:
#         img: [B, 3, H, W]
#         L: 扰动强度
#     """
#     # 1. 傅里叶变换 (Old Style)
#     # signal_ndim=2 表示只在最后两个维度(H, W)上做变换
#     # onesided=True 对应 rfft (实数输入)
#     # 输出形状: [B, 3, H, W, 2]，最后一个维度存的是 (实部, 虚部)
#     fft = torch.rfft(img, signal_ndim=2, normalized=False, onesided=True)

#     # 2. 手动提取振幅(Amp) 和 相位(Phase)
#     # 在旧版中，复数是分开存的
#     real = fft[..., 0]
#     imag = fft[..., 1]

#     # Amp = sqrt(real^2 + imag^2)
#     amp = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)  # 加个极小值防报错
#     # Phase = atan2(imag, real)
#     pha = torch.atan2(imag, real)

#     # 3. 给振幅加噪 (核心步骤不变)
#     noise = torch.rand_like(amp).to(img.device)
#     # 缩放到 [-L, L] 之间，然后 +1
#     amp_noise = amp * (1 + (noise - 0.5) * 2 * L)

#     # 4. 重建复数
#     # Real = Amp * cos(Phase)
#     # Imag = Amp * sin(Phase)
#     real_new = amp_noise * torch.cos(pha)
#     imag_new = amp_noise * torch.sin(pha)

#     # 重新堆叠回 [..., 2] 的格式
#     fft_new = torch.stack([real_new, imag_new], dim=-1)

#     # 5. 逆变换 (Inverse FFT)
#     # signal_sizes 是必须的，否则 output shape 可能会差 1 个像素
#     img_new = torch.irfft(fft_new, signal_ndim=2, normalized=False,
#                           onesided=True, signal_sizes=img.shape[-2:])

#     return img_new
def fourier_augmentation(img, L=0.01):
    """
    傅里叶振幅扰动（兼容 PyTorch >= 1.8）
    Args:
        img: [B, 3, H, W]
        L: 扰动强度
    """
    # 1. 傅里叶变换，输出复数张量 [B, 3, H, W//2+1]
    fft = torch.fft.rfft2(img, norm=None)

    # 2. 提取振幅和相位
    real = fft.real
    imag = fft.imag
    amp = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
    pha = torch.atan2(imag, real)

    # 3. 给振幅加噪（逻辑与原版完全一致）
    noise = torch.rand_like(amp).to(img.device)
    amp_noise = amp * (1 + (noise - 0.5) * 2 * L)

    # 4. 重建复数张量
    real_new = amp_noise * torch.cos(pha)
    imag_new = amp_noise * torch.sin(pha)
    fft_new = torch.complex(real_new, imag_new)

    # 5. 逆变换，signal_sizes 对应原版的 signal_sizes=img.shape[-2:]
    img_new = torch.fft.irfft2(fft_new, s=img.shape[-2:], norm=None)

    return img_new





def evaluate_isic_setting1(model, config, log, run):
    """FoB_SAM's held-out, within-category next-image pairs with BSP predictions."""
    from PIL import Image
    from dataloaders.isic_setting_1 import (
        get_isic_setting_1_categories, get_superpix_isic_dataset,
    )
    from dataloaders.isic_paths import UPSTREAM_COMMIT

    device = next(model.parameters()).device
    model.eval()
    result_dir = Path(run.observers[0].dir)
    categories = get_isic_setting_1_categories(config['isic_setting_1_base_path'])
    category_results, records = {}, []
    for category in categories:
        dataset = get_superpix_isic_dataset(
            base_path=config['isic_setting_1_base_path'],
            data_root=config['path']['isic']['data_dir'],
            fold=config['eval_fold'], category=category, train=False,
            image_size=config['isic_image_size'],
            superpixel_mode=config['isic_superpixel_mode'])
        loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False,
                            num_workers=config['num_workers'], pin_memory=True, drop_last=False)
        scores = Scores()
        with torch.no_grad():
            for sample in loader:
                support = [[shot.float().to(device) for shot in way]
                           for way in sample['support_images']]
                masks = [[shot.float().to(device) for shot in way]
                         for way in sample['support_fg_labels']]
                queries = [image.float().to(device) for image in sample['query_images']]
                target = torch.cat([label.long() for label in sample['query_labels']], dim=0)
                probability, _, _, _, _ = model(support, masks, queries, train=False)
                prediction = probability.argmax(dim=1).cpu()
                for batch_index in range(prediction.shape[0]):
                    scores.record(prediction[batch_index], target[batch_index])
                    query_id = sample['query_name'][batch_index]
                    records.append({
                        'category': category, 'query': query_id,
                        'support': sample['support_name'][batch_index],
                        'dice': float(scores.patient_dice[-1]),
                        'iou': float(scores.patient_iou[-1]),
                    })
                    if config['save_isic_predictions']:
                        folder = result_dir / 'interm_preds' / category
                        folder.mkdir(parents=True, exist_ok=True)
                        Image.fromarray((prediction[batch_index].numpy() * 255).astype(np.uint8)).save(
                            str(folder / (query_id + '_prediction.png')))
        category_results[category] = {
            'count': len(dataset),
            'dice': sum(float(value) for value in scores.patient_dice) / len(dataset),
            'iou': sum(float(value) for value in scores.patient_iou) / len(dataset),
        }
        log.info('ISIC %s: %s pairs, Dice %.6f, IoU %.6f', category, len(dataset),
                 category_results[category]['dice'], category_results[category]['iou'])
    results = {
        'ablation_variant': config['ablation_variant'],
        'fold': config['eval_fold'], 'upstream_commit': UPSTREAM_COMMIT,
        'superpixel_mode': config['isic_superpixel_mode'],
        'checkpoint': config['reload_model_path'], 'categories': category_results,
        # FoB_SAM's test.py reports an image-weighted average across categories.
        'mean_dice': sum(record['dice'] for record in records) / len(records),
        'mean_iou': sum(record['iou'] for record in records) / len(records),
        'macro_category_dice': sum(value['dice'] for value in category_results.values()) / len(categories),
        'macro_category_iou': sum(value['iou'] for value in category_results.values()) / len(categories),
        'pairs': records,
    }
    (result_dir / 'isic_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    run.log_scalar('isic_mean_dice', results['mean_dice'])
    run.log_scalar('isic_mean_iou', results['mean_iou'])
    log.info('Total Mean Dice: %.6f; Mean IoU: %.6f', results['mean_dice'], results['mean_iou'])
    return results['mean_dice']


@ex.automain
def main(_run, _config, _log):
    from models.ablation import module_flags, validate_checkpoint
    if not _config['reload_model_path']:
        raise ValueError('Evaluation requires reload_model_path for the intended run')
    verified = validate_checkpoint(_config['reload_model_path'], _config)
    _log.info('Ablation %s: %s', _config['ablation_variant'], module_flags(_config['ablation_variant']))
    if not verified:
        _log.warning('Legacy full checkpoint: no ablation metadata; provenance was not verified.')
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/interm_preds', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

        file_handler = logging.FileHandler(os.path.join(f'{_run.observers[0].dir}', f'logger.log'))
        file_handler.setLevel('INFO')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(formatter)
        _log.handlers.append(file_handler)
        _log.info(f'Run "{_config["exp_str"]}" with ID "{_run.observers[0].dir[-1]}"')

    if _config['seed'] is not None:
        random.seed(_config['seed'])
        torch.manual_seed(_config['seed'])
        torch.cuda.manual_seed_all(_config['seed'])
        cudnn.deterministic = True

    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info(f'Create model...')
    model = FewShotSeg(pretrained_weights=None,
                       ablation_variant=_config["ablation_variant"])
    model.cuda()
    
    # --- 修改开始 ---
    checkpoint_path = _config['reload_model_path']
    _log.info(f'Loading weights from: {checkpoint_path}')
    
    if not checkpoint_path:
        raise ValueError('Evaluation requires reload_model_path for the intended run')
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # 检查是否包含 'module.' 前缀 (DataParallel 留下的坑)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') # 去掉 module. 前缀
        new_state_dict[name] = v
        
    # 加载并获取加载报告
    missing_keys, unexpected_keys = model.load_state_dict(
        new_state_dict, strict=not _config['allow_partial_checkpoint'])
    
    # 打印警告，让你看到是否真的加载了
    if len(missing_keys) > 0:
        _log.warning(f'WARNING: Missing keys (未加载的层): {missing_keys[:5]} ... (total {len(missing_keys)})')
    if len(unexpected_keys) > 0:
        _log.warning(f'WARNING: Unexpected keys (多余的层): {unexpected_keys[:5]} ...')
        
    _log.info('Weights loaded.')
    if _config['dataset'] == 'isic':
        return evaluate_isic_setting1(model, _config, _log, _run)
    # --- 修改结束 ---
    
    # 🔥🔥 [Fix] 这里的关键是把 _Superpix 后缀去掉，才能在 path 字典里找到对应的 key
    base_dataset_name = _config['dataset'].replace('_Superpix', '')

    data_config = {
        # ✅ 使用处理过的 base_dataset_name (即 'CHAOST2') 去查路径
        'data_dir': _config['path'][base_dataset_name]['data_dir'],
        'dataset': _config['dataset'],
        'n_shot': _config['n_shot'],
        'n_way': _config['n_way'],
        'n_query': _config['n_query'],
        'n_sv': _config['n_sv'],
        'max_iter': _config['max_iters_per_load'],
        'eval_fold': _config['eval_fold'],
        'min_size': _config['min_size'],
        'max_slices': _config['max_slices'],
        'supp_idx': _config['supp_idx'],
    }
    test_dataset = TestDataset(data_config)
    test_loader = DataLoader(test_dataset,
                             batch_size=_config['batch_size'],
                             shuffle=False,
                             num_workers=_config['num_workers'],
                             pin_memory=True,
                             drop_last=False)

    labels = get_label_names(_config['dataset'])

    class_dice = {}
    class_iou = {}

    _log.info(f'Starting validation...')
    for label_val, label_name in labels.items():

        if label_name == 'BG':
            continue
        elif (not np.intersect1d([label_val], _config['test_label'])):
            continue

        _log.info(f'Test Class: {label_name}')

        support_sample = test_dataset.getSupport(label=label_val, all_slices=False, N=_config['n_part'])

        test_dataset.label = label_val
        # ============================================================
        # 🔥🔥 [Step 4: 自监督 TTA (Test-Time Adaptation)] 修正版 🔥🔥
        # ============================================================

        # 1. 准备原材料
        supp_imgs = support_sample['image'].float().cuda()  # [N, 3, H, W]
        supp_mask = support_sample['label'].float().cuda()  # [N, H, W] 或 [N, 1, H, W]

        # 维度修正：确保 mask 是 [N, H, W]
        if supp_mask.dim() == 4 and supp_mask.shape[1] == 1:
            supp_mask = supp_mask.squeeze(1)

        # 2. 制造困难样本：傅里叶增强
        supp_imgs_aug = fourier_augmentation(supp_imgs, L=0.02)

        # 3. 设置模型状态
        model.train()
        model.encoder.train()

        # 冻结策略：只更新 BN
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.encoder.named_parameters():
            if "bn" in name:
                param.requires_grad = True

        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

        # 🔥🔥🔥 关键修正：把大张量拆成 "List of Tensors" 以适配模型接口 🔥🔥🔥
        # 目标格式: [ [Shot1, Shot2, ...] ] (Way=1)
        # 原图列表
        orig_img_list = [[supp_imgs]]       
        aug_img_list = [[supp_imgs_aug]]
        mask_list = [[supp_mask]]           

        # 4. 闭环学习循环
        for tta_step in range(0):
            optimizer.zero_grad()

            # (A) 老师视角 (原图)
            # Query 传入 [supp_imgs] (列表长度为1)，满足 assert n_queries == 1
            # 但 supp_imgs 内部 batch=3，所以会输出 3 张预测图
            output_orig, _, _, _, _ = model(
                orig_img_list, mask_list, [supp_imgs], train=True
            )

            # 损失 1: 监督损失
            # output_orig 现在是 [3, 2, H, W]，supp_mask 是 [3, H, W] -> 维度完美匹配！
            log_prob = torch.log(torch.clamp(output_orig, 1e-8, 1.0))
            loss_sup = model.criterion(log_prob, supp_mask.long())

            # (B) 学生视角 (增强图)
            output_aug, _, _, _, _ = model(
                aug_img_list, mask_list, [supp_imgs_aug], train=True
            )

            # 损失 2: 一致性损失
            loss_cons = torch.nn.functional.mse_loss(output_aug, output_orig.detach())

            # (C) 更新
            loss = loss_sup + 0.5 * loss_cons
            loss.backward()
            optimizer.step()
            

        # 5. 状态恢复 & 固化
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # ============================================================
        # 🔥🔥 [Step 4: 自监督 TTA (Test-Time Adaptation)] 结束 🔥🔥
        # ============================================================

        with torch.no_grad():
            model.eval()

            support_image = [support_sample['image'][[i]].float().cuda() for i in
                             range(support_sample['image'].shape[0])]
            support_fg_mask = [support_sample['label'][[i]].float().cuda() for i in
                               range(support_sample['image'].shape[0])]

            # Loop through query volumes.
            scores = Scores()
           # ✅ 外层循环使用 i
            for i, sample in enumerate(test_loader):

                # Unpack query data.
                query_image = [sample['image'][i].float().cuda() for i in range(sample['image'].shape[0])]
                query_label = sample['label'].long()
                query_id = sample['id'][0].split('image_')[1][:-len('.nii.gz')]

                # Compute output.
                query_pred = torch.zeros(query_label.shape[-3:])
                C_q = sample['image'].shape[1]

                idx_ = np.linspace(0, C_q, _config['n_part'] + 1).astype('int')
                for sub_chunck in range(_config['n_part']):
                    if idx_[sub_chunck] == idx_[sub_chunck + 1]:
                        continue  # No slice in this part; do not concatenate an empty list.
                    support_image_s = [support_image[sub_chunck]]
                    support_fg_mask_s = [support_fg_mask[sub_chunck]]
                    query_image_s = query_image[0][idx_[sub_chunck]:idx_[sub_chunck + 1]]
                    query_pred_s = []
                    
                    # ✅【修复 1】内层循环变量名改成 slice_idx，不再覆盖 i
                    for slice_idx in range(query_image_s.shape[0]):
                        _pred_s, _, _, _, _ = model([support_image_s], [support_fg_mask_s], [query_image_s[[slice_idx]]], train=False)
                        query_pred_s.append(_pred_s)
                    
                    query_pred_s = torch.cat(query_pred_s, dim=0)
                    query_pred_s = query_pred_s.argmax(dim=1).cpu()
                    query_pred[idx_[sub_chunck]:idx_[sub_chunck + 1]] = query_pred_s

                # -------------------------------------------------------------------
                # ✅【修复 2】强制打印 DEBUG 信息 (删除了 if i==0)
                # -------------------------------------------------------------------
                _log.info(f"\n====== DEBUG INFO for Patient {i} ({query_id}) ======")
                _log.info(f"👉 Target Label ID: {label_val}")
                
                # 检查 GT
                unique_gt = torch.unique(query_label)
                _log.info(f"👉 GT Unique Values: {unique_gt.tolist()}")
                
                # 检查二值化 GT (关键!)
                query_label_binary_debug = (query_label == label_val).long()
                _log.info(f"👉 Binary GT Unique Values (Should be [0, 1]): {torch.unique(query_label_binary_debug).tolist()}")

                # 检查预测
                unique_pred = torch.unique(query_pred)
                _log.info(f"👉 Model Pred Values (Should be [0, 1]): {unique_pred.tolist()}")
                
                # 检查 Support Mask 是否为空 (Dice=0 的最大嫌疑人)
                if 'support_fg_mask' in locals():
                    # support_fg_mask 是一个 list of tensors，我们求和看看
                    supp_sum = sum([m.sum().item() for m in support_fg_mask])
                    _log.info(f"🖼️ Support Mask Pixel Count: {supp_sum} (If 0, model is blind!)")
                
                _log.info("===================================================\n")
                # -------------------------------------------------------------------

                _log.info(f"\n====== DEBUG INFO for Patient {i} ({query_id}) ======")
                _log.info(f"👉 Target Label ID: {label_val}")
                _log.info(f"👉 GT Unique Values: {torch.unique(query_label).tolist()}")
                
                # 这里的 query_label 已经被 datasets.py 处理成 0/1 了
                # 所以我们必须用 == 1 来提取前景
                gt_fixed = (query_label == 1).long()
                
                _log.info(f"👉 Fixed GT Values (Should be [0, 1]): {torch.unique(gt_fixed).tolist()}")
                _log.info(f"👉 Model Pred Values: {torch.unique(query_pred).tolist()}")
                
                if 'support_fg_mask' in locals():
                    supp_sum = sum([m.sum().item() for m in support_fg_mask])
                    _log.info(f"🖼️ Support Mask Pixel Count: {supp_sum}")
                _log.info("===================================================\n")

                # =================================================================
                # 🔥🔥 [Fix] 形状与数值修正 (修复报错的核心) 🔥🔥
                # =================================================================
                
                # 1. 维度修正: query_label 通常是 [1, D, H, W]，我们需要 [D, H, W]
                # 使用 .squeeze(0) 去掉 batch 维度
                if gt_fixed.dim() == 4:
                    gt_fixed = gt_fixed.squeeze(0)
                
                # 2. 记录分数
                # 现在的 query_pred 是 [D, H, W], gt_fixed 也是 [D, H, W]
                # 且两者都只包含 0 和 1，Dice 计算终于可以正常工作了！
                scores.record(query_pred, gt_fixed)

                _log.info(
                    f'Tested query volume: {sample["id"][0][len(_config["path"][_config["dataset"]]["data_dir"]):]}.')
                _log.info(f'Dice score: {scores.patient_dice[-1].item()}')

                file_name = os.path.join(f'{_run.observers[0].dir}/interm_preds',
                                         f'prediction_{query_id}_{label_name}.nii.gz')
                itk_pred = sitk.GetImageFromArray(query_pred)
                sitk.WriteImage(itk_pred, file_name, True)
                _log.info(f'{query_id} has been saved. ')

            class_dice[label_name] = torch.tensor(scores.patient_dice).mean().item()
            class_iou[label_name] = torch.tensor(scores.patient_iou).mean().item()
            _log.info(f'Test Class: {label_name}')
            _log.info(f'Mean class IoU: {class_iou[label_name]}')
            _log.info(f'Mean class Dice: {class_dice[label_name]}')

    _log.info(f'Final results...')
    _log.info(f'Mean IoU: {class_iou}')
    _log.info(f'Mean Dice: {class_dice}')

    def dict_Avg(Dict):
        A = sum(Dict.values()) / len(Dict)
        return A
    res = dict_Avg(class_dice)

    results = {
        'ablation_variant': _config['ablation_variant'],
        'modules': module_flags(_config['ablation_variant']),
        'dataset': _config['dataset'], 'fold': _config['eval_fold'], 'seed': _config['seed'],
        'support_index': _config['supp_idx'], 'checkpoint': _config['reload_model_path'],
        'class_dice': class_dice, 'class_iou': class_iou, 'mean_dice': res,
        'mean_iou': dict_Avg(class_iou),
    }
    if _run.observers:
        (Path(_run.observers[0].dir) / 'ablation_results.json').write_text(
            json.dumps(results, indent=2), encoding='utf-8')
    _log.info(f'Total Mean Dice: {res}')
    _log.info(f'End of validation.')
    return 1













# #!/usr/bin/env python
# """
# For evaluation
# Extended from ADNet code by Hansen et al.
# """
# import shutil
# import SimpleITK as sitk
# import torch.backends.cudnn as cudnn
# import torch.optim
# from torch.utils.data import DataLoader
# from models.cow import FewShotSeg
# from dataloaders.datasets import TestDataset
# from dataloaders.dataset_specifics import *
# from utils import *
# from config import ex
# # os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# @ex.automain
# def main(_run, _config, _log):
#     if _run.observers:
#         os.makedirs(f'{_run.observers[0].dir}/interm_preds', exist_ok=True)
#         for source_file, _ in _run.experiment_info['sources']:
#             os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
#                         exist_ok=True)
#             _run.observers[0].save_file(source_file, f'source/{source_file}')
#         shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

#         file_handler = logging.FileHandler(os.path.join(f'{_run.observers[0].dir}', f'logger.log'))
#         file_handler.setLevel('INFO')
#         formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
#         file_handler.setFormatter(formatter)
#         _log.handlers.append(file_handler)
#         _log.info(f'Run "{_config["exp_str"]}" with ID "{_run.observers[0].dir[-1]}"')

#     if _config['seed'] is not None:
#         random.seed(_config['seed'])
#         torch.manual_seed(_config['seed'])
#         torch.cuda.manual_seed_all(_config['seed'])
#         cudnn.deterministic = True

#     cudnn.enabled = True
#     cudnn.benchmark = True
#     torch.cuda.set_device(device=_config['gpu_id'])
#     torch.set_num_threads(1)

#     _log.info(f'Create model...')
#     model = FewShotSeg()
#     model.cuda()
#     model.load_state_dict(torch.load(_config['reload_model_path'], map_location='cpu'))

#     _log.info(f'Load data...')
#     data_config = {
#         'data_dir': _config['path'][_config['dataset']]['data_dir'],
#         'dataset': _config['dataset'],
#         'n_shot': _config['n_shot'],
#         'n_way': _config['n_way'],
#         'n_query': _config['n_query'],
#         'n_sv': _config['n_sv'],
#         'max_iter': _config['max_iters_per_load'],
#         'eval_fold': _config['eval_fold'],
#         'min_size': _config['min_size'],
#         'max_slices': _config['max_slices'],
#         'supp_idx': _config['supp_idx'],
#     }
#     test_dataset = TestDataset(data_config)
#     test_loader = DataLoader(test_dataset,
#                              batch_size=_config['batch_size'],
#                              shuffle=False,
#                              num_workers=_config['num_workers'],
#                              pin_memory=True,
#                              drop_last=True)

#     labels = get_label_names(_config['dataset'])

#     class_dice = {}
#     class_iou = {}

#     _log.info(f'Starting validation...')
#     for label_val, label_name in labels.items():

#         if label_name == 'BG':
#             continue
#         elif (not np.intersect1d([label_val], _config['test_label'])):
#             continue

#         _log.info(f'Test Class: {label_name}')

#         support_sample = test_dataset.getSupport(label=label_val, all_slices=False, N=_config['n_part'])

#         test_dataset.label = label_val

#         with torch.no_grad():
#             model.eval()

#             support_image = [support_sample['image'][[i]].float().cuda() for i in
#                              range(support_sample['image'].shape[0])]
#             support_fg_mask = [support_sample['label'][[i]].float().cuda() for i in
#                                range(support_sample['image'].shape[0])]

#             # Loop through query volumes.
#             scores = Scores()
#             for i, sample in enumerate(test_loader):

#                 # Unpack query data.
#                 query_image = [sample['image'][i].float().cuda() for i in
#                                range(sample['image'].shape[0])]
#                 query_label = sample['label'].long()
#                 query_id = sample['id'][0].split('image_')[1][:-len('.nii.gz')]

#                 # Compute output.
#                 query_pred = torch.zeros(query_label.shape[-3:])
#                 C_q = sample['image'].shape[1]

#                 idx_ = np.linspace(0, C_q, _config['n_part'] + 1).astype('int')
#                 for sub_chunck in range(_config['n_part']):  # n_part = 3
#                     support_image_s = [support_image[sub_chunck]]
#                     support_fg_mask_s = [support_fg_mask[sub_chunck]]
#                     query_image_s = query_image[0][idx_[sub_chunck]:idx_[sub_chunck + 1]]
#                     query_pred_s = []
#                     for i in range(query_image_s.shape[0]):
#                         _pred_s, _, _, _, _ = model([support_image_s], [support_fg_mask_s], [query_image_s[[i]]],
#                                            train=False)
#                         query_pred_s.append(_pred_s)
#                     query_pred_s = torch.cat(query_pred_s, dim=0)
#                     query_pred_s = query_pred_s.argmax(dim=1).cpu()

#                     query_pred[idx_[sub_chunck]:idx_[sub_chunck + 1]] = query_pred_s

#                 scores.record(query_pred, query_label)

#                 _log.info(
#                     f'Tested query volume: {sample["id"][0][len(_config["path"][_config["dataset"]]["data_dir"]):]}.')
#                 _log.info(f'Dice score: {scores.patient_dice[-1].item()}')

#                 file_name = os.path.join(f'{_run.observers[0].dir}/interm_preds',
#                                          f'prediction_{query_id}_{label_name}.nii.gz')
#                 itk_pred = sitk.GetImageFromArray(query_pred)
#                 sitk.WriteImage(itk_pred, file_name, True)
#                 _log.info(f'{query_id} has been saved. ')

#             class_dice[label_name] = torch.tensor(scores.patient_dice).mean().item()
#             class_iou[label_name] = torch.tensor(scores.patient_iou).mean().item()
#             _log.info(f'Test Class: {label_name}')
#             _log.info(f'Mean class IoU: {class_iou[label_name]}')
#             _log.info(f'Mean class Dice: {class_dice[label_name]}')

#     _log.info(f'Final results...')
#     _log.info(f'Mean IoU: {class_iou}')
#     _log.info(f'Mean Dice: {class_dice}')

#     def dict_Avg(Dict):
#         A = sum(Dict.values()) / len(Dict)
#         return A
#     res = dict_Avg(class_dice)

#     _log.info(f'Total Mean Dice: {res}')
#     _log.info(f'End of validation.')
#     return 1

