"""
Dataset for Training and Testing
Extended from ADNet code by Hansen et al.
"""
import torch
from torch.utils.data import Dataset
import torchvision.transforms as deftfx
import glob
import os
import SimpleITK as sitk
import random
import numpy as np
import warnings
from . import image_transforms as myit
from .dataset_specifics import *
from .protocol import excluded_slices


class TestDataset(Dataset):

    def __init__(self, args):
        args = dict(args)
        args['dataset'] = args['dataset'].replace('_Superpix', '')
      # 直接在 data_dir 下找 image*，不要加子文件夹
        if args['dataset'] == 'CMR':
            self.image_dirs = glob.glob(os.path.join(args['data_dir'], 'image*'))
        elif args['dataset'] == 'CHAOST2':
            self.image_dirs = glob.glob(os.path.join(args['data_dir'], 'image*'))
        elif args['dataset'] == 'SABS':
            self.image_dirs = glob.glob(os.path.join(args['data_dir'], 'image*'))
        self.image_dirs = sorted(self.image_dirs, key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0]))

        # remove test fold!
        self.FOLD = get_folds(args['dataset'])
        self.image_dirs = [elem for idx, elem in enumerate(self.image_dirs) if idx in self.FOLD[args['eval_fold']]]

        # split into support/query
        idx = np.arange(len(self.image_dirs))
        self.support_dir = self.image_dirs[idx[args['supp_idx']]]
        self.image_dirs.pop(idx[args['supp_idx']])  # remove support
        self.label = None

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, idx):
        img_path = self.image_dirs[idx]
        
        # 1. 读取 3D 原始数据
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        
        # 2. 🔥🔥 [Fix] 正确的 Instance Normalization (单切片归一化)
        # 对 3D 数据的每一层 (Depth) 单独做归一化
        norm_img_slices = []
        for i in range(img.shape[0]):  # 遍历层数
            slice_i = img[i]
            # (x - mean) / std
            norm_slice = (slice_i - slice_i.mean()) / (slice_i.std() + 1e-5)
            norm_img_slices.append(norm_slice)
            
        # 3. 将归一化后的切片重新拼回 3D 数组
        img = np.array(norm_img_slices)

        # 4. 堆叠成 3 通道 (符合 ResNet 输入要求)
        img = np.stack(3 * [img], axis=1)

        lbl = sitk.GetArrayFromImage(
            sitk.ReadImage(img_path.split('image_')[0] + 'label_' + img_path.split('image_')[-1]))
        lbl[lbl == 200] = 1
        lbl[lbl == 500] = 2
        lbl[lbl == 600] = 3
        
        # 使用 self.label (注意：在 test.py 里必须手动修改这个值)
        lbl = 1 * (lbl == self.label)

        sample = {'id': img_path}

        # Evaluation protocol (只取有前景的切片测试)
        idx = lbl.sum(axis=(1, 2)) > 0
        sample['image'] = torch.from_numpy(img[idx])
        sample['label'] = torch.from_numpy(lbl[idx])

        return sample

    def get_support_index(self, n_shot, C):
        """
        Selecting intervals according to Ouyang et al.
        """
        if n_shot == 1:
            pcts = [0.5]
        else:
            half_part = 1 / (n_shot * 2)
            part_interval = (1.0 - 1.0 / n_shot) / (n_shot - 1)
            pcts = [half_part + part_interval * ii for ii in range(n_shot)]

        return (np.array(pcts) * C).astype('int')

    def getSupport(self, label=None, all_slices=True, N=None):
        if label is None:
            raise ValueError('Need to specify label class!')

        img_path = self.support_dir
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        img = (img - img.mean()) / img.std()
        img = np.stack(3 * [img], axis=1)

        lbl = sitk.GetArrayFromImage(
            sitk.ReadImage(img_path.split('image_')[0] + 'label_' + img_path.split('image_')[-1]))
        lbl[lbl == 200] = 1
        lbl[lbl == 500] = 2
        lbl[lbl == 600] = 3
        lbl = 1 * (lbl == label)
        if np.sum(lbl) == 0:
            raise ValueError(f"严重错误: Support ID {img_path} 中不包含 Label {label}！请检查 supp_idx 配置或标签映射逻辑。")

        sample = {}
        if all_slices:
            sample['image'] = torch.from_numpy(img)
            sample['label'] = torch.from_numpy(lbl)
        else:
            # select N labeled slices
            if N is None:
                raise ValueError('Need to specify number of labeled slices!')
            idx = lbl.sum(axis=(1, 2)) > 0
            idx_ = self.get_support_index(N, idx.sum())

            sample['image'] = torch.from_numpy(img[idx][idx_])
            sample['label'] = torch.from_numpy(lbl[idx][idx_])

        return sample


class TrainDataset(Dataset):

    def __init__(self, args):
        self.n_shot = args['n_shot']
        self.n_way = args['n_way']
        self.n_query = args['n_query']
        self.n_sv = args['n_sv']
        self.max_iter = args['max_iter']
        self.read = True  # read images before get_item
        self.train_sampling = 'neighbors'
        self.min_size = args['min_size']
        self.test_label = args['test_label']
        self.exclude_label = args['exclude_label']
        self.use_gt = args['use_gt']
        self.exclude_label_mode = args.get('exclude_label_mode', 'all')
        if self.exclude_label and len(self.exclude_label) > 1 and self.exclude_label_mode == 'all':
            warnings.warn('Legacy Setting 2 uses slice-level AND: only slices containing ALL excluded labels are removed. '
                          'Use exclude_label_mode=any explicitly for OR; old scores are not interchangeable.')

        # reading the paths (leaving the reading of images into memory to __getitem__)
        base_dataset_name = args['dataset'].replace('_Superpix', '')

        # 2. Find image and label files based on the actual data_dir provided
        #    The previous code had hardcoded subdirs, we'll just use data_dir directly
        self.image_dirs = glob.glob(os.path.join(args['data_dir'], 'image_*.nii.gz'))
        self.label_dirs = glob.glob(os.path.join(args['data_dir'], 'label_*.nii.gz'))

        # 3. Find superpixel files (if using a Superpix dataset) in the SAME directory
        self.sprvxl_dirs = []  # Initialize as empty list
        if 'Superpix' in args['dataset']:
            # Try to get scale from args (like 'MIDDLE'), default if not found
            # You might need to adjust 'MIDDLE' if your command uses a different scale
            superpix_scale = args.get('superpix_scale', 'MIDDLE')
            self.sprvxl_dirs = glob.glob(os.path.join(args['data_dir'], f'superpix-{superpix_scale}_*.nii.gz'))

        # 4. Check if files were actually found BEFORE trying to sort them
        if not self.image_dirs:
            raise FileNotFoundError(f"No 'image_*.nii.gz' files found in the specified data_dir: {args['data_dir']}")
        if not self.label_dirs:
            raise FileNotFoundError(f"No 'label_*.nii.gz' files found in the specified data_dir: {args['data_dir']}")
        # Only raise error for superpixels if the dataset name requires them
        if 'Superpix' in args['dataset'] and not self.sprvxl_dirs:
            raise FileNotFoundError(
                f"Dataset specified as '{args['dataset']}' but no 'superpix-{superpix_scale}_*.nii.gz' files found in {args['data_dir']}")
        self.label_source = 'anatomical' if self.use_gt else 'superpixel'
        if not self.use_gt and not self.sprvxl_dirs:
            if not args.get('allow_gt_fallback', False):
                raise ValueError('use_gt=False but no pseudo-label files were loaded. Select a _Superpix dataset '
                                 'with matching files, use_gt=True, or explicitly allow_gt_fallback=True '
                                 '(historical anatomical-label protocol, not pseudo-label self-supervision).')
            self.label_source = 'anatomical_fallback'
            warnings.warn('Explicit anatomical-label fallback enabled despite use_gt=False; this is NOT pseudo-label training.')

        # 5. Now, sort the found file paths (using os.path.basename to be safe)
        self.image_dirs = sorted(self.image_dirs,
                                 key=lambda x: int(os.path.basename(x).split('_')[-1].split('.nii.gz')[0]))
        self.label_dirs = sorted(self.label_dirs,
                                 key=lambda x: int(os.path.basename(x).split('_')[-1].split('.nii.gz')[0]))
        if self.sprvxl_dirs:  # Sort only if superpixels were found
            self.sprvxl_dirs = sorted(self.sprvxl_dirs,
                                      key=lambda x: int(os.path.basename(x).split('_')[-1].split('.nii.gz')[0]))
        def case_ids(paths):
            return [int(os.path.basename(p).split('_')[-1].split('.nii.gz')[0]) for p in paths]
        if case_ids(self.image_dirs) != case_ids(self.label_dirs):
            raise ValueError('Image and label case IDs do not match; refusing truncated/misaligned zip loading')
        if self.sprvxl_dirs and case_ids(self.image_dirs) != case_ids(self.sprvxl_dirs):
            raise ValueError('Image and pseudo-label case IDs do not match')

        print(
            f"Found {len(self.image_dirs)} images, {len(self.label_dirs)} labels, {len(self.sprvxl_dirs)} superpixels.")  # Add print for confirmation
        # --- End Replacement Block ---


        # remove test fold!
        self.FOLD = get_folds(base_dataset_name)
        self.image_dirs = [elem for idx, elem in enumerate(self.image_dirs) if idx not in self.FOLD[args['eval_fold']]]
        self.label_dirs = [elem for idx, elem in enumerate(self.label_dirs) if idx not in self.FOLD[args['eval_fold']]]
        if self.sprvxl_dirs:
            self.sprvxl_dirs = [elem for idx, elem in enumerate(self.sprvxl_dirs) if
                                idx not in self.FOLD[args['eval_fold']]]
        # read images
        if self.read:
            self.images = {}
            self.labels = {}
            self.sprvxls = {}
            if len(self.sprvxl_dirs) > 0:
                # 情况 A: 有超像素 (比如 CMR_Superpix)，zip 三个列表
                print(f"Loading data WITH superpixels...")
                for image_dir, label_dir, sprvxl_dir in zip(self.image_dirs, self.label_dirs, self.sprvxl_dirs):
                    self.images[image_dir] = sitk.GetArrayFromImage(sitk.ReadImage(image_dir))
                    self.labels[label_dir] = sitk.GetArrayFromImage(sitk.ReadImage(label_dir))
                    self.sprvxls[sprvxl_dir] = sitk.GetArrayFromImage(sitk.ReadImage(sprvxl_dir))
            else:
                # 情况 B: 没有超像素 (你的 CHAOST2)，只 zip 图像和标签
                print(f"Loading data WITHOUT superpixels (Standard Mode)...")
                for image_dir, label_dir in zip(self.image_dirs, self.label_dirs):
                    self.images[image_dir] = sitk.GetArrayFromImage(sitk.ReadImage(image_dir))
                    self.labels[label_dir] = sitk.GetArrayFromImage(sitk.ReadImage(label_dir))
            
            # 打印一下加载了多少张，让你放心
            print(f"✅ Successfully loaded {len(self.images)} images into memory.")
    def __len__(self):
        return self.max_iter

    def gamma_tansform(self, img):
        gamma_range = (0.5, 1.5)
        gamma = np.random.rand() * (gamma_range[1] - gamma_range[0]) + gamma_range[0]
        cmin = img.min()
        irange = (img.max() - cmin + 1e-5)

        img = img - cmin + 1e-5
        img = irange * np.power(img * 1.0 / irange, gamma)
        img = img + cmin

        return img

    def geom_transform(self, img, mask):

        affine = {'rotate': 5, 'shift': (5, 5), 'shear': 5, 'scale': (0.9, 1.2)}
        alpha = 10
        sigma = 5
        order = 3

        tfx = []
        tfx.append(myit.RandomAffine(affine.get('rotate'),
                                     affine.get('shift'),
                                     affine.get('shear'),
                                     affine.get('scale'),
                                     affine.get('scale_iso', True),
                                     order=order))
        tfx.append(myit.ElasticTransform(alpha, sigma))
        transform = deftfx.Compose(tfx)

        if len(img.shape) > 4:
            n_shot = img.shape[1]
            for shot in range(n_shot):
                cat = np.concatenate((img[0, shot], mask[:, shot])).transpose(1, 2, 0)
                cat = transform(cat).transpose(2, 0, 1)
                img[0, shot] = cat[:3, :, :]
                mask[:, shot] = np.rint(cat[3:, :, :])

        else:
            for q in range(img.shape[0]):
                cat = np.concatenate((img[q], mask[q][None])).transpose(1, 2, 0)
                cat = transform(cat).transpose(2, 0, 1)
                img[q] = cat[:3, :, :]
                mask[q] = np.rint(cat[3:, :, :].squeeze())

        return img, mask

    def __getitem__(self, idx):

        # sample patient idx
        pat_idx = random.choice(range(len(self.image_dirs)))

        if self.read:
            # get image/supervoxel volume from dictionary
            img = self.images[self.image_dirs[pat_idx]]
            gt = self.labels[self.label_dirs[pat_idx]]
            if len(self.sprvxl_dirs) > 0:
                sprvxl = self.sprvxls[self.sprvxl_dirs[pat_idx]]
            else:
                sprvxl = gt.copy() # 或者 None，取决于你后面怎么用，通常用 gt 代替占位
        else:
            # read image/supervoxel volume into memory
            img = sitk.GetArrayFromImage(sitk.ReadImage(self.image_dirs[pat_idx]))
            gt = sitk.GetArrayFromImage(sitk.ReadImage(self.label_dirs[pat_idx]))
            if len(self.sprvxl_dirs) > 0:
                sprvxl = sitk.GetArrayFromImage(sitk.ReadImage(self.sprvxl_dirs[pat_idx]))
            else:
                sprvxl = gt.copy()
        if self.exclude_label is not None:  # identify the slices containing test labels
            idx = np.arange(gt.shape[0])
            exclude_idx = excluded_slices(gt, self.exclude_label, self.exclude_label_mode)
            exclude_idx = idx[exclude_idx]
        else:
            exclude_idx = []

        # normalize
        img = (img - img.mean()) / img.std()

        # chose training label
        if self.use_gt:
            lbl = gt.copy()
        else:
            lbl = sprvxl.copy()
        # lbl is label numpy

        # sample class(es) (gt/supervoxel)
        unique = list(np.unique(lbl))
        unique.remove(0)
        if self.use_gt:
            unique = list(set(unique) - set(self.test_label))

        size = 0
        while size < self.min_size:
            n_slices = (self.n_shot * self.n_way) + self.n_query - 1
            while n_slices < ((self.n_shot * self.n_way) + self.n_query):

                cls_idx = random.choice(unique)   # cls_idx is sampled class id

                # extract slices containing the sampled class
                sli_idx = np.sum(lbl == cls_idx, axis=(1, 2)) > 0
                idx = np.arange(lbl.shape[0])
                sli_idx = idx[sli_idx]
                sli_idx = list(set(sli_idx) - set(np.intersect1d(sli_idx, exclude_idx)))
               
                n_slices = len(sli_idx)

            # generate possible subsets with successive slices (size = self.n_shot * self.n_way + self.n_query)
            subsets = []
            for i in range(len(sli_idx)):
                if not subsets:
                    subsets.append([sli_idx[i]])
                elif sli_idx[i - 1] + 1 == sli_idx[i]:
                    subsets[-1].append(sli_idx[i])
                else:
                    subsets.append([sli_idx[i]])
            i = 0
            while i < len(subsets):
                if len(subsets[i]) < (self.n_shot * self.n_way + self.n_query):
                    del subsets[i]
                else:
                    i += 1
            if not len(subsets):
                return self.__getitem__(idx + np.random.randint(low=0, high=self.max_iter - 1, size=(1,)))

            # sample support and query slices
            i = random.choice(np.arange(len(subsets)))  # subset index
            i = random.choice(subsets[i][:-(self.n_shot * self.n_way + self.n_query - 1)])
            sample = np.arange(i, i + (self.n_shot * self.n_way) + self.n_query)

            lbl_cls = 1 * (lbl == cls_idx)

            size = max(np.sum(lbl_cls[sample[0]]), np.sum(lbl_cls[sample[1]]))

        # invert order
        if np.random.random(1) > 0.5:
            sample = sample[::-1]  # successive slices (inverted)

        sup_lbl = lbl_cls[sample[:self.n_shot * self.n_way]][None,]  # n_way * (n_shot * C) * H * W
        qry_lbl = lbl_cls[sample[self.n_shot * self.n_way:]]  # n_qry * C * H * W

        sup_img = img[sample[:self.n_shot * self.n_way]][None,]  # n_way * (n_shot * C) * H * W
        sup_img = np.stack((sup_img, sup_img, sup_img), axis=2)
        qry_img = img[sample[self.n_shot * self.n_way:]]  # n_qry * C * H * W
        qry_img = np.stack((qry_img, qry_img, qry_img), axis=1)



        # gamma transform
        if np.random.random(1) > 0.5:
            qry_img = self.gamma_tansform(qry_img)
        else:
            sup_img = self.gamma_tansform(sup_img)

        # geom transform
        if np.random.random(1) > 0.5:
            qry_img, qry_lbl = self.geom_transform(qry_img, qry_lbl)
        else:
            sup_img, sup_lbl = self.geom_transform(sup_img, sup_lbl)

        sample = {'support_images': sup_img,
                  'support_fg_labels': sup_lbl,
                  'query_images': qry_img,
                  'query_labels': qry_lbl,
                  'selected_class': cls_idx}

        return sample
