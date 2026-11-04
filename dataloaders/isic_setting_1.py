"""FoB_SAM Setting I loader adapted for portable BSP data paths.

Source: primebo1/FoB_SAM @ 40d0e23d75f81f07bb7f34cb641db51ef490d025.
superpixel_mode='upstream' retains upstream interpolation and integer indexing.
"""
import os
import torch.utils.data as data
import numpy as np
import random
import torch
from dataloaders.transform import get_polyp_transform
import cv2
import torch.nn.functional as F
from dataloaders.isic_paths import (
    DEFAULT_BASE_PATH, DEFAULT_DATA_ROOT, project_path, image_and_mask_roots,
    find_data_file, load_fold_splits, resolve_fold,
)

IMAGENET_MEAN = np.array([0.40, 0.40, 0.40])
IMAGENET_STD  = np.array([0.22, 0.22, 0.22])
ISIC_SETTING_1_CATEGORIES = ['melanoma', 'nevus', 'seborrheic_keratosis']
ISIC_SETTING_1_SPLIT_FILE = 'isic_in_5_folds.json'


def resolve_isic_setting_1_base_path(base_path=None):
    return str(project_path(base_path, DEFAULT_BASE_PATH))


def load_isic_setting_1_split(base_path=None):
    return load_fold_splits(base_path)


def resolve_isic_setting_1_fold(fold_splits, fold):
    return resolve_fold(fold_splits, fold)


def get_isic_setting_1_categories(base_path=None, fold_splits=None):
    fold_splits = fold_splits or load_isic_setting_1_split(base_path)
    available_categories = set()
    for split in fold_splits.values():
        available_categories.update(split.keys())

    ordered_categories = [cat for cat in ISIC_SETTING_1_CATEGORIES if cat in available_categories]
    ordered_categories.extend(sorted(available_categories - set(ordered_categories)))
    return ordered_categories


class ISICDataset(data.Dataset):

    def __init__(self, root, image_root=None, gt_root=None, trainsize=256, augmentations=None,
                 train=True, sam_trans=None, image_size=(256, 256), ds_mean=IMAGENET_MEAN,
                 ds_std=IMAGENET_STD, base_path=None, fold=None, category=None,
                 data_root=None, superpixel_mode='upstream'):
        self.trainsize = trainsize
        self.augmentations = augmentations
        self.base_path = resolve_isic_setting_1_base_path(base_path or root)
        self.fold = fold
        self.category = category
        self.data_root = project_path(data_root, DEFAULT_DATA_ROOT)
        if superpixel_mode not in ('upstream', 'corrected'):
            raise ValueError('superpixel_mode must be upstream or corrected')
        self.superpixel_mode = superpixel_mode
        self.label_source = 'slic_superpixels' if train else 'lesion_ground_truth'

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size

        if image_root is not None and gt_root is not None:
            self.images = [
                os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
            self.gts = [
                os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')]

            for subdir in os.listdir(image_root):
                if not os.path.isdir(os.path.join(image_root, subdir)):
                    continue
                subdir_image_root = os.path.join(image_root, subdir)
                subdir_gt_root = os.path.join(gt_root, subdir)

                self.images.extend([
                    os.path.join(subdir_image_root, f)
                    for f in os.listdir(subdir_image_root)
                    if f.endswith('.jpg') or f.endswith('.png')
                ])

                self.gts.extend([
                    os.path.join(subdir_gt_root, f)
                    for f in os.listdir(subdir_gt_root)
                    if f.endswith('.png')
                ])
        elif fold is not None:
            self.images, self.gts = self.get_image_gt_pairs_from_json(
                self.base_path,
                fold=fold,
                split="train" if train else "test",
                category=category
            )
        else:
            self.images, self.gts = self.get_image_gt_pairs(
                root, split="train" if train else "test")

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        self.mean = torch.tensor(ds_mean).view(3, 1, 1)
        self.std = torch.tensor(ds_std).view(3, 1, 1)

        self.size = len(self.images)
        if not self.size:
            raise ValueError('No ISIC images found for fold/category selection')
        if not train and self.size < 2:
            raise ValueError('Evaluation needs two different images per category')
        missing = [path for path in self.images + self.gts if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError('Missing {} data files; first: {}. Run '
                                    'data/isic/prepare_setting1_dataset.py first.'.format(
                                        len(missing), missing[0]))
        self.train = train
        self.sam_trans = sam_trans

        if self.sam_trans is not None:
            self.mean = torch.tensor([0.0, 0.0, 0.0]).view(3, 1, 1)
            self.std = torch.tensor([1.0, 1.0, 1.0]).view(3, 1, 1)

    def get_image_gt_pairs(self, dir_root: str, split="train"):
        image_paths = []
        gt_paths = []

        for folder in os.listdir(dir_root):
            split_file = os.path.join(dir_root, folder, "split.txt")

            if os.path.isfile(split_file):
                image_root = os.path.join(dir_root, folder, "images")
                gt_root = os.path.join(dir_root, folder, "masks")

                image_paths_tmp, gt_paths_tmp = self.get_image_gt_pairs_from_text_file(
                    image_root, gt_root, split_file, split=split)

                image_paths.extend(image_paths_tmp)
                gt_paths.extend(gt_paths_tmp)

        return image_paths, gt_paths

    def get_image_gt_pairs_from_text_file(self, image_root, gt_root, text_file, split="train"):
        splits = {"train": [], "val": [], "test": []}
        current_split = None

        with open(text_file, 'r') as file:
            for line in file:
                line = line.strip()
                if line in splits:
                    current_split = line
                elif line and current_split:
                    splits[current_split].append(line)

        file_names = splits[split]

        image_paths = [os.path.join(image_root, name + '.png') for name in file_names]
        gt_paths = [os.path.join(gt_root, name + '.png') for name in file_names]

        return image_paths, gt_paths

    def get_image_gt_pairs_from_json(self, base_path, fold, split="train", category=None):
        fold_splits = load_isic_setting_1_split(base_path)
        fold_key = resolve_isic_setting_1_fold(fold_splits, fold)

        if split == "train":
            image_ids = []
            for current_fold, current_split in fold_splits.items():
                if current_fold == fold_key:
                    continue
                for current_category in get_isic_setting_1_categories(base_path, fold_splits):
                    image_ids.extend(current_split.get(current_category, []))
            gt_root = os.path.join(base_path, 'superpixels')
            gt_suffix = '_mask.png'
        elif split == "test":
            target_categories = [category] if category is not None else get_isic_setting_1_categories(base_path, fold_splits)
            invalid_categories = [cat for cat in target_categories if cat not in fold_splits[fold_key]]
            if invalid_categories:
                raise ValueError(f'Unknown ISIC setting 1 categories: {invalid_categories}')
            image_ids = []
            for current_category in target_categories:
                image_ids.extend(fold_splits[fold_key].get(current_category, []))
            _, gt_root = image_and_mask_roots(self.data_root)
            gt_suffix = '_segmentation.png'
        else:
            raise ValueError(f'Unsupported split "{split}" for ISIC setting 1.')

        image_root, _ = image_and_mask_roots(self.data_root)
        image_ids = sorted(image_ids)
        image_paths = [str(find_data_file(image_root, image_id, '.jpg')) for image_id in image_ids]
        gt_paths = [os.path.join(gt_root, image_id + gt_suffix) if split == 'train'
                    else str(find_data_file(gt_root, image_id, gt_suffix)) for image_id in image_ids]

        return image_paths, gt_paths

    def normalize(self, image):
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = (image - self.mean) / self.std
        return image

    def process_image_gt(self, image, gt, dataset=""):
        original_size = tuple(image.shape[:2])

        if self.augmentations:
            image, mask = self.augmentations(image, gt)
        else:
            mask = gt

        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        mask = cv2.resize(mask, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)

        image = self.normalize(image)

        mask = (mask > 0.5).astype(np.float32)
        mask = torch.from_numpy(mask)

        return {
            'image': self.sam_trans.preprocess(image).squeeze(0) if self.sam_trans else image,
            'label': self.sam_trans.preprocess(mask) if self.sam_trans else mask,
            'original_size': torch.tensor(original_size),
            'image_size': torch.tensor(self.image_size),
            'case': dataset
        }

    def get_dataset_name_from_path(self, path):
        return ""

    def __getitem__(self, index):
        image = self.cv2_loader(self.images[index], is_mask=False)
        gt = self.cv2_loader(self.gts[index], is_mask=True)
        return self.process_image_gt(image, gt, "")

    def cv2_loader(self, path, is_mask):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if is_mask else cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError('Cannot read image/mask: {}'.format(path))
        if is_mask:
            img[img > 0] = 1
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __len__(self):
        return self.size


class SuperpixISICDataset(ISICDataset):

    @staticmethod
    def minmax(image):
        if torch.is_tensor(image):
            return (image - image.min()) / (image.max() - image.min()).clamp_min(1e-8)
        return (image - image.min()) / max(float(image.max() - image.min()), 1e-8)

    def resize_image(self, image):
        return cv2.resize(image, (self.image_size[1], self.image_size[0]))

    def __getitem__(self, index):
        if self.train:
            image = self.cv2_loader(self.images[index], is_mask=False)
            gt = self.cv2_loader(self.gts[index], is_mask=False)

            image = self.resize_image(image)
            if self.superpixel_mode == 'corrected':
                gt = cv2.resize(gt, (self.image_size[1], self.image_size[0]),
                                interpolation=cv2.INTER_NEAREST)
            else:
                gt = self.resize_image(gt)

            gt = gt[:, :, 0]

            fgpath = os.path.basename(self.gts[index]).split('.png')[0]
            fgpath = os.path.join(os.path.dirname(self.gts[index]), fgpath + '.png')

            fg = self.cv2_loader(fgpath, is_mask=True)
            fg = self.resize_image(fg)

            if self.superpixel_mode == 'upstream':
                gt[1 - fg] = 0  # Intentional upstream indexing; recorded in experiment config.
            else:
                gt[fg == 0] = 0

            unique_ids = np.unique(gt)
            unique_ids = unique_ids[unique_ids != 0]
            if not len(unique_ids):
                raise ValueError('No usable superpixel IDs in {}'.format(self.gts[index]))

            sp_id = random.choice(unique_ids.tolist())
            sp = (gt == sp_id).astype(np.uint8)

            out = self.process_image_gt(image, sp, "")
            support_image, support_sp = out["image"], out["label"]
            support_image = self.minmax(support_image)
            support_sp = self.erode_label(support_sp.unsqueeze(0), kernel_size=3).squeeze(0)

            out = self.process_image_gt(image, sp, "")
            query_image, query_sp = out["image"], out["label"]
            query_image = self.minmax(query_image)
            query_sp = self.erode_label(query_sp.unsqueeze(0), kernel_size=3).squeeze(0)

            batch = {
                "support_images": [[support_image]],
                "support_fg_labels": [[support_sp]],
                "query_images": [query_image],
                "query_labels": [query_sp],
                "scan_id": [""]
            }

        else:
            support_image = self.cv2_loader(self.images[index], is_mask=False)

            support_gt = self.cv2_loader(self.gts[index], is_mask=True)
            support_image = self.resize_image(support_image)
            support_image = self.minmax(support_image)
            support_gt = self.resize_image(support_gt)
            support_sp = (support_gt > 0).astype(np.uint8)  # Foreground and background only

            query_index = (index + 1) % self.size  # Use the next image as the query image
            query_image = self.cv2_loader(self.images[query_index], is_mask=False)

            query_gt = self.cv2_loader(self.gts[query_index], is_mask=True)
            query_image = self.resize_image(query_image)
            query_image = self.minmax(query_image)
            query_gt = self.resize_image(query_gt)
            query_sp = (query_gt > 0).astype(np.uint8)  # Foreground and background only

            dataset = self.get_dataset_name_from_path(self.images[index])

            batch = {
                "support_images": [[support_image.transpose(2, 0, 1)]],
                "support_fg_labels": [[support_sp]],
                "query_images": [query_image.transpose(2, 0, 1)],
                "query_labels": [query_sp],
                "scan_id": [dataset],
                "support_name": os.path.splitext(os.path.basename(self.images[index]))[0],
                "query_name": os.path.splitext(os.path.basename(self.images[query_index]))[0],
            }

        return batch

    def erode_label(self, label, kernel_size=3):
        label_erode = F.max_pool2d(1 - label, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        return 1 - label_erode


def get_superpix_isic_dataset(image_size=(256,256), sam_trans=None, image_root=None, gt_root=None,
                              train=True, base_path=None, fold=None, category=None,
                              data_root=None, superpixel_mode='upstream'):
    transform_train, transform_test = get_polyp_transform()

    ds_train = SuperpixISICDataset(
        root=base_path or image_root,
        image_root=image_root,
        gt_root=gt_root,
        augmentations=transform_train,
        sam_trans=sam_trans,
        image_size=image_size,
        train=train,
        base_path=base_path,
        fold=fold,
        category=category,
        data_root=data_root,
        superpixel_mode=superpixel_mode,
    )

    return ds_train
