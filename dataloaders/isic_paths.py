import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = None
DEFAULT_BASE_PATH = None
UPSTREAM_COMMIT = '40d0e23d75f81f07bb7f34cb641db51ef490d025'
CATEGORIES = ('melanoma', 'nevus', 'seborrheic_keratosis')


def project_path(value, default):
    if value is None and default is None:
        raise ValueError('External path required; supply data_root and ISIC base path explicitly.')
    path = Path(value).expanduser() if value is not None else Path(default)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_fold_splits(base_path=None):
    base = project_path(base_path, DEFAULT_BASE_PATH)
    with (base / 'isic_in_5_folds.json').open(encoding='utf-8') as stream:
        folds = json.load(stream)
    if set(folds) != {'fold_{}'.format(i) for i in range(1, 6)}:
        raise ValueError('ISIC Setting I requires fold_1 through fold_5')
    seen = set()
    for name, split in folds.items():
        if set(split) != set(CATEGORIES):
            raise ValueError('Unexpected categories in {}'.format(name))
        for ids in split.values():
            for image_id in ids:
                if not isinstance(image_id, str) or Path(image_id).name != image_id:
                    raise ValueError('Invalid image ID in fold manifest')
                if image_id in seen:
                    raise ValueError('Duplicate image ID across folds: {}'.format(image_id))
                seen.add(image_id)
    return folds


def resolve_fold(folds, fold):
    key = str(fold)
    if key.isdigit():
        key = 'fold_{}'.format(int(key))
    if key not in folds:
        raise ValueError('ISIC Setting I fold must be 1..5; received {!r}'.format(fold))
    return key


def image_and_mask_roots(data_root=None):
    root = project_path(data_root, DEFAULT_DATA_ROOT)

    candidates = [(root / 'train' / 'images', root / 'train' / 'masks'),
                  (root / 'ISIC2018_Task1-2_Training_Input',
                   root / 'ISIC2018_Task1_Training_GroundTruth'),
                  (root / 'images', root / 'gt'),
                  (root / 'images', root / 'masks')]
    for images, masks in candidates:
        if images.is_dir() and masks.is_dir():
            return images, masks
    raise FileNotFoundError(
        'Missing ISIC training data under {}. Expected train/images/*.jpg and '
        'train/masks/*_segmentation.png; see ISIC2018_SERVER.md.'.format(root))


def find_data_file(folder, image_id, suffix):

    folder = Path(folder)
    direct = folder / (image_id + suffix)
    if direct.is_file():
        return direct
    matches = [folder / category / (image_id + suffix) for category in CATEGORIES]
    matches = [path for path in matches if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError('Expected exactly one file for {} under {}'.format(
            image_id + suffix, folder))
    return matches[0]


def all_image_ids(folds):
    return sorted(image_id for split in folds.values() for ids in split.values()
                  for image_id in ids)
