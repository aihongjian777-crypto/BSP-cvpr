"""FoB_SAM SLIC generation with portable paths and a server preflight CLI.

Upstream: primebo1/FoB_SAM @ 40d0e23d75f81f07bb7f34cb641db51ef490d025.
The SLIC algorithm and its parameters are unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloaders.isic_paths import (
    DEFAULT_DATA_ROOT, DEFAULT_BASE_PATH, UPSTREAM_COMMIT, project_path,
    image_and_mask_roots, load_fold_splits, all_image_ids, find_data_file,
)


def generate_pseudo_mask(image_path, n_segments=5, compactness=15):
    import numpy as np
    from skimage.segmentation import slic
    from skimage.io import imread

    image = imread(str(image_path))
    segments = slic(image, n_segments=n_segments, compactness=compactness, start_label=1)
    return segments.astype(np.uint8)


def check_dataset(data_root=None, base_path=None, require_superpixels=False):
    base = project_path(base_path, DEFAULT_BASE_PATH)
    folds = load_fold_splits(base)
    images, masks = image_and_mask_roots(data_root)
    ids = all_image_ids(folds)
    missing = []
    for image_id in ids:
        for folder, suffix in ((images, '.jpg'), (masks, '_segmentation.png')):
            try:
                find_data_file(folder, image_id, suffix)
            except FileNotFoundError as error:
                missing.append(str(error))
        if require_superpixels and not (base / 'superpixels' / (image_id + '_mask.png')).is_file():
            missing.append('Missing pseudo-label: ' + image_id)
    if missing:
        raise FileNotFoundError('{} missing/ambiguous files; first errors:\n{}'.format(
            len(missing), '\n'.join(missing[:8])))
    counts = {fold: sum(len(values) for values in split.values()) for fold, split in folds.items()}
    print('ISIC Setting I: {} images and real masks; folds {}'.format(len(ids), counts))
    if require_superpixels:
        print('All {} SLIC pseudo-label files are present.'.format(len(ids)))
    print('Image source: {}'.format(images))
    print('Pseudo-label destination: {}'.format(base / 'superpixels'))
    return images, ids


def generate_all_superpixels(data_root=None, base_path=None, overwrite=False):
    import skimage
    from skimage.io import imsave
    from tqdm import tqdm

    images, ids = check_dataset(data_root, base_path)
    base = project_path(base_path, DEFAULT_BASE_PATH)
    destination = base / 'superpixels'
    destination.mkdir(parents=True, exist_ok=True)
    metadata_path = destination / 'generation.json'
    signature = {'method': 'skimage.segmentation.slic', 'n_segments': 5,
                 'compactness': 15, 'start_label': 1, 'skimage_version': skimage.__version__,
                 'upstream_commit': UPSTREAM_COMMIT}
    metadata = {'signature': signature, 'source_sha256': {}}
    if metadata_path.is_file() and not overwrite:
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if metadata.get('signature') != signature:
            raise ValueError('Existing pseudo-label parameters/library version differ. '
                             'Use --overwrite to regenerate the complete cache.')
    elif any(destination.glob('*_mask.png')) and not overwrite:
        raise ValueError('Existing pseudo-labels have no generation metadata; use --overwrite '
                         'to explicitly regenerate them with the original SLIC method.')
    generated = skipped = 0
    for image_id in tqdm(ids, desc='FoB_SAM SLIC'):
        image_path = find_data_file(images, image_id, '.jpg')
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        output = destination / (image_id + '_mask.png')
        previous = metadata['source_sha256'].get(image_id)
        if output.is_file() and previous == digest and not overwrite:
            skipped += 1
            continue
        if previous is not None and previous != digest and not overwrite:
            raise ValueError('Source image changed: {}. Use --overwrite.'.format(image_id))
        mask = generate_pseudo_mask(image_path)
        imsave(str(output), mask, check_contrast=False)
        metadata['source_sha256'][image_id] = digest
        generated += 1
        temp_path = metadata_path.with_suffix('.json.tmp')
        temp_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        temp_path.replace(metadata_path)
    print('Generated: {}; reused: {}. Original images/masks were not modified.'.format(
        generated, skipped))
    return generated, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--base-path', required=True)
    parser.add_argument('--check-only', action='store_true', help='Read-only file/manifest check')
    parser.add_argument('--require-superpixels', action='store_true')
    parser.add_argument('--overwrite', action='store_true', help='Regenerate existing pseudo-labels')
    args = parser.parse_args()
    if args.check_only:
        check_dataset(args.data_root, args.base_path, args.require_superpixels)
    else:
        generate_all_superpixels(args.data_root, args.base_path, args.overwrite)


if __name__ == '__main__':
    main()
