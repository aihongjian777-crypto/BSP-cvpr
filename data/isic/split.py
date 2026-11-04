"""Optional FoB_SAM class_id.csv categorization with portable source paths.

The active Setting I loader reads categories from JSON, so this copy is optional.
Source: primebo1/FoB_SAM @ 40d0e23d75f81f07bb7f34cb641db51ef490d025.
"""
import argparse
import csv
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from dataloaders.isic_paths import (
    DEFAULT_DATA_ROOT, DEFAULT_BASE_PATH, CATEGORIES, image_and_mask_roots,
    project_path, find_data_file,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default=str(DEFAULT_DATA_ROOT))
    parser.add_argument('--output-root', default=str(DEFAULT_BASE_PATH / 'categorized'))
    args = parser.parse_args()
    images, masks = image_and_mask_roots(args.data_root)
    destination = project_path(args.output_root, DEFAULT_BASE_PATH / 'categorized')
    with (Path(__file__).resolve().parent / 'class_id.csv').open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    pairs = []
    for row in rows:
        if row['Class'] not in CATEGORIES:
            raise ValueError('Unknown category: ' + row['Class'])
        for folder, suffix, kind in ((images, '.jpg', 'images'),
                                     (masks, '_segmentation.png', 'masks')):
            source = find_data_file(folder, row['ID'], suffix)
            target = destination / kind / row['Class'] / source.name
            pairs.append((source, target))
    for source, target in pairs:
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            shutil.copy2(str(source), str(target))
    print('Categorized {} images and masks under {}'.format(len(rows), destination))


if __name__ == '__main__':
    main()
