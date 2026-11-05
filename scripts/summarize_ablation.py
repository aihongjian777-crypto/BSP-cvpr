"""Export class-macro Dice rows for one dataset, requiring matched evaluations."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.ablation import VARIANTS


def summarize(result_paths):
    grouped = {variant: {} for variant in VARIANTS}
    classes = None
    datasets = set()
    for path in result_paths:
        record = json.loads(Path(path).read_text(encoding='utf-8'))
        variant = record['ablation_variant']
        if variant not in grouped:
            raise ValueError('Unknown variant in ' + str(path))
        if 'categories' in record:
            raise ValueError('Use ablation_results.json for the MRI/CT table; ISIC has its own result format.')
        datasets.add(record['dataset'])
        values = record['class_dice']
        if not values or (classes is not None and set(classes) != set(values)):
            raise ValueError('Class sets must match across every run')
        classes = list(values) if classes is None else classes
        key = (record['fold'], record['seed'], record['support_index'])
        if key in grouped[variant]:
            raise ValueError('Duplicate evaluation for {} {}; select explicit run files'.format(variant, key))
        grouped[variant][key] = values
    if len(datasets) != 1:
        raise ValueError('Summarize exactly one dataset at a time')
    expected = set(grouped['full'])
    if not expected or any(set(runs) != expected for runs in grouped.values()):
        raise ValueError('All four variants must contain the same fold/seed/support-index evaluations')
    rows = []
    for variant, runs in grouped.items():
        row = {'variant': variant, **{name.upper(): int(flag) for name, flag in VARIANTS[variant].items()}}
        for name in classes:
            row[name] = 100 * sum(values[name] for values in runs.values()) / len(runs)
        row['Avg'] = sum(row[name] for name in classes) / len(classes)
        row['evaluations'] = len(runs)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path, nargs='+', help='Explicit ablation_results.json files')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/ablation/table.csv')
    args = parser.parse_args()
    rows = summarize(args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Saved class-macro Dice (%) table:', args.output)


if __name__ == '__main__':
    main()
