import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.ablation import VARIANTS


def build_command(args, variant):
    command = [sys.executable, str(ROOT / (args.mode + '.py')), 'with']
    if args.dataset == 'isic':
        command.append('isic_setting1')
    command.extend([
        'ablation_variant=' + variant,
        'mode=' + args.mode, 'dataset=' + args.dataset,
        'eval_fold=' + str(args.fold), 'seed=' + str(args.seed),
        'n_steps=' + str(args.steps), 'num_workers=' + str(args.workers),
        'path.log_dir=' + json.dumps(str(ROOT / ('exps_on_ABLATION' if args.mode == 'train'
                                                 else 'results/ablation'))),
    ])
    command.append('data_root=' + json.dumps(str(Path(args.data_root).expanduser().resolve())))
    if args.pretrained_weights:
        command.append('pretrained_weights=' + json.dumps(str(Path(args.pretrained_weights).expanduser().resolve())))
    if args.isic_base_path:
        command.append('isic_setting_1_base_path=' + json.dumps(str(Path(args.isic_base_path).expanduser().resolve())))
    if args.mode == 'test':
        command.extend(['reload_model_path=' + json.dumps(str(Path(args.checkpoint).resolve())),
                        'supp_idx=' + str(args.support_index)])
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train', 'test'])
    parser.add_argument('--variant', choices=list(VARIANTS) + ['all'], default='full')
    parser.add_argument('--dataset', choices=['CHAOST2', 'CMR', 'SABS', 'isic'], default='CHAOST2')
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--pretrained-weights')
    parser.add_argument('--isic-base-path')
    parser.add_argument('--checkpoint')
    parser.add_argument('--support-index', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.mode == 'train' and not args.pretrained_weights:
        parser.error('train requires --pretrained-weights /external/weights.pth')
    if args.dataset == 'isic' and not args.isic_base_path:
        parser.error('isic requires --isic-base-path /external/ISIC_setting_1')
    if args.mode == 'test' and (not args.checkpoint or args.variant == 'all'):
        parser.error('test requires one --variant and its own --checkpoint')
    valid_folds = range(1, 6) if args.dataset == 'isic' else range(5)
    if args.fold not in valid_folds:
        parser.error('fold must be 1..5 for isic or 0..4 for MRI/CT')
    if args.steps <= 0 or args.workers < 0:
        parser.error('steps must be positive and workers nonnegative')
    variants = list(VARIANTS) if args.variant == 'all' else [args.variant]
    for variant in variants:
        command = build_command(args, variant)
        print(subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=str(ROOT), check=True)


if __name__ == '__main__':
    main()
