"""Named module controls and checkpoint provenance, without ML dependencies."""
import json
from pathlib import Path


VARIANTS = {
    'full': {'sepc': True, 'qrr': True, 'swpa': True},
    'wo_sepc': {'sepc': False, 'qrr': True, 'swpa': True},
    'wo_qrr': {'sepc': True, 'qrr': False, 'swpa': True},
    'wo_swpa': {'sepc': True, 'qrr': True, 'swpa': False},
    'sepc_only': {'sepc': True, 'qrr': False, 'swpa': False},
}


def module_flags(variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown ablation variant {!r}; choose {}'.format(variant, ', '.join(VARIANTS)))
    return dict(VARIANTS[variant])


def save_metadata(snapshot_dir, config):
    metadata = {
        'variant': config['ablation_variant'],
        'modules': module_flags(config['ablation_variant']),
        'dataset': config['dataset'], 'fold': config['eval_fold'], 'seed': config['seed'],
        'data_root': config.get('path', {}).get(config['dataset'], {}).get('data_dir'),
        'pretrained_weights': config.get('pretrained_weights'),
        'sepc_off': 'global context; same-class random replacement; retain MLP and boundary prototypes',
        'qrr_off': 'identity on normalized prototypes',
        'swpa_off': 'uniform mean of prototype features and scaled cosine responses',
    }
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (snapshot_dir / 'ablation.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def validate_checkpoint(checkpoint, config):
    """State shapes alone cannot distinguish the ablation configurations."""
    sidecar = Path(checkpoint).parent / 'ablation.json'
    if not sidecar.is_file():
        if config['ablation_variant'] != 'full':
            raise ValueError('Ablated checkpoints require ablation.json beside the weights; '
                             'do not evaluate full-model weights as an ablation.')
        return False  # Original main-branch checkpoints are supported for full only.
    metadata = json.loads(sidecar.read_text(encoding='utf-8'))
    expected = {
        'variant': config['ablation_variant'],
        'modules': module_flags(config['ablation_variant']),
        'dataset': config['dataset'], 'fold': config['eval_fold'], 'seed': config['seed'],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError('Checkpoint {} mismatch: recorded {!r}, requested {!r}'.format(
                key, metadata.get(key), value))
    return True
