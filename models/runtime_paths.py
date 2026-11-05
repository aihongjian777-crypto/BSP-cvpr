"""Validate command-supplied resources before allocating a model or using CUDA."""
from pathlib import Path


def validate_runtime_paths(config, training):
    root = config['path'].get(config['dataset'], {}).get('data_dir')
    if not root:
        raise ValueError('Specify data_root=/path/to/dataset in the command.')
    if not Path(root).expanduser().is_dir():
        raise FileNotFoundError('Dataset directory does not exist: {}'.format(root))
    if training:
        weights = config.get('pretrained_weights')
        if not isinstance(weights, str) or not weights:
            raise ValueError('Training requires pretrained_weights=/path/to/resnet_weights.pth')
        if not Path(weights).expanduser().is_file():
            raise FileNotFoundError('Pretrained weights do not exist: {}'.format(weights))
    else:
        checkpoint = config.get('reload_model_path')
        if not checkpoint or not Path(checkpoint).expanduser().is_file():
            raise FileNotFoundError('Specify reload_model_path=/path/to/trained_checkpoint.pth')
    if config['dataset'] == 'isic':
        base = config.get('isic_setting_1_base_path')
        if not base or not (Path(base).expanduser() / 'isic_in_5_folds.json').is_file():
            raise FileNotFoundError('Specify isic_setting_1_base_path=/external/ISIC_setting_1 '
                                    'containing isic_in_5_folds.json and superpixels/')
