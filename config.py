import glob
import itertools
import os
import ast
from pathlib import Path
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
from utils import *

sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
sacred.SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment("BSP", save_git_info=False)
ex.captured_out_filter = apply_backspaces_and_linefeeds


source_folders = ['.', './dataloaders', './models', './utils', './scripts/isic']
sources_to_save = list(itertools.chain.from_iterable(
    [glob.glob(f'{folder}/*.py') for folder in source_folders]))
for source_file in sources_to_save:
    ex.add_source_file(source_file)


@ex.config
def cfg():

    seed = 2025
    ablation_variant = 'full'
    gpu_id = 0
    num_workers = 1
    mode = 'train'

    dataset = 'CHAOST2'
    exclude_label = None
    if dataset == 'CMR':
        n_sv = 1000
    else:
        n_sv = 5000
    min_size = 200
    max_slices = 3
    use_gt = False
    allow_gt_fallback = False
    exclude_label_mode = 'all'
    superpix_scale = 'MIDDLE'
    data_root = None
    pretrained_weights = None
    allow_partial_checkpoint = False
    eval_fold = 0
    test_label = [1, 2, 3, 4]
    supp_idx = 1
    n_part = 3

    isic_setting = 1
    isic_setting_1_base_path = None
    isic_image_size = 256
    isic_superpixel_mode = 'upstream'
    save_isic_predictions = True

    n_steps = 50000
    batch_size = 1
    n_shot = 1
    n_way = 1
    n_query = 1
    lr_step_gamma = 0.9
    bg_wt = 0.1
    t_loss_scaler = 0.0
    ignore_label = 255
    print_interval = 100
    save_snapshot_every = 5000
    max_iters_per_load = 5000

    reload_model_path = None

    optim_type = 'sgd'
    optim = {
        'lr': 1e-3,
        'momentum': 0.9,
        'weight_decay': 0.0005,
    }

    exp_str = '_'.join(
        [mode]
        + [dataset, ]
        + [f'cv{eval_fold}'])

    path = {
        'log_dir': './runs',
        'CHAOST2': {'data_dir': None},
        'SABS': {'data_dir': None},
        'CMR': {'data_dir': None},
        'isic': {'data_dir': None},
    }


@ex.named_config
def isic_setting1():

    dataset = 'isic'
    isic_setting = 1
    eval_fold = 1
    test_label = None
    exclude_label = None
    use_gt = False
    n_way = 1
    n_shot = 1
    n_query = 1
    batch_size = 1
    n_part = 1
    path = {'log_dir': str(Path(__file__).resolve().parent / 'exps_on_ISIC2018')}


@ex.named_config
def ablation_full():
    ablation_variant = 'full'


@ex.named_config
def ablation_wo_sepc():
    ablation_variant = 'wo_sepc'


@ex.named_config
def ablation_wo_qrr():
    ablation_variant = 'wo_qrr'


@ex.named_config
def ablation_wo_swpa():
    ablation_variant = 'wo_swpa'


@ex.named_config
def ablation_sepc_only():
    ablation_variant = 'sepc_only'


@ex.config_hook
def add_observer(config, command_name, logger):

    if config.get('data_root'):
        config['data_root'] = str(Path(config['data_root']).expanduser().resolve())
        config['path'].setdefault(config['dataset'], {})['data_dir'] = config['data_root']
    for key in ('pretrained_weights', 'isic_setting_1_base_path', 'reload_model_path'):
        if config.get(key):
            config[key] = str(Path(config[key]).expanduser().resolve())
    for key in ('test_label', 'exclude_label'):
        if isinstance(config.get(key), str):
            config[key] = ast.literal_eval(config[key])
    if config['dataset'] == 'isic':
        if config['isic_setting'] != 1:
            raise ValueError('This BSP integration implements ISIC Setting I only')
        if config['eval_fold'] not in (1, 2, 3, 4, 5):
            raise ValueError('ISIC Setting I eval_fold must be 1..5')
        if any(config[key] != 1 for key in ('n_way', 'n_shot', 'n_query')):
            raise ValueError('The current BSP model requires 1 way, 1 shot and 1 query')
        if config['use_gt'] or config['exclude_label']:
            raise ValueError('ISIC Setting I trains with SLIC labels and all disease categories')
        config['exp_str'] = '{}_isic_setting1_cv{}'.format(config['mode'], config['eval_fold'])
    from models.ablation import module_flags
    config["ablation_modules"] = module_flags(config["ablation_variant"])
    config["exp_str"] += "_{}_seed{}".format(config["ablation_variant"], config["seed"])
    exp_name = f'{ex.path}_{config["exp_str"]}'
    observer = FileStorageObserver.create(os.path.join(config['path']['log_dir'], exp_name))
    ex.observers.append(observer)
    return config
