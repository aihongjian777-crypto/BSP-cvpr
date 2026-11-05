"""Checks for explicit resource paths; no dataset or ML installation needed."""
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.runtime_paths import validate_runtime_paths
from dataloaders.isic_paths import project_path
from scripts.run_ablation import build_command


class RuntimePathTests(unittest.TestCase):
    def test_missing_data_never_falls_back_to_repository(self):
        with self.assertRaisesRegex(ValueError, 'data_root'):
            validate_runtime_paths({'dataset': 'CHAOST2', 'path': {}}, True)
        with self.assertRaises(ValueError):
            project_path(None, None)

    def test_external_files_train_and_test_have_distinct_requirements(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            data = base / 'dataset outside code'
            data.mkdir()
            weights = base / 'pretrained.pth'
            config = dict(dataset='CHAOST2', path={'CHAOST2': {'data_dir': str(data)}})
            with self.assertRaisesRegex(ValueError, 'pretrained_weights'):
                validate_runtime_paths(config, True)
            config['pretrained_weights'] = str(weights)
            with self.assertRaises(FileNotFoundError):
                validate_runtime_paths(config, True)
            weights.touch()
            validate_runtime_paths(config, True)
            config['pretrained_weights'] = None
            config['reload_model_path'] = str(weights)
            validate_runtime_paths(config, False)

    def test_isic_requires_external_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            weights = root / 'weights.pth'
            weights.touch()
            config = dict(dataset='isic', path={'isic': {'data_dir': folder}},
                          pretrained_weights=str(weights))
            with self.assertRaisesRegex(FileNotFoundError, 'isic_setting_1_base_path'):
                validate_runtime_paths(config, True)
            config['isic_setting_1_base_path'] = folder
            (root / 'isic_in_5_folds.json').write_text('{}')
            validate_runtime_paths(config, True)

    def test_launcher_transmits_paths_with_spaces_as_single_arguments(self):
        args = SimpleNamespace(mode='train', dataset='CHAOST2', fold=2, seed=1234,
                               steps=50000, workers=4, data_root=str(ROOT / 'tmp/data with spaces'),
                               pretrained_weights=str(ROOT / 'tmp/weights with spaces.pth'),
                               isic_base_path=None)
        command = build_command(args, 'sepc_only')
        self.assertEqual(len([item for item in command if item.startswith('data_root=')]), 1)
        self.assertEqual(len([item for item in command if item.startswith('pretrained_weights=')]), 1)
        self.assertIn('ablation_variant=sepc_only', command)


if __name__ == '__main__':
    unittest.main()
