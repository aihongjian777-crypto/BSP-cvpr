"""CPU checks for the actual model methods, without allocating the ResNet/MLPs."""
import ast
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.ablation import module_flags, save_metadata, validate_checkpoint
from scripts.summarize_ablation import summarize

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import cv2
except ImportError:
    torch = None


class ProvenanceTests(unittest.TestCase):
    def test_rejects_wrong_variant_fold_and_seed(self):
        config = dict(ablation_variant='wo_qrr', dataset='CHAOST2', eval_fold=0, seed=2025)
        with tempfile.TemporaryDirectory() as folder:
            save_metadata(folder, config)
            checkpoint = Path(folder) / '50000.pth'
            self.assertTrue(validate_checkpoint(checkpoint, config))
            for key, wrong in [('ablation_variant', 'full'), ('eval_fold', 1), ('seed', 99)]:
                with self.assertRaises(ValueError):
                    validate_checkpoint(checkpoint, dict(config, **{key: wrong}))

    def test_legacy_checkpoint_only_allowed_for_full(self):
        with tempfile.TemporaryDirectory() as folder:
            config = dict(ablation_variant='full')
            self.assertFalse(validate_checkpoint(Path(folder) / 'old.pth', config))
            with self.assertRaises(ValueError):
                validate_checkpoint(Path(folder) / 'old.pth', dict(ablation_variant='wo_sepc'))

    def test_summary_requires_matched_evaluations(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for variant in ('full', 'wo_sepc', 'wo_qrr', 'wo_swpa'):
                path = Path(folder) / (variant + '.json')
                path.write_text(json.dumps(dict(ablation_variant=variant, dataset='CHAOST2',
                    fold=0, seed=2025, support_index=1, class_dice={'LIVER': 0.8, 'RK': 0.6})))
                paths.append(path)
            rows = summarize(paths)
            self.assertAlmostEqual(rows[0]['Avg'], 70.0)
            with self.assertRaises(ValueError):
                summarize(paths[:-1])
            with self.assertRaises(ValueError):
                summarize(paths + paths[:1])


@unittest.skipIf(torch is None, 'PyTorch/OpenCV are required for numerical model checks')
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse((ROOT / 'models/cow.py').read_text(encoding='utf-8'))
        model_class = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                           and node.name == 'FewShotSeg')
        namespace = dict(torch=torch, nn=nn, F=F, np=np, cv2=cv2, random=random)
        exec(compile(ast.Module(body=[model_class], type_ignores=[]), 'model_methods', 'exec'), namespace)
        cls.model_class = namespace['FewShotSeg']
        torch.set_num_threads(1)

    def make_model(self, variant):
        model = self.model_class.__new__(self.model_class)
        nn.Module.__init__(model)
        for name, value in module_flags(variant).items():
            setattr(model, 'use_' + name, value)
        model.device = torch.device('cpu')
        model.scaler = 20.0
        model.ifr_chunk_size = 2
        model.fuse = nn.Conv2d(2, 1, 1, bias=False)
        model.fuse.weight.data[:] = torch.tensor([0.2, 0.7]).view(1, 2, 1, 1)
        return model

    def inputs(self):
        torch.manual_seed(7)
        return (torch.randn(6, 4), torch.randn(1, 4, 3, 2), torch.randn(1, 4, 2, 3),
                torch.ones(1, 4, 4), torch.tensor([0.4]), torch.tensor([-0.2]))

    def dense_reference(self, model, pts, query, support, mask, tq, ts):
        # Independent, unchunked definition of QRR followed by aggregation.
        base = F.normalize(pts[:-1], dim=-1)
        query_unit = F.normalize(query, dim=1)
        if model.use_qrr:
            cosine = torch.einsum('kc,nchw->knhw', base, query_unit)
            weights = torch.sigmoid(cosine * (1 + 20 * tq.sigmoid()))
            descriptor = torch.einsum('knhw,nchw->kc', weights, query) / (
                weights.sum(dim=(1, 2, 3)).view(-1, 1) + 1e-5)
            support_unit = F.normalize(support, dim=1).permute(0, 2, 3, 1).reshape(-1, 4)
            query_flat = query_unit.permute(0, 2, 3, 1).reshape(-1, 4)
            keep = F.interpolate(mask.unsqueeze(0), support.shape[-2:], mode='bilinear',
                                 align_corners=True).reshape(-1) > 0
            gate = torch.sigmoid((support_unit[keep] @ query_flat.t()).mean()) if keep.any() else 0
            refined = base + 0.2 * (1 - gate) * base + 0.7 * gate * descriptor
        else:
            refined = base
        logits = torch.einsum('kc,nchw->knhw', F.normalize(refined, dim=-1), query_unit)
        logits = logits * (1 + 20 * ts.sigmoid())
        weights = logits.softmax(dim=0) if model.use_swpa else torch.ones_like(logits) / len(refined)
        matched = torch.einsum('knhw,kc->nchw', weights, refined)
        response = (weights * logits).sum(dim=0).unsqueeze(1)
        return torch.cat([query, matched, response], dim=1)

    def test_all_query_paths_match_dense_definition_and_backpropagate(self):
        for variant in ('full', 'wo_sepc', 'wo_qrr', 'wo_swpa'):
            model = self.make_model(variant)
            inputs = list(self.inputs())
            inputs[0].requires_grad_()
            inputs[1].requires_grad_()
            actual = model.IFR_FD(*inputs)
            expected = self.dense_reference(model, *inputs)
            self.assertTrue(torch.allclose(actual, expected, atol=3e-6, rtol=3e-6), variant)
            actual.square().mean().backward()
            self.assertTrue(torch.isfinite(inputs[0].grad).all(), variant)
            self.assertTrue(torch.isfinite(inputs[1].grad).all(), variant)
            self.assertEqual(model.fuse.weight.grad is None, variant == 'wo_qrr')

    def test_no_qrr_never_calls_gate_and_handles_empty_support(self):
        model = self.make_model('wo_qrr')
        model.get_fuse_factor = lambda *a: self.fail('QRR gate should not execute')
        inputs = list(self.inputs())
        inputs[3].zero_()
        self.assertTrue(torch.isfinite(model.IFR_FD(*inputs)).all())

    def test_no_swpa_features_are_spatially_uniform(self):
        model = self.make_model('wo_swpa')
        actual = model.IFR_FD(*self.inputs())
        self.assertTrue(torch.allclose(actual[:, 4:8, :, :], actual[:, 4:8, :1, :1].expand(1, 4, 3, 2)))

    def test_no_sepc_support_context_uses_only_global_prototypes(self):
        model = self.make_model('wo_sepc')
        model.supp_decoder = nn.Identity()
        torch.manual_seed(3)
        feature, fg, bg = torch.randn(1, 4, 3, 2), torch.randn(5, 4), torch.randn(7, 4)
        actual = model.getSelfPred(feature, fg, bg)
        expected = torch.cat([feature, fg[-1].view(1, 4, 1, 1).expand_as(feature),
                              bg[-1].view(1, 4, 1, 1).expand_as(feature)], dim=1)
        self.assertTrue(torch.allclose(actual, expected))

    def test_no_sepc_samples_class_masks_and_preserves_counts(self):
        model = self.make_model('wo_sepc')
        # Avoid allocating the large MLPs; execute production mask/selection methods.
        model.get_fg_fts = lambda *args: torch.zeros(1).expand(1, 512, 256, 256)
        model.mlp1 = lambda *args: torch.zeros(512, 100)
        model.mlp2 = lambda *args: torch.zeros(512, 600)
        sampled = []
        def sample(features, mask, count):
            sampled.append((mask.clone(), count))
            return torch.zeros(count, 512)
        model.get_random_pts = sample
        features = torch.zeros(1, 512, 2, 2)
        mask = torch.zeros(1, 256, 256)
        mask[:, 64:192, 64:192] = 1
        fg = model.get_fg_pts(features, mask, torch.ones_like(mask))
        bg = model.get_bg_pts(features, mask, torch.zeros_like(mask))
        self.assertEqual(tuple(fg.shape), (102, 512))
        self.assertEqual(tuple(bg.shape), (602, 512))
        self.assertTrue(torch.equal(sampled[0][0], mask))
        self.assertTrue(torch.equal(sampled[1][0], 1 - mask))
        self.assertEqual([item[1] for item in sampled], [50, 100])


if __name__ == '__main__':
    unittest.main(verbosity=2)
