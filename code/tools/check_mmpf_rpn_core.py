# Copyright (c) OpenMMLab. All rights reserved.
"""Dependency-light smoke test for M2-PF-RPN core tensor paths.

This script intentionally stubs only the MMDetection registry/structures and
the parent PFRPN class.  It executes the real multimodal fusion and M2PFRPN
files so channel, scale and missing-modality errors can be checked without a
full CUDA/MMCV installation.
"""
import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]


class DummyRegistry:

    def register_module(self, *args, **kwargs):
        if args and isinstance(args[0], type):
            return args[0]

        def deco(cls):
            return cls
        return deco


class DummyBackbone(nn.Module):

    def __init__(self, out_channels=(16, 32, 64)):
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x):
        size = 64
        feats = []
        for idx, channels in enumerate(self.out_channels):
            spatial = size // (2 ** (3 + idx))
            feats.append(
                torch.zeros(
                    x.shape[0], channels, spatial, spatial,
                    device=x.device, dtype=x.dtype) + x.mean() * 0.0)
        return tuple(feats)


class DummyNeck(nn.Module):

    def forward(self, feats):
        return tuple(feats)


class DummyPFRPN(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.embed_dims = 16
        self.backbone = DummyBackbone()
        self.neck = DummyNeck()
        self.with_neck = True
        self.topk = kwargs.get('topk', 2)
        self.num_pseudo_tokens = kwargs.get('num_pseudo_tokens', 3)
        self.sp_thr = kwargs.get('sp_thr', 0.3)
        self.sp_iter_num = kwargs.get('sp_iter_num', 3)


def _install_stubs():
    mmdet = types.ModuleType('mmdet')
    mmdet.__path__ = []
    models = types.ModuleType('mmdet.models')
    models.__path__ = []
    detectors = types.ModuleType('mmdet.models.detectors')
    detectors.__path__ = []
    layers = types.ModuleType('mmdet.models.layers')
    layers.__path__ = []
    registry = types.ModuleType('mmdet.registry')
    structures = types.ModuleType('mmdet.structures')

    registry.MODELS = DummyRegistry()
    structures.OptSampleList = object
    structures.SampleList = object

    sys.modules.update({
        'mmdet': mmdet,
        'mmdet.models': models,
        'mmdet.models.detectors': detectors,
        'mmdet.models.layers': layers,
        'mmdet.registry': registry,
        'mmdet.structures': structures,
    })
    models.detectors = detectors
    models.layers = layers


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_install_stubs()
fusion_module = _load_module(
    'mmdet.models.layers.multimodal_fusion',
    ROOT / 'mmdet' / 'models' / 'layers' / 'multimodal_fusion.py')

pf_rpn = types.ModuleType('mmdet.models.detectors.pf_rpn')
pf_rpn.PFRPN = DummyPFRPN
sys.modules['mmdet.models.detectors.pf_rpn'] = pf_rpn

detector_module = _load_module(
    'mmdet.models.detectors.m2_pf_rpn',
    ROOT / 'mmdet' / 'models' / 'detectors' / 'm2_pf_rpn.py')

stem = fusion_module.LightweightModalityStem(2, [16, 32, 64])
stem_out = stem(torch.randn(2, 2, 64, 64))
assert [tuple(x.shape) for x in stem_out] == [
    (2, 16, 8, 8), (2, 32, 4, 4), (2, 64, 2, 2)]

fusion = fusion_module.GatedMultiModalFusion(
    in_channels=[[16, 32, 64], [8, 16, 32], [16, 32, 64]],
    out_channels=[16, 32, 64],
    token_dim=32,
    num_modalities=3)
feats = (
    tuple(torch.randn(2, c, 8 // (2**i), 8 // (2**i))
          for i, c in enumerate([16, 32, 64])),
    tuple(torch.randn(2, c, 8 // (2**i), 8 // (2**i))
          for i, c in enumerate([8, 16, 32])),
    tuple(torch.randn(2, c, 8 // (2**i), 8 // (2**i))
          for i, c in enumerate([16, 32, 64])),
)
mask = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
fused, tokens = fusion(feats, mask)
assert [tuple(x.shape) for x in fused] == [
    (2, 16, 8, 8), (2, 32, 4, 4), (2, 64, 2, 2)]
assert tuple(tokens.shape) == (2, 3, 32)
assert torch.allclose(tokens[0, 1], torch.zeros_like(tokens[0, 1]))

model = detector_module.M2PFRPN(
    modality_channels=dict(optical=3, sar=4, topography=5, change=3),
    modality_order=('optical', 'sar', 'topography', 'change'),
    neck=dict(in_channels=[16, 32, 64]),
    fusion_cfg=dict(
        in_channels=[[16, 32, 64]] * 4,
        out_channels=[16, 32, 64],
        token_dim=16))
inputs = torch.randn(2, 15, 64, 64)
model_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0]])
feats, tokens = model.extract_feat_with_modality_tokens(inputs, model_mask)
assert [tuple(x.shape) for x in feats] == [
    (2, 16, 8, 8), (2, 32, 4, 4), (2, 64, 2, 2)]
assert tuple(tokens.shape) == (2, 4, 16)
assert torch.allclose(tokens[0, 3], torch.zeros_like(tokens[0, 3]))

levels = detector_module.M2PFRPN._assign_levels(
    torch.tensor([[0., 0., 10., 10.], [0., 0., 40., 40.],
                  [0., 0., 120., 120.]]))
assert levels.tolist() == [0, 1, 2]
model_loss = sum(x.mean() for x in feats) + tokens.mean()
model_loss.backward()
print('M2-PF-RPN core smoke test: OK')