# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

pytest.importorskip('mmcv')

from mmdet.models.detectors.m2_pf_rpn import M2PFRPN  # noqa: E402
from mmdet.models.layers.multimodal_fusion import (  # noqa: E402
    GatedMultiModalFusion, LightweightModalityStem)


def _feature_pyramid(channels, batch_size=2, size=16):
    feats = []
    for level, channel in enumerate(channels):
        spatial = size // (2 ** (level + 1))
        feats.append(torch.randn(batch_size, channel, spatial, spatial))
    return tuple(feats)


def test_lightweight_modality_stem():
    stem = LightweightModalityStem(2, [8, 16, 32])
    outputs = stem(torch.randn(2, 2, 64, 64))
    assert [tuple(x.shape) for x in outputs] == [
        (2, 8, 8, 8), (2, 16, 4, 4), (2, 32, 2, 2)]


def test_gated_fusion_missing_modality():
    fusion = GatedMultiModalFusion(
        in_channels=[[8, 16, 32], [8, 16, 32], [8, 16, 32]],
        out_channels=[8, 16, 32],
        token_dim=16,
        num_modalities=3)
    feats = (
        _feature_pyramid([8, 16, 32]),
        _feature_pyramid([8, 16, 32]),
        _feature_pyramid([8, 16, 32]),
    )
    mask = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    fused, tokens = fusion(feats, mask)
    assert len(fused) == 3
    assert tuple(tokens.shape) == (2, 3, 16)
    # Missing modality token must stay zero.
    assert torch.allclose(tokens[0, 1], torch.zeros_like(tokens[0, 1]))


def test_level_assignment():
    bboxes = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],
        [0.0, 0.0, 40.0, 40.0],
        [0.0, 0.0, 120.0, 120.0],
    ])
    levels = M2PFRPN._assign_levels(bboxes)
    assert levels.tolist() == [0, 1, 2]