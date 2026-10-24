# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for missing-modality masking, incl. optical-missing support."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mmdet.models.detectors.m2_pf_rpn import M2PFRPN  # noqa: E402
from mmdet.models.layers.multimodal_fusion import (  # noqa: E402
    GatedMultiModalFusion)


class _FusionStub:
    num_modalities = 4


class _DropStub:
    def __init__(self, allow_optical_dropout, prob=1.0):
        self.allow_optical_dropout = allow_optical_dropout
        self.modality_dropout_prob = prob


class PrepareMaskTest(unittest.TestCase):

    def _prepare(self, mask):
        return GatedMultiModalFusion._prepare_mask(
            _FusionStub(), mask, batch_size=mask.shape[0],
            device=torch.device('cpu'))

    def test_full_mask_untouched(self):
        mask = torch.tensor([[1., 1., 1., 1.]])
        out = self._prepare(mask)
        self.assertTrue(torch.equal(out, mask))

    def test_optical_missing_is_honoured(self):
        mask = torch.tensor([[0., 1., 1., 0.]])
        out = self._prepare(mask)
        self.assertEqual(float(out[0, 0]), 0.0)
        self.assertEqual(float(out[0, 1]), 1.0)

    def test_all_missing_restores_optical(self):
        mask = torch.tensor([[0., 0., 0., 0.]])
        out = self._prepare(mask)
        self.assertEqual(float(out[0, 0]), 1.0)
        self.assertEqual(float(out[0].sum()), 1.0)

    def test_partial_batch(self):
        mask = torch.tensor([[0., 1., 1., 1.], [0., 0., 0., 0.]])
        out = self._prepare(mask)
        self.assertEqual(float(out[0, 0]), 0.0)
        self.assertEqual(float(out[1, 0]), 1.0)


class DropModalitiesTest(unittest.TestCase):

    def test_default_never_drops_optical(self):
        mask = torch.tensor([[1., 1., 1., 1.]] * 32)
        out = M2PFRPN._drop_modalities(_DropStub(False), mask)
        self.assertTrue(bool((out[:, 0] == 1).all()))
        # at least one modality was actually dropped
        self.assertTrue(bool((out.sum(dim=1) < 4).any()))

    def test_optical_dropout_can_drop_optical(self):
        torch.manual_seed(0)
        mask = torch.tensor([[1., 1., 1., 1.]] * 64)
        out = M2PFRPN._drop_modalities(_DropStub(True), mask)
        self.assertTrue(bool((out[:, 0] == 0).any()),
                        'optical should be droppable when enabled')

    def test_never_drops_last_modality(self):
        for allow in (True, False):
            mask = torch.tensor([[1., 0., 0., 0.]])
            out = M2PFRPN._drop_modalities(_DropStub(allow), mask)
            self.assertEqual(float(out.sum()), 1.0)

    def test_single_modality_returns_unchanged(self):
        mask = torch.tensor([[1.]])
        out = M2PFRPN._drop_modalities(_DropStub(True), mask)
        self.assertTrue(torch.equal(out, mask))


if __name__ == '__main__':
    unittest.main()