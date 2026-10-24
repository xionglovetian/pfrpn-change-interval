# Copyright (c) OpenMMLab. All rights reserved.
"""Regression guards for data-pipeline transform ordering.

A resize inserted *before* ``LoadAnnotations`` resizes the image but leaves
gt_bboxes in the original frame, which silently drives every IoU to ~0 while
still producing plausible-looking output.  These tests pin the required order
for every M1 config and for the eval/export dataset builder.
"""
import sys
import unittest
from pathlib import Path

from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

import eval_m1_split as E  # noqa: E402

CONFIGS = [
    'configs/m2-pf-rpn/m2_pf_rpn_multimodal.py',
    'configs/m2-pf-rpn/m2_pf_rpn_cloud_full.py',
    'configs/m2-pf-rpn/m2_pf_rpn_cloud_dsc.py',
]
RESIZE_TYPES = ('FixScaleResize', 'RandomChoiceResize', 'RandomResize',
                'Resize')


def _order(step_types, a, b):
    if a not in step_types or b not in step_types:
        return True  # not part of this pipeline
    return step_types.index(a) < step_types.index(b)


class PipelineOrderTest(unittest.TestCase):

    def test_training_pipelines_load_annotations_before_resize(self):
        for rel in CONFIGS[:2]:
            path = ROOT / rel
            if not path.exists():
                continue
            cfg = Config.fromfile(str(path))
            types = [t['type'] for t in cfg.train_dataloader.dataset.pipeline]
            resizes = [t for t in RESIZE_TYPES if t in types]
            self.assertTrue(
                resizes, f'{rel} has no resize step')
            for res in resizes:
                self.assertLess(
                    types.index('LoadAnnotations'), types.index(res),
                    f'{rel}: LoadAnnotations must precede {res}')

    def test_eval_builder_resizes_after_loading_annotations(self):
        cfg = Config.fromfile(str(ROOT / CONFIGS[1]))
        ds_cfg = E._build_eval_dataset_cfg(cfg, 'test', 320)
        types = [t['type'] for t in ds_cfg['pipeline']]
        self.assertIn('LoadAnnotations', types)
        self.assertIn('FixScaleResize', types)
        self.assertLess(types.index('LoadAnnotations'),
                        types.index('FixScaleResize'))
        # random augmentation must not leak into evaluation
        for banned in ('RandomFlip', 'RandomCrop', 'RandomChoiceResize'):
            self.assertNotIn(banned, types)

    def test_eval_builder_keeps_requested_split(self):
        cfg = Config.fromfile(str(ROOT / CONFIGS[1]))
        for split in ('val', 'test'):
            ds_cfg = E._build_eval_dataset_cfg(cfg, split, 320)
            self.assertEqual(ds_cfg['ann_file'], f'manifests/{split}.json')


if __name__ == '__main__':
    unittest.main()