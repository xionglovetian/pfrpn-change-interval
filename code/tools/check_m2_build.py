# Copyright (c) OpenMMLab. All rights reserved.
"""Build the M2-PF-RPN model from a config without running inference."""
import argparse

from mmengine.config import Config

import mmdet.models  # noqa: F401
import mmdet.datasets  # noqa: F401
import mmdet.models.detectors.m2_pf_rpn  # noqa: F401
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config',
        default='configs/m2-pf-rpn/m2_pf_rpn_sen12landslides_smoke.py',
        nargs='?')
    args = parser.parse_args()
    register_all_modules()
    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    print('model:', type(model).__name__)
    print('parameters:', total)
    print('trainable:', trainable)


if __name__ == '__main__':
    main()