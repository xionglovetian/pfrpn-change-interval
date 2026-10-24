# Copyright (c) OpenMMLab. All rights reserved.
"""Run one real-data training loss calculation for M2-PF-RPN."""
import argparse

import torch
from mmengine.config import Config
from mmengine.dataset import default_collate

import mmdet.models  # noqa: F401
import mmdet.datasets  # noqa: F401
import mmdet.models.detectors.m2_pf_rpn  # noqa: F401
from mmdet.registry import DATASETS, MODELS
from mmdet.utils import register_all_modules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config',
        default='configs/m2-pf-rpn/m2_pf_rpn_sen12landslides_smoke.py',
        nargs='?')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()
    register_all_modules()
    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model)
    model.train()
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    batch = default_collate([dataset[args.index]])
    processed = model.data_preprocessor(batch, training=True)
    losses = model.loss(processed['inputs'], processed['data_samples'])
    print('sample:', dataset.get_data_info(args.index)['sample_id'])
    print('input_shape:', tuple(processed['inputs'].shape))
    for key, value in losses.items():
        print(key, float(value.detach().cpu()))
    if args.backward:
        total = sum(losses.values())
        total.backward()
        grad_sq = sum(
            float(p.grad.detach().pow(2).sum())
            for p in model.parameters() if p.grad is not None)
        print('grad_norm:', grad_sq ** 0.5)


if __name__ == '__main__':
    main()