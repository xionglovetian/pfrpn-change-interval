# Copyright (c) OpenMMLab. All rights reserved.
"""Run one real-data forward pass through M2-PF-RPN."""
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
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    register_all_modules()
    cfg = Config.fromfile(args.config)
    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    model = MODELS.build(cfg.model).to(device)
    model.eval()

    dataloader_cfg = cfg.get('test_dataloader', None) or cfg.train_dataloader
    dataset = DATASETS.build(dataloader_cfg.dataset)
    batch = default_collate([dataset[args.index]])
    with torch.no_grad():
        processed = model.data_preprocessor(batch, training=False)
        inputs = processed['inputs'].to(device)
        results = model.predict(
            inputs, processed['data_samples'], rescale=True)
    pred = results[0].pred_instances
    print('sample:', dataset.get_data_info(args.index)['sample_id'])
    print('device:', device)
    print('input_shape:', tuple(inputs.shape))
    print('predictions:', len(pred))
    if len(pred) > 0:
        print('bboxes_shape:', tuple(pred.bboxes.shape))
        print('scores_shape:', tuple(pred.scores.shape))


if __name__ == '__main__':
    main()