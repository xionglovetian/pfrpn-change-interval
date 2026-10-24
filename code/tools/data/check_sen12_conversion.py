# Copyright (c) OpenMMLab. All rights reserved.
"""Check a converted Sen12Landslides sample and its M1 channel layout."""
import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'root', nargs='?',
        default='E:/PF-RPN/data/landslide_multimodal_smoke')
    parser.add_argument('--split', default='train')
    parser.add_argument('--index', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    items = json.loads(
        (root / 'manifests' / f'{args.split}.json').read_text(
            encoding='utf-8'))
    item = items[args.index]
    arrays = [
        np.load(root / item['modalities'][key])
        for key in ('optical_pre', 'sar_pre', 'sar_post', 'topography',
                    'change')
    ]
    sample = np.concatenate(arrays, axis=-1)
    assert sample.shape[-1] == 15, sample.shape
    print('sample_id:', item['sample_id'])
    print('channels:', sample.shape[-1])
    print('shape:', sample.shape)
    print('bboxes:', len(item['labels']['annotations']))
    print('modality_mask:', item['modality_mask'])


if __name__ == '__main__':
    main()