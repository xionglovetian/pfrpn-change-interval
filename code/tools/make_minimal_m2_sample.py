# Copyright (c) OpenMMLab. All rights reserved.
"""Generate a deterministic minimal M2-PF-RPN data sample."""
import argparse
import json
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', help='Output sample directory.')
    parser.add_argument('--size', type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(42)
    height = width = args.size

    arrays = {
        'optical_pre.npy': rng.random((height, width, 3), dtype=np.float32),
        'sar_pre.npy': rng.random((height, width, 2), dtype=np.float32),
        'sar_post.npy': rng.random((height, width, 2), dtype=np.float32),
        'topography.npy': rng.random((height, width, 5), dtype=np.float32),
        'change.npy': rng.random((height, width, 3), dtype=np.float32),
    }
    for name, array in arrays.items():
        np.save(os.path.join(out_dir, name), array)

    manifest = [{
        'sample_id': 'event_001_tile_000123',
        'event_id': 'event_001',
        'image_id': 'tile_000123',
        'shape': [height, width],
        'modality_mask': [1, 1, 1, 1],
        'modalities': {
            'optical_pre': 'optical_pre.npy',
            'sar_pre': 'sar_pre.npy',
            'sar_post': 'sar_post.npy',
            'topography': 'topography.npy',
            'change': 'change.npy',
        },
        'labels': {
            'annotations': [{
                'bbox_xyxy': [10.0, 10.0, 32.0, 32.0],
                'category_id': 1,
            }],
            'negatives': [{
                'type': 'bare_soil',
                'bbox_xyxy': [34.0, 34.0, 55.0, 55.0],
            }],
        },
    }]
    with open(os.path.join(out_dir, 'manifest.json'), 'w',
              encoding='utf-8') as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
    print(f'wrote minimal M2-PF-RPN sample to {out_dir}')


if __name__ == '__main__':
    main()