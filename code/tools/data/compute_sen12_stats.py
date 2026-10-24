# Copyright (c) OpenMMLab. All rights reserved.
"""Compute per-channel normalization statistics for converted M1 data.

Statistics are computed on the *training split only*.  Including val/test
would let the evaluation data influence the input normalization, which is a
(mild) form of information leakage and is avoidable at zero cost.

The manifest files are read explicitly by split name.  A previous version used
``manifests/*.json``, which also swept up auxiliary JSON files such as
``merge_report.json`` / ``split_report.json``; those are dicts, so iterating
them yields strings and the run crashes on ``item['modalities']``.
"""
import argparse
import json
from pathlib import Path

import numpy as np

KEYS = ('optical_pre', 'sar_pre', 'sar_post', 'topography', 'change')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'root', nargs='?',
        default='data/landslide_multimodal',
        help='Converted M1 dataset root.')
    parser.add_argument(
        '--splits', nargs='+', default=['train'],
        help='Splits used for the statistics. Default: train only.')
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    sums = {}
    sqs = {}
    count = 0
    used = {}
    for split in args.splits:
        path = root / 'manifests' / f'{split}.json'
        if not path.exists():
            raise FileNotFoundError(f'missing manifest: {path}')
        items = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(items, list):
            raise TypeError(f'{path} must contain a list of samples')
        for item in items:
            arrays = [
                np.load(root / item['modalities'][key]).astype(np.float64)
                for key in KEYS
            ]
            sample = np.concatenate(arrays, axis=-1)
            flat = sample.reshape(-1, sample.shape[-1])
            count += flat.shape[0]
            for i in range(flat.shape[1]):
                sums[i] = sums.get(i, 0.0) + flat[:, i].sum()
                sqs[i] = sqs.get(i, 0.0) + (flat[:, i] ** 2).sum()
        used[split] = len(items)

    mean = np.array([sums[i] / count for i in range(len(sums))])
    var = np.array([sqs[i] / count for i in range(len(sums))]) - mean ** 2
    std = np.sqrt(np.clip(var, 1e-12, None))
    output = dict(mean=mean.tolist(), std=std.tolist(), keys=list(KEYS),
                  splits=list(args.splits), tiles=used, pixels=int(count))
    (root / 'norm_stats.json').write_text(
        json.dumps(output, indent=2), encoding='utf-8')
    print('splits used =', args.splits, ' tiles =', used, ' pixels =', count)
    print('mean =', [round(float(x), 6) for x in mean])
    print('std  =', [round(float(x), 6) for x in std])


if __name__ == '__main__':
    main()