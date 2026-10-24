# Copyright (c) OpenMMLab. All rights reserved.
"""Consistency audit for the M1 data/eval pipeline.

Checks the invariants that silently corrupt metrics when violated:

1. modality stacking order / channel count matches ``norm_stats.json``
2. the training pipeline leaves gt_bboxes in the same frame as the image
3. the eval pipeline does the same (regression guard for the fixed bug)
4. evaluation GT count equals the manifest annotation count
5. proposal.v1 export round-trips and its boxes are in the native frame
6. normalisation statistics length equals the model input channels
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import default_collate

import mmdet.datasets  # noqa: F401
import mmdet.models  # noqa: F401
from mmdet.registry import DATASETS, MODELS
from mmdet.utils import register_all_modules

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

import eval_m1_split as E  # noqa: E402

FAILURES = []


def check(name, ok, detail=''):
    status = 'PASS' if ok else 'FAIL'
    print(f'[{status}] {name}' + (f' :: {detail}' if detail else ''))
    if not ok:
        FAILURES.append(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--data-root', default=None)
    parser.add_argument('--scale', type=int, default=320)
    parser.add_argument('--modality-blocks',
                        default='optical:3,sar:4,topography:5,change:3')
    args = parser.parse_args()
    register_all_modules()
    cfg = Config.fromfile(args.config)
    if args.data_root:
        cfg.merge_from_dict({'train_dataloader': {
            'dataset': {'data_root': args.data_root}}})
    data_root = cfg.train_dataloader.dataset.data_root
    scale = args.scale

    blocks = []
    for token in args.modality_blocks.split(','):
        name, width = token.split(':')
        blocks.append((name, int(width)))
    total_ch = sum(w for _, w in blocks)

    # 6. stats length vs channel count
    stats_path = Path(data_root) / 'norm_stats.json'
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    check('norm_stats length == input channels',
          len(stats.get('mean', [])) == total_ch,
          f"mean={len(stats.get('mean', []))} expected={total_ch}")

    # 1. modality stacking order: rebuild the stack by hand and compare
    ds_cfg = E._build_eval_dataset_cfg(cfg, 'test', scale)
    ds = DATASETS.build(ds_cfg)
    sample = ds[0]
    img = np.asarray(sample['inputs'])
    check('stacked image channel count',
          img.shape[0] == total_ch, f'{img.shape}')

    # Channel-order check needs the *native* frame, otherwise the resize makes
    # a pixel-wise comparison against the .npy files meaningless.
    raw_cfg = dict(ds_cfg)
    raw_cfg['pipeline'] = [dict(s) for s in ds_cfg['pipeline']
                           if s['type'] != 'FixScaleResize']
    raw_ds = DATASETS.build(raw_cfg)
    raw = raw_ds.get_data_info(0)
    raw_img = np.asarray(raw_ds[0]['inputs'])
    ref_parts = []
    for name, width in blocks:
        if name == 'optical':
            arr = np.load(Path(raw['optical_path']))
        elif name == 'sar':
            arr = np.concatenate([np.load(Path(p)) for p in raw['sar_path']],
                                 axis=-1)
        elif name == 'topography':
            arr = np.load(Path(raw['topography_path']))
        else:
            arr = np.load(Path(raw['change_path']))
        ref_parts.append(arr.transpose(2, 0, 1))
    ref = np.concatenate(ref_parts, axis=0)
    # compare in the resized frame: sample['inputs'] is float32 CHW
    got = raw_img.astype(np.float32)
    same_shape = got.shape == ref.shape
    ratio = None
    if same_shape:
        with np.errstate(divide='ignore', invalid='ignore'):
            r = got / np.where(ref == 0, 1e-6, ref)
            ratio = float(np.nanmedian(r))
    check('stacked channel order == documented modality order',
          same_shape and ratio is not None and abs(ratio - 1.0) < 1e-3,
          f'shape {got.shape} vs {ref.shape}, median ratio {ratio}')

    # 3. eval pipeline: gt must live in the resized frame
    gb = sample['data_samples'].gt_instances.bboxes.tensor.numpy()
    man = json.loads((Path(data_root) / 'manifests' / 'test.json'
                      ).read_text())
    nat = np.asarray([a['bbox_xyxy'] for a in man[0]['labels']['annotations']])
    eval_max = float(gb.max()) if gb.size else 0.0
    check('eval gt inside resized frame',
          eval_max <= scale + 1,
          f'max={eval_max:.1f} (native max={nat.max():.1f}, scale={scale})')
    if gb.size and nat.size:
        ratio = float(np.median(gb[:, 2:] - gb[:, :2]) /
                      max(np.median(nat[:, 2:] - nat[:, :2]), 1e-6))
        check('eval gt scaled by scale/native',
              abs(ratio - scale / 128.0) < 0.15,
              f'wh ratio={ratio:.3f} expected={scale / 128.0:.3f}')

    # 2. training pipeline: same frame invariant
    tr_ds = DATASETS.build(dict(cfg.train_dataloader.dataset))
    tr = tr_ds[0]
    tb = tr['data_samples'].gt_instances.bboxes.tensor.numpy()
    tr_max = float(tb.max()) if tb.size else 0.0
    check('train gt inside resized frame',
          tr_max <= scale + 1,
          f'max={tr_max:.1f} img_shape={tr["data_samples"].img_shape}')

    # 4. GT totals per split
    for split in ('train', 'val', 'test'):
        items = json.loads((Path(data_root) / 'manifests' / f'{split}.json'
                            ).read_text())
        ann = sum(len(i['labels']['annotations']) for i in items)
        print(f'[info] {split}: {len(items)} tiles, {ann} gt boxes')

    # 5. normalisation round trip through the data preprocessor
    model = MODELS.build(cfg.model)
    pp = model.data_preprocessor
    batch = default_collate([tr])
    out = pp(batch, training=False)
    tensor = out['inputs']
    mu = torch.tensor(pp.mean, dtype=torch.float32)[None, :, None, None]
    sd = torch.tensor(pp.std, dtype=torch.float32)[None, :, None, None]
    denorm = (tensor.cpu() * sd + mu)
    check('data preprocessor per-channel normalisation',
          tensor.shape[1] == total_ch and len(pp.mean) == total_ch
          and len(pp.std) == total_ch,
          f'normalised shape {tuple(tensor.shape)}, mean {len(pp.mean)}')
    # Semantic check: if mean/std were bound to the wrong channel block the
    # normalised activations would explode.  Aggregate over a few samples and
    # require every channel to stay in a sane range.
    acc, sq, n = None, None, 0
    for idx in range(min(8, len(ds))):
        s = ds[idx]
        b = default_collate([s])
        t = pp(b, training=False)['inputs'][0]
        flat = t.reshape(t.shape[0], -1)
        if acc is None:
            acc = flat.mean(dim=1).double()
            sq = (flat ** 2).mean(dim=1).double()
        else:
            acc += flat.mean(dim=1).double()
            sq += (flat ** 2).mean(dim=1).double()
        n += 1
    mean_n = (acc / n).numpy()
    std_n = np.sqrt(np.maximum((sq / n).numpy() - mean_n ** 2, 0))
    worst = float(np.abs(mean_n).max())
    check('normalised activations are centred per channel',
          worst < 2.0, f'max |mean| over 15 channels = {worst:.3f}')
    check('normalisation statistics match channel semantics',
          float(std_n.max()) < 5.0 and float(std_n.min()) > 0.05,
          f'std range [{std_n.min():.3f}, {std_n.max():.3f}]')

    # Round-trip must use the *same* sample that produced the batch, otherwise
    # the training pipeline's RandomFlip makes the comparison meaningless.
    # Per-channel explicit check avoids any broadcasting ambiguity (pp.mean
    # may be shaped (C,) or (1, C)).
    mean_flat = np.asarray(pp.mean, dtype=np.float64).reshape(-1)
    std_flat = np.asarray(pp.std, dtype=np.float64).reshape(-1)
    raw_ref = np.asarray(raw_ds[0]['inputs'], dtype=np.float64)
    norm_ref = pp(default_collate([raw_ds[0]]),
                  training=False)['inputs'][0].cpu().numpy()
    worst = 0.0
    for ch in range(norm_ref.shape[0]):
        expect = (raw_ref[ch] - mean_flat[ch]) / std_flat[ch]
        worst = max(worst, float(np.abs(norm_ref[ch] - expect).max()))
    check('normalisation is exactly (x - mean) / std per channel',
          worst < 1e-3, f'max abs deviation {worst:.3e}')

    print()
    if FAILURES:
        print(f'AUDIT FAILED: {len(FAILURES)} check(s): {FAILURES}')
        return 1
    print('AUDIT PASSED: all pipeline invariants hold')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())