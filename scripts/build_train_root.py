#!/usr/bin/env python
"""Build the TRAINING data root for the calendar-window arm (Linux/Windows).

Runs on the GPU box after migration.  Produces, for one arm:

    <out_root>/manifests/{train,val,test}.json   POSIX absolute paths
    <out_root>/norm_stats.json                   15-dim, change from the arm
    <out_root>/config.py                         cloud config with data_root set

Sources
-------
train/val   ``variants_calendar/``  (8,193 tiles: published window where present,
                                    mid-series surrogate where the tile has no
                                    event date)
test        ``variants_bracket_test/`` + ``variants/win_pair/`` + deployed npy
            (673 positives event-anchored, 1,086 negatives deployed long window)

Usage
-----
python build_train_root.py \
    --data /root/autodl-tmp/PF-RPN/data \
    --man   /root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb/manifests \
    --calendar /root/autodl-tmp/_mig/variants_calendar \
    --test-arrays /root/autodl-tmp/_mig/variants/win_pair,/root/autodl-tmp/_mig/variants_bracket_test \
    --config /root/autodl-tmp/PF-RPN/configs/m2-pf-rpn/m2_pf_rpn_cloud_full.py \
    --out /root/autodl-tmp/data_calendar
"""
import argparse
import json
import os
import os.path as osp

import numpy as np

CHANGE_SLICE = slice(12, 15)


def resolve_deployed(rel, data_root):
    """'../landslide_multimodal_lb_pos/events/...' -> absolute under data_root."""
    i = rel.find('landslide_multimodal_lb_')
    if i < 0:
        raise ValueError('unexpected manifest path: %s' % rel)
    return osp.join(data_root, rel[i:].replace('\\', '/'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='.../PF-RPN/data')
    ap.add_argument('--man', required=True, help='.../manifests dir (source)')
    ap.add_argument('--calendar', required=True, help='train/val change arrays')
    ap.add_argument('--test-arrays', default='',
                    help='comma-separated dirs for test positive arrays')
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--norm-source', default=None,
                    help='defaults to <data>/landslide_multimodal_hn_lb/'
                         'norm_stats.json')
    args = ap.parse_args()

    norm_src = args.norm_source or osp.join(
        args.data, 'landslide_multimodal_hn_lb', 'norm_stats.json')
    orig = json.load(open(norm_src, encoding='utf-8'))
    test_dirs = [d for d in args.test_arrays.split(',') if d.strip()]

    os.makedirs(osp.join(args.out, 'manifests'), exist_ok=True)
    sums = None
    report = {}

    for split in ('train', 'val', 'test'):
        src = osp.join(args.man, split + '.json')
        rows = json.load(open(src, encoding='utf-8'))
        out_rows, n_var, n_dep = [], 0, 0
        for row in rows:
            row = json.loads(json.dumps(row))
            mods = row['modalities']
            tile = row.get('image_id') or row.get('sample_id')
            deployed = resolve_deployed(mods.get('change', ''), args.data) \
                if mods.get('change') else None

            chosen, is_variant = deployed, False
            if split in ('train', 'val'):
                cand = osp.join(args.calendar, tile + '.npy')
                if osp.exists(cand):
                    chosen, is_variant = cand, True
            else:
                for d in test_dirs:
                    cand = osp.join(d, tile + '.npy')
                    if osp.exists(cand):
                        chosen, is_variant = cand, True
                        break
            if chosen is None:
                raise RuntimeError('no change array for %s (%s)' % (tile, split))
            if is_variant:
                n_var += 1
            else:
                n_dep += 1

            for key, rel in list(mods.items()):
                if key != 'change':
                    mods[key] = resolve_deployed(rel, args.data)
            mods['change'] = chosen

            # accumulate change-channel stats on the TRAIN split only
            if split == 'train' and is_variant:
                arr = np.load(chosen).reshape(-1, 3).astype(np.float64)
                s, q = arr.sum(0), (arr ** 2).sum(0)
                if sums is None:
                    sums = [s, q, arr.shape[0]]
                else:
                    sums[0] += s
                    sums[1] += q
                    sums[2] += arr.shape[0]
            out_rows.append(row)

        with open(osp.join(args.out, 'manifests', split + '.json'), 'w',
                  encoding='utf-8') as fh:
            json.dump(out_rows, fh, ensure_ascii=False)
        report[split] = dict(rows=len(out_rows), variant=n_var, deployed=n_dep)
        print('%-6s rows=%5d variant=%5d deployed=%5d' %
              (split, len(out_rows), n_var, n_dep), flush=True)

    mean, std = list(orig['mean']), list(orig['std'])
    if sums is not None:
        s, q, cnt = sums
        mean[CHANGE_SLICE] = [float(x / cnt) for x in s]
        std[CHANGE_SLICE] = [float(np.sqrt(max(y / cnt - (x / cnt) ** 2, 0.0)))
                             for x, y in zip(s, q)]
        print('change-channel stats from %d train pixels: mean=%s std=%s'
              % (cnt, [round(v, 4) for v in mean[12:]],
                 [round(v, 4) for v in std[12:]]), flush=True)
    json.dump(dict(mean=mean, std=std, keys=orig.get('keys'),
                   note='calendar arm'), open(osp.join(args.out,
                                                       'norm_stats.json'), 'w'),
              indent=2)

    cfg = open(args.config, encoding='utf-8').read()
    for old in ("data_root = 'data/landslide_multimodal_hn_lb/'",
                "data_root = 'data/landslide_multimodal_hn_lb'"):
        cfg = cfg.replace(old, "data_root = r'%s'" % args.out)
    base = osp.join(osp.dirname(osp.abspath(args.config)),
                    'm2_pf_rpn_multimodal.py')
    cfg = cfg.replace("_base_ = ['./m2_pf_rpn_multimodal.py']",
                      "_base_ = [r'%s']" % base)
    open(osp.join(args.out, 'config.py'), 'w', encoding='utf-8').write(cfg)
    print('wrote %s/config.py (data_root=%s)' % (args.out, args.out))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
