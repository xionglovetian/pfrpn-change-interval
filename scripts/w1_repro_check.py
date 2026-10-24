#!/usr/bin/env python
"""W1 precondition: does the deployed change.npy really equal (first, last)?

Recomputes the change stack from the raw NetCDF patch with the long-baseline
rule (frames 0 and -1 after sortby(time)) and compares it byte-for-byte with the
deployed ``change.npy`` -- and the transposed ``mask.npy`` -- for a sample of
tiles.  If max|diff| is ~1e-7 the whole window-variant pipeline is comparable
with the published numbers; if not, the deployed stacks were built with another
rule and every arm built on top of it would be invalid.

ASCII only (piped to a bash process, so keep it plain).
"""
import argparse
import glob
import json
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from w1_cloud_scan import (build_index, change_stack, load_tile,  # noqa: E402
                           load_ids)


def pick_tiles(man, s2_index, lb_roots, per_event=12):
    """Sample tiles per event from the test split, restricted to positives."""
    with open(osp.join(man, 'test.json'), encoding='utf-8') as fh:
        rows = json.load(fh)
    by_event = {}
    for row in rows:
        if not row.get('labels', {}).get('annotations'):
            continue
        tid = row['image_id']
        if tid not in s2_index:
            continue
        found = None
        for root in lb_roots:
            cand = osp.join(root, 'events', row['event_id'], 'tiles', tid)
            if osp.exists(osp.join(cand, 'change.npy')):
                found = cand
                break
        if found is None:
            continue
        by_event.setdefault(row['event_id'], []).append((tid, found))
    picked = []
    for ev, items in by_event.items():
        picked.extend(items[:per_event])
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='/root/autodl-tmp/PF-RPN')
    ap.add_argument('--s2', default='/root/autodl-tmp/Sen12Landslides/data/'
                                    'data_harmonized/s2')
    ap.add_argument('--per-event', type=int, default=12)
    ap.add_argument('--out', default='/root/autodl-tmp/_w1/repro_check.json')
    args = ap.parse_args()

    data = osp.join(args.repo, 'data')
    lb_roots = [osp.join(data, 'landslide_multimodal_lb_pos'),
                osp.join(data, 'landslide_multimodal_lb_neg'),
                osp.join(data, 'landslide_multimodal'),
                osp.join(data, 'landslide_multimodal_full')]
    index = build_index(args.s2)
    picked = pick_tiles(osp.join(data, 'landslide_multimodal_hn_lb/manifests'),
                        index, lb_roots, args.per_event)
    print('tiles picked: %d' % len(picked), flush=True)

    rows = []
    for tid, tile_dir in picked:
        try:
            tile = load_tile(index[tid])
        except Exception as exc:                              # noqa: BLE001
            rows.append(dict(tile=tid, err='%s: %s' % (type(exc).__name__, exc)))
            continue
        n = tile['ndvi'].shape[0]
        mine = change_stack(tile, 0, n - 1)
        ref = np.load(osp.join(tile_dir, 'change.npy'))
        d = float(np.max(np.abs(ref - mine))) if ref.shape == mine.shape else -1.0
        # mask must match only after the spatial transpose
        mref = np.load(osp.join(tile_dir, 'mask.npy')) > 0
        mask_ok = bool(np.array_equal(mref, tile['mask']))
        # a single-pair diff for context (what a short window would look like)
        ppd = tile['attrs'].get('pre_post_dates', '')
        rows.append(dict(tile=tid, shape=list(ref.shape), max_abs_diff=d,
                         mask_match=mask_ok, ppd=str(ppd)[:40]))
        print('  %-28s maxdiff=%.6g mask_ok=%s' % (tid, d, mask_ok), flush=True)

    good = [r for r in rows if 'max_abs_diff' in r and r['max_abs_diff'] >= 0]
    if good:
        best = min(good, key=lambda r: r['max_abs_diff'])
        worst = max(good, key=lambda r: r['max_abs_diff'])
        print('\nSUMMARY n=%d  min=%.3g (%s)  max=%.3g (%s)  mask_ok=%d/%d'
              % (len(good), best['max_abs_diff'], best['tile'],
                 worst['max_abs_diff'], worst['tile'],
                 sum(1 for r in good if r['mask_match']), len(good)), flush=True)
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(rows, fh, indent=1)
    print('wrote %s' % args.out, flush=True)


if __name__ == '__main__':
    sys.exit(main())
