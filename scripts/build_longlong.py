#!/usr/bin/env python
"""Channel-count control: 6-channel change built as [long | long].

Same width as the fusion arm, zero new information.  If this arm matches the
3-channel anchor, the fusion gain is genuine interval complementarity; if it
climbs on its own, the gain is input width rather than the interval.
"""
import glob
import json
import os
import os.path as osp
from multiprocessing import Pool

import numpy as np

DATA = '/root/autodl-tmp/PF-RPN/data'
MAN = osp.join(DATA, 'landslide_multimodal_hn_lb/manifests')
W1 = '/root/autodl-tmp/_w1'
OUT = osp.join(W1, 'variants_longlong')
ROOT = '/root/autodl-tmp/data_longlong'
ORIG = osp.join(DATA, 'landslide_multimodal_hn_lb/norm_stats.json')
CFG_SRC = '/root/autodl-tmp/PF-RPN/configs/m2-pf-rpn/m2_pf_rpn_cloud_full.py'


def resolve(rel):
    i = rel.find('landslide_multimodal_lb_')
    return osp.join(DATA, rel[i:].replace('\\', '/'))


def deployed(event, tile):
    for kind in ('landslide_multimodal_lb_pos', 'landslide_multimodal_lb_neg'):
        p = osp.join(DATA, kind, 'events', event, 'tiles', tile, 'change.npy')
        if osp.exists(p):
            return p
    return None


def one(task):
    event, tile = task
    try:
        p = deployed(event, tile)
        if p is None:
            return (tile, 0, 'no deployed change')
        a = np.load(p).astype(np.float32)
        np.save(osp.join(OUT, tile + '.npy'),
                np.concatenate([a, a], -1))
        return (tile, 1, None)
    except Exception as exc:                                   # noqa: BLE001
        return (tile, 0, '%s: %s' % (type(exc).__name__, exc))


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = {sp: json.load(open(osp.join(MAN, sp + '.json')))
            for sp in ('train', 'val', 'test')}
    seen, tasks = set(), []
    for sp in ('train', 'val', 'test'):
        for r in rows[sp]:
            t = r['image_id']
            if t in seen:
                continue
            seen.add(t)
            tasks.append((r['event_id'], t))
    print('tiles:', len(tasks), flush=True)
    with Pool(24) as pool:
        res = pool.map(one, tasks, chunksize=16)
    bad = [r for r in res if not r[1]]
    print('longlong written:', sum(r[1] for r in res), 'failed:', len(bad),
          bad[:2], flush=True)

    os.makedirs(osp.join(ROOT, 'manifests'), exist_ok=True)
    sums = [np.zeros(6), np.zeros(6), 0]
    for sp in ('train', 'val', 'test'):
        out_rows = []
        for r in rows[sp]:
            r = json.loads(json.dumps(r))
            mods = r['modalities']
            p6 = osp.join(OUT, r['image_id'] + '.npy')
            for k, rel in list(mods.items()):
                if k != 'change':
                    mods[k] = resolve(rel)
            mods['change'] = p6 if osp.exists(p6) else resolve(mods['change'])
            if sp == 'train' and osp.exists(p6):
                a = np.load(p6).reshape(-1, 6).astype(np.float64)
                sums[0] += a.sum(0)
                sums[1] += (a ** 2).sum(0)
                sums[2] += a.shape[0]
            out_rows.append(r)
        json.dump(out_rows, open(osp.join(ROOT, 'manifests', sp + '.json'), 'w'),
                  ensure_ascii=False)
        print('%-6s %d rows' % (sp, len(out_rows)), flush=True)

    orig = json.load(open(ORIG))
    m6 = (sums[0] / sums[2]).tolist()
    s6 = np.sqrt(np.maximum(sums[1] / sums[2] - (sums[0] / sums[2]) ** 2, 0)).tolist()
    mean = list(orig['mean'][:12]) + m6
    std = list(orig['std'][:12]) + s6
    json.dump(dict(mean=mean, std=std, keys=orig.get('keys'),
                   note='6ch control [long|long]'),
              open(osp.join(ROOT, 'norm_stats.json'), 'w'), indent=2)
    print('change6 stats mean', [round(v, 4) for v in m6], flush=True)

    cfg = open(CFG_SRC, encoding='utf-8').read()
    for old in ("data_root = 'data/landslide_multimodal_hn_lb/'",
                "data_root = 'data/landslide_multimodal_hn_lb'"):
        cfg = cfg.replace(old, "data_root = r'%s'" % ROOT)
    base = osp.join(osp.dirname(osp.abspath(CFG_SRC)), 'm2_pf_rpn_multimodal.py')
    cfg = cfg.replace("_base_ = ['./m2_pf_rpn_multimodal.py']",
                      "_base_ = [r'%s']" % base)
    cfg += ("\n\n_modality_channels = dict(optical=3, sar=4, topography=5, change=6)\n"
            "modality_channels = _modality_channels\n"
            "model['modality_channels'] = _modality_channels\n"
            "model['data_preprocessor']['mean'] = %r\n"
            "model['data_preprocessor']['std'] = %r\n"
            "train_dataloader['dataset']['pipeline'][0]['modality_channels'] = "
            "_modality_channels\n" % (mean, std))
    open(osp.join(ROOT, 'config.py'), 'w', encoding='utf-8').write(cfg)
    print('wrote', osp.join(ROOT, 'config.py'), flush=True)


if __name__ == '__main__':
    main()
