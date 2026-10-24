#!/usr/bin/env python
"""LOEO fold root for one held-out event: long-deploy and plus1-train variants.

Builds two roots sharing the same test manifest (the held-out event):

    data_loeo_<ev>_long     change -> deployed long stack   (deploy/eval root)
    data_loeo_<ev>_plus1    change -> +/-1 calendar stack    (training root)

The fold's baseline comes from the archived LOEO checkpoint that already exists
on this machine, evaluated on the long root, so no extra training is needed for
the comparison.
"""
import json
import os
import os.path as osp
import sys

import numpy as np

DATA = '/root/autodl-tmp/PF-RPN/data'
MAN = osp.join(DATA, 'landslide_multimodal_hn_lb/manifests')
W1 = '/root/autodl-tmp/_w1'
ORIG = osp.join(DATA, 'landslide_multimodal_hn_lb/norm_stats.json')
CFG_SRC = '/root/autodl-tmp/PF-RPN/configs/m2-pf-rpn/m2_pf_rpn_cloud_full.py'
P1_DIRS = (osp.join(W1, 'variants_plus1'), osp.join(W1, 'variants_plus1_test'))


def resolve(rel):
    i = rel.find('landslide_multimodal_lb_')
    return osp.join(DATA, rel[i:].replace('\\', '/'))


def plus1(tile):
    for d in P1_DIRS:
        p = osp.join(d, tile + '.npy')
        if osp.exists(p):
            return p
    return None


def build(event, out_root, use_plus1):
    os.makedirs(osp.join(out_root, 'manifests'), exist_ok=True)
    orig = json.load(open(ORIG))
    sums = [np.zeros(3), np.zeros(3), 0]
    stats = {}
    for split, src in (('train', osp.join(MAN, 'loeo', event + '_train.json')),
                       ('test', osp.join(MAN, 'event_' + event + '.json'))):
        rows = json.load(open(src, encoding='utf-8'))
        out = []
        nv = nd = 0
        for r in rows:
            r = json.loads(json.dumps(r))
            mods = r['modalities']
            tile = r.get('image_id') or r.get('sample_id')
            dep = resolve(mods['change'])
            chosen = dep
            if use_plus1:
                p = plus1(tile)
                if p:
                    chosen, nv = p, nv + 1
                else:
                    nd += 1
            else:
                nd += 1
            for k, rel in list(mods.items()):
                if k != 'change':
                    mods[k] = resolve(rel)
            mods['change'] = chosen
            if split == 'train' and use_plus1 and chosen != dep:
                a = np.load(chosen).reshape(-1, 3).astype(np.float64)
                sums[0] += a.sum(0)
                sums[1] += (a ** 2).sum(0)
                sums[2] += a.shape[0]
            out.append(r)
        json.dump(out, open(osp.join(out_root, 'manifests', split + '.json'), 'w'),
                  ensure_ascii=False)
        stats[split] = dict(rows=len(out), variant=nv, deployed=nd)
    mean, std = list(orig['mean']), list(orig['std'])
    if use_plus1 and sums[2]:
        mean[12:15] = (sums[0] / sums[2]).tolist()
        std[12:15] = np.sqrt(np.maximum(
            sums[1] / sums[2] - (sums[0] / sums[2]) ** 2, 0)).tolist()
    json.dump(dict(mean=mean, std=std, keys=orig.get('keys'),
                   note='loeo %s %s' % (event, 'plus1' if use_plus1 else 'long')),
              open(osp.join(out_root, 'norm_stats.json'), 'w'), indent=2)
    cfg = open(CFG_SRC, encoding='utf-8').read()
    for old in ("data_root = 'data/landslide_multimodal_hn_lb/'",
                "data_root = 'data/landslide_multimodal_hn_lb'"):
        cfg = cfg.replace(old, "data_root = r'%s'" % out_root)
    base = osp.join(osp.dirname(osp.abspath(CFG_SRC)), 'm2_pf_rpn_multimodal.py')
    cfg = cfg.replace("_base_ = ['./m2_pf_rpn_multimodal.py']",
                      "_base_ = [r'%s']" % base)
    open(osp.join(out_root, 'config.py'), 'w', encoding='utf-8').write(cfg)
    print('%s -> %s  %s' % (event, out_root, stats), flush=True)


if __name__ == '__main__':
    event = sys.argv[1] if len(sys.argv) > 1 else 'indonesia'
    build(event, '/root/autodl-tmp/data_loeo_%s_long' % event, False)
    build(event, '/root/autodl-tmp/data_loeo_%s_plus1' % event, True)
