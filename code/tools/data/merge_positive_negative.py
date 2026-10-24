# Copyright (c) OpenMMLab. All rights reserved.
"""Merge LD positives with S12LS-AD hard negatives into one M1 manifest set.

Positives keep the event-level split of ``positive_root``.  Negatives come from
the S12LS-AD conversion (``--skip-annotated``) and are re-assigned to whichever
split already contains their inventory, so an event can never straddle two
subsets.  Modality paths are rewritten with a ``../`` prefix because the mixed
set lives beside the two source roots.
"""
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

SPLITS = ('train', 'val', 'test')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--positive-root', required=True)
    parser.add_argument('--negative-root', required=True)
    parser.add_argument('--out-root', required=True)
    parser.add_argument('--positive-prefix', default='../landslide_multimodal_full')
    parser.add_argument('--negative-prefix', default='../landslide_multimodal_adneg')
    parser.add_argument('--drop-inventories', nargs='*', default=[])
    return parser.parse_args()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _rewrite(entry, prefix):
    out = json.loads(json.dumps(entry))
    for key, value in list(out.get('modalities', {}).items()):
        if isinstance(value, str) and not value.startswith('../'):
            out['modalities'][key] = f'{prefix}/{value}'
    return out


def main():
    args = parse_args()
    pos = Path(args.positive_root)
    neg = Path(args.negative_root)
    out = Path(args.out_root)
    (out / 'manifests').mkdir(parents=True, exist_ok=True)

    event_to_split = {}
    merged = {s: [] for s in SPLITS}
    n_pos = {}
    for split in SPLITS:
        items = _read(pos / 'manifests' / f'{split}.json')
        n_pos[split] = len(items)
        for item in items:
            event_to_split[str(item.get('event_id'))] = split
            merged[split].append(_rewrite(item, args.positive_prefix))

    # negatives: concatenate the AD-side split files, then re-split by event
    raw_neg = []
    for split in SPLITS:
        path = neg / 'manifests' / f'{split}.json'
        if path.exists():
            raw_neg.extend(_read(path))

    dropped = defaultdict(int)
    unassigned = defaultdict(int)
    n_neg = defaultdict(int)
    for item in raw_neg:
        event = str(item.get('event_id'))
        if event in args.drop_inventories:
            dropped[event] += 1
            continue
        if item['labels']['annotations']:
            continue  # only unannotated patches are negatives
        split = event_to_split.get(event)
        if split is None:
            unassigned[event] += 1
            continue
        merged[split].append(_rewrite(item, args.negative_prefix))
        n_neg[split] += 1

    for split in SPLITS:
        (out / 'manifests' / f'{split}.json').write_text(
            json.dumps(merged[split], indent=2, ensure_ascii=False),
            encoding='utf-8')

    stats = pos / 'norm_stats.json'
    if stats.exists():
        shutil.copyfile(stats, out / 'norm_stats.json')

    report = dict(
        positive=n_pos, negative=dict(n_neg),
        total={s: n_pos[s] + n_neg[s] for s in SPLITS},
        dropped_inventories=dict(dropped),
        unassigned_inventories=dict(unassigned),
        event_to_split=event_to_split,
        note='Negatives are S12LS-AD unannotated patches re-split by event so '
             'that they never cross an event boundary.')
    (out / 'manifests' / 'merge_report.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()