# Copyright (c) OpenMMLab. All rights reserved.
"""Create a provisional event-level split for M1 experiments.

M0 owns the final split.  This tool exists so that multi-event data can be
organised without leakage, and offers two strategies:

* ``balance`` (default): greedy assignment of whole events that also matches
  the requested tile-count ratios, so val/test stay large enough to be
  meaningful.
* ``random``: shuffle whole events (the original behaviour).

Either way an event (inventory) is never split across subsets.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

NAMES = ['train', 'val', 'test']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_manifest')
    parser.add_argument('out_dir')
    parser.add_argument('--ratios', type=float, nargs=3,
                        default=[0.7, 0.15, 0.15])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--strategy', choices=['random', 'balance'], default='balance',
        help='balance matches tile-count ratios; random shuffles events.')
    return parser.parse_args()


def balanced_split(grouped, ratios):
    """Greedy event-level split that also matches tile-count ratios."""
    sizes = {event: len(items) for event, items in grouped.items()}
    total = sum(sizes.values())
    targets = {name: total * ratio for name, ratio in zip(NAMES, ratios)}
    current = {name: 0 for name in NAMES}
    split_events = {name: [] for name in NAMES}
    order = sorted(sizes, key=lambda event: (-sizes[event], event))
    for event in order:
        chosen = max(
            NAMES,
            key=lambda name: (targets[name] - current[name],
                              -len(split_events[name])))
        split_events[chosen].append(event)
        current[chosen] += sizes[event]
    # Guarantee non-empty val/test when enough events exist.
    for name in NAMES[1:]:
        if not split_events[name]:
            donor = max(NAMES, key=lambda n: len(split_events[n]))
            if len(split_events[donor]) > 1:
                split_events[name].append(split_events[donor].pop())
    return split_events


def random_split(grouped, ratios, seed):
    events = list(grouped)
    rng = random.Random(seed)
    rng.shuffle(events)
    n_events = len(events)
    n_train = max(1, int(round(n_events * ratios[0]))) if n_events else 0
    n_val = max(1, int(round(n_events * ratios[1]))) if n_events > 2 else 0
    n_test = max(0, n_events - n_train - n_val)
    if n_events >= 3 and n_test == 0:
        n_test = 1
        n_train = max(1, n_train - 1)
    split_events = {
        'train': events[:n_train],
        'val': events[n_train:n_train + n_val],
        'test': events[n_train + n_val:n_train + n_val + n_test],
    }
    assigned = {e for values in split_events.values() for e in values}
    split_events['train'].extend([e for e in events if e not in assigned])
    return split_events


def main():
    args = parse_args()
    data = json.loads(Path(args.input_manifest).read_text(encoding='utf-8'))
    grouped = defaultdict(list)
    for item in data:
        grouped[str(item.get('event_id', 'unknown'))].append(item)

    assert abs(sum(args.ratios) - 1.0) < 1e-6, 'ratios must sum to 1'
    if args.strategy == 'balance':
        split_events = balanced_split(grouped, args.ratios)
    else:
        split_events = random_split(grouped, args.ratios, args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, event_ids in split_events.items():
        items = [item for event_id in event_ids for item in grouped[event_id]]
        (out_dir / f'{split}.json').write_text(
            json.dumps(items, indent=2, ensure_ascii=False), encoding='utf-8')
        counts[split] = dict(events=event_ids, num_events=len(event_ids),
                             num_tiles=len(items))

    (out_dir / 'split_manifest.json').write_text(
        json.dumps(
            dict(seed=args.seed, ratios=args.ratios, strategy=args.strategy,
                 splits=counts,
                 note='Provisional event-level split; M0 owns final split.'),
            indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(counts, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()