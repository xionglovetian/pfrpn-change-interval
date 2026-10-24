# Copyright (c) OpenMMLab. All rights reserved.
"""Validate event-level separation of M1 split manifests."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _split_entities(items):
    events = defaultdict(set)
    images = defaultdict(set)
    for item in items:
        event_id = str(item.get('event_id', 'unknown'))
        image_id = str(item.get('image_id', item.get('sample_id', '')))
        events[event_id].add(image_id)
        images[image_id].add(event_id)
    return events, images


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest_dir')
    parser.add_argument('--output', default=None)
    parser.add_argument('--splits', nargs='+',
                        default=['train', 'val', 'test'])
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.manifest_dir)
    split_items = {
        split: _load(root / f'{split}.json')
        for split in args.splits
    }

    event_sets = {}
    image_sets = {}
    for split, items in split_items.items():
        events, images = _split_entities(items)
        event_sets[split] = set(events)
        image_sets[split] = set(images)

    event_leaks = {}
    image_leaks = {}
    for i, left in enumerate(args.splits):
        for right in args.splits[i + 1:]:
            event_overlap = sorted(event_sets[left] & event_sets[right])
            image_overlap = sorted(image_sets[left] & image_sets[right])
            if event_overlap:
                event_leaks[f'{left}-{right}'] = event_overlap
            if image_overlap:
                image_leaks[f'{left}-{right}'] = image_overlap

    report = {
        'manifest_dir': str(root),
        'splits': {
            split: {
                'num_samples': len(items),
                'num_events': len(event_sets[split]),
                'events': sorted(event_sets[split]),
            }
            for split, items in split_items.items()
        },
        'event_leaks': event_leaks,
        'image_leaks': image_leaks,
        'event_level_passed': not event_leaks,
        'image_level_passed': not image_leaks,
        'note': 'A split can only be used for paper results when '
                'event_level_passed is true.',
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        Path(args.output).write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding='utf-8')


if __name__ == '__main__':
    main()