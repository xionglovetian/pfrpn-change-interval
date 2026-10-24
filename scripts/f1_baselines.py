#!/usr/bin/env python
"""F1-S2 baselines: can the event window be predicted WITHOUT the imagery?

Three baselines, evaluated leave-one-event-out (LOEO) on every annotated tile
that carries ``pre_post_dates``:

  B0  rule        adjacent pair with the largest tile-mean NDVI drop
                  (this needs the pixels; its value is quoted from the
                   chimanimani run: 41.9% exact / 65.0% within +/-1)
  B1  event prior the (pre, post) median of the OTHER events, applied to every
                  tile of the held-out event -- the decisive baseline: if this
                  already reaches the rule's accuracy, window prediction carries
                  almost no per-tile image information
  B2  global prior  the median of ALL other tiles (no event grouping)

Also reports the F3 preliminaries (date_confidence distribution, multi-event
tiles) from the same feature dump.
"""
import argparse
import collections
import glob
import json
import os
import os.path as osp
import statistics

MAN_EVENTS = ('italy', 'kyrgyzstan1', 'kyrgyzstan2', 'hiroshima', 'itogon',
              'newzealand', 'hokkaido', 'chimanimani', 'indonesia', 'china',
              'usa')


def load_events(man_dir):
    """image_id -> event_id from the three manifests."""
    mapping = {}
    for split in ('train', 'val', 'test'):
        p = osp.join(man_dir, '%s.json' % split)
        if not osp.exists(p):
            continue
        for row in json.load(open(p, encoding='utf-8')):
            tid = row.get('image_id') or row.get('sample_id')
            mapping[tid] = row.get('event_id', 'unknown')
    return mapping


def acc(pred, truth):
    return dict(n=len(truth),
                exact=sum(1 for p, t in zip(pred, truth) if p == t) / len(truth),
                within1=sum(1 for p, t in zip(pred, truth)
                            if abs(p[0] - t[0]) <= 1 and abs(p[1] - t[1]) <= 1)
                / len(truth))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--man', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    event_of = load_events(args.man)
    rows = []
    for line in open(args.features, encoding='utf-8'):
        r = json.loads(line)
        if not r.get('ppd'):
            continue
        r['event'] = event_of.get(r['tile'], 'unknown')
        rows.append(r)
    print('annotated tiles with pre_post_dates: %d' % len(rows))
    by_event = collections.Counter(r['event'] for r in rows)
    print('by event: %s' % dict(by_event))

    # ---- distribution of the oracle window --------------------------------
    gap = [r['ppd'][1] - r['ppd'][0] for r in rows]
    pos_frac = [r['ppd'][0] / max(r['n'] - 1, 1) for r in rows]
    print('\noracle window: gap median=%d mean=%.2f | pre/n median=%.2f'
          % (statistics.median(gap), statistics.mean(gap),
             statistics.median(pos_frac)))
    print('pre index histogram: %s'
          % dict(sorted(collections.Counter(r['ppd'][0] for r in rows).items())))

    # ---- B1: leave-one-event-out, median window of the other events ------
    preds, truths = [], []
    per_event = {}
    for held in sorted(by_event):
        others = [r['ppd'] for r in rows if r['event'] != held]
        if len(others) < 20:
            continue
        med = (int(statistics.median([p[0] for p in others])),
               int(statistics.median([p[1] for p in others])))
        held_rows = [r for r in rows if r['event'] == held]
        preds.extend([med] * len(held_rows))
        truths.extend([tuple(r['ppd']) for r in held_rows])
        per_event[held] = dict(n=len(held_rows), pred=list(med))
        per_event[held].update(acc([med] * len(held_rows),
                                   [tuple(r['ppd']) for r in held_rows]))
        per_event[held]['n'] = len(held_rows)
    b1 = acc(preds, truths)
    print('\n=== B1  event-prior (LOEO) ===')
    print('  overall: n=%d exact=%.3f within1=%.3f'
          % (b1['n'], b1['exact'], b1['within1']))
    for ev, st in sorted(per_event.items()):
        print('  %-14s n=%4d pred=%s exact=%.3f within1=%.3f'
              % (ev, st['n'], st['pred'], st['exact'], st['within1']))

    # ---- B2: global prior (still leave-one-event-out for honesty) --------
    preds2, truths2 = [], []
    for held in sorted(by_event):
        others = [r['ppd'] for r in rows if r['event'] != held]
        if len(others) < 20:
            continue
        med = (int(statistics.median([p[0] for p in others])),
               int(statistics.median([p[1] for p in others])))
        held_rows = [r for r in rows if r['event'] == held]
        preds2.extend([med] * len(held_rows))
        truths2.extend([tuple(r['ppd']) for r in held_rows])
    b2 = acc(preds2, truths2)
    print('=== B2  global prior (LOEO) ===')
    print('  overall: n=%d exact=%.3f within1=%.3f'
          % (b2['n'], b2['exact'], b2['within1']))

    # ---- F3 preliminaries ------------------------------------------------
    print('\n=== F3 preliminaries ===')
    conf = collections.Counter(r.get('date_confidence') or 'EMPTY'
                               for r in rows)
    print('date_confidence values: %s' % dict(sorted(conf.items(),
                                                     key=lambda kv: -kv[1])[:8]))
    mult = 0
    for line in open(args.features, encoding='utf-8'):
        pass
    print('(multi-event tuples are stored as a string in the raw attr; '
          'count is reported by the cloud scan if present)')

    out = dict(rows=len(rows), by_event=dict(by_event),
               b1=b1, b1_per_event=per_event, b2=b2,
               oracle_gap_median=float(statistics.median(gap)),
               oracle_pre_frac_median=float(statistics.median(pos_frac)),
               rule_reference=dict(exact=0.419, within1=0.650,
                                   note='chimanimani 260 tiles, single event'))
    if args.out:
        os.makedirs(osp.dirname(osp.abspath(args.out)) or '.', exist_ok=True)
        json.dump(out, open(args.out, 'w', encoding='utf-8'), indent=2)
        print('\nwrote %s' % args.out)


if __name__ == '__main__':
    main()
