#!/usr/bin/env python
"""F1' : is the annotation window reproducible from the calendar alone?

For every annotated tile we know the event date (a public fact, e.g. from a
disaster database) and the tile's own 15 acquisition dates (carried inside the
NetCDF patch).  This script asks whether "bracket the event date with the two
nearest acquisitions" reproduces the published ``pre_post_dates`` window.

If it does, the "which frames" question is not a learning problem at all: it is
a calendar rule that needs no imagery, no annotation and no new training, and
which can be applied to negatives via a neighbouring event's calendar.
"""
import argparse
import collections
import json
import os
import os.path as osp
import statistics


def d(s):
    import datetime
    return datetime.date.fromisoformat(s)


def bracket(dates, event, mode='strict'):
    """Return (pre, post) indices bracketing the event date, or None."""
    ds = [d(x) for x in dates]
    e = d(event)
    if mode == 'strict':
        pre = max((i for i, x in enumerate(ds) if x < e), default=None)
        post = min((j for j, x in enumerate(ds) if x > e), default=None)
    elif mode == 'inclusive':
        pre = max((i for i, x in enumerate(ds) if x <= e), default=None)
        post = min((j for j, x in enumerate(ds) if x > e), default=None)
    elif mode == 'nearest':
        pre = min(range(len(ds)), key=lambda i: abs((ds[i] - e).days))
        post = pre + 1 if pre + 1 < len(ds) else pre
    else:
        raise ValueError(mode)
    if pre is None or post is None or pre >= post:
        return None
    return int(pre), int(post)


def acc(pairs):
    if not pairs:
        return dict(n=0, exact=0.0, within1=0.0)
    n = len(pairs)
    return dict(n=n,
                exact=sum(1 for p, t in pairs if p == t) / n,
                within1=sum(1 for p, t in pairs
                            if abs(p[0] - t[0]) <= 1 and
                            abs(p[1] - t[1]) <= 1) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dates', required=True)
    ap.add_argument('--features', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    event_of = {}
    for line in open(args.features, encoding='utf-8'):
        r = json.loads(line)
        event_of[r['tile']] = r
    # event id comes from the manifests via the features' tile prefix map built
    # by f1_baselines; here we only need (tile -> ppd, event_date)
    rows = []
    for line in open(args.dates, encoding='utf-8'):
        r = json.loads(line)
        f = event_of.get(r['tile'])
        row_date = (r.get('event_date') or '').strip()
        if not f or not f.get('ppd') or not row_date or row_date in ('None', 'nan'):
            continue
        try:
            d(row_date)
        except ValueError:
            continue
        r['event_date'] = row_date
        r['ppd'] = tuple(f['ppd'])
        rows.append(r)
    print('annotated tiles with event_date + acquisition calendar: %d' % len(rows))

    out = {}
    for mode in ('strict', 'inclusive', 'nearest'):
        pairs, missing = [], 0
        for r in rows:
            b = bracket(r['dates'], r['event_date'], mode)
            if b is None:
                missing += 1
                continue
            pairs.append((b, r['ppd']))
        st = acc(pairs)
        st['unbracketable'] = missing
        out[mode] = st
        print('  %-10s n=%4d exact=%.3f within1=%.3f  unbracketable=%d'
              % (mode, st['n'], st['exact'], st['within1'], missing))

    # how far is the event from its bracketing frames?
    gaps_pre, gaps_post = [], []
    for r in rows:
        ds = [d(x) for x in r['dates']]
        e = d(r['event_date'])
        pre = max((i for i, x in enumerate(ds) if x < e), default=None)
        post = min((j for j, x in enumerate(ds) if x > e), default=None)
        if pre is None or post is None:
            continue
        gaps_pre.append((e - ds[pre]).days)
        gaps_post.append((ds[post] - e).days)
    if gaps_pre:
        print('\n  event -> previous acquisition: median %d d (p90 %d d)'
              % (statistics.median(gaps_pre),
                 sorted(gaps_pre)[int(0.9 * len(gaps_pre))]))
        print('  event -> next acquisition:     median %d d (p90 %d d)'
              % (statistics.median(gaps_post),
                 sorted(gaps_post)[int(0.9 * len(gaps_post))]))

    # per-event breakdown of the strict rule vs the published window
    print('\n  per-tile-bracket agreement by event (strict):')
    by_ev = collections.defaultdict(list)
    for r in rows:
        ev = r['tile'].rsplit('_', 1)[0]
        for known in ('kyrgyzstan1', 'kyrgyzstan2', 'newzealand', 'chimanimani',
                      'hiroshima', 'hokkaido', 'indonesia', 'china', 'italy',
                      'itogon', 'usa'):
            if r['tile'].startswith(known):
                ev = known
                break
        b = bracket(r['dates'], r['event_date'], 'strict')
        if b is not None:
            by_ev[ev].append((b, r['ppd']))
    for ev, prs in sorted(by_ev.items(), key=lambda kv: -len(kv[1])):
        st = acc(prs)
        print('    %-14s n=%4d exact=%.3f within1=%.3f'
              % (ev, st['n'], st['exact'], st['within1']))

    if args.out:
        os.makedirs(osp.dirname(osp.abspath(args.out)) or '.', exist_ok=True)
        json.dump(dict(modes=out,
                       per_event={k: acc(v) for k, v in by_ev.items()}),
                  open(args.out, 'w', encoding='utf-8'), indent=2)
        print('\nwrote %s' % args.out)


if __name__ == '__main__':
    main()
