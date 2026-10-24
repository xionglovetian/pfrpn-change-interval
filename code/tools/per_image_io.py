# Copyright (c) OpenMMLab. All rights reserved.
"""Shared per-image metric dump for the M1 evaluators and paired statistics.

A dump is a compressed ``.npz`` describing one evaluated run on one split:

* ``image_id`` / ``event_id`` (U)  -- pairing keys, one entry per scored image
* ``num_gt`` (int64)               -- GT boxes on the image (0 => negative tile)
* ``ar_<k>`` (float64)             -- per-image recall@k, 0.0 when ``num_gt == 0``
* ``scores`` / ``flags`` / ``offsets``
      concatenated, score-sorted detections for the whole split, so a
      recall@FPPI curve can be rebuilt after resampling images.

``tools/eval_m1_split.py`` (runs the model) and
``tools/eval_proposal_metrics.py`` (scores exported ``proposal.v1``) both write
this layout; ``tools/paired_stats.py`` consumes it.  Keeping the format in one
module stops the two evaluators from drifting apart.
"""
from __future__ import annotations

import json

import numpy as np

SCHEMA = 'm1_per_image.v1'


def save_dump(path, *, image_ids, event_ids, num_gt, ar_by_k, scores, flags,
              offsets, meta=None):
    """Write a per-image dump to ``path``.

    ``ar_by_k`` maps ``k`` -> per-image recall array aligned with ``image_ids``.
    """
    payload = {
        'schema': np.asarray(SCHEMA),
        'image_id': np.asarray([str(x) for x in image_ids], dtype='U'),
        'event_id': np.asarray([str(x) for x in event_ids], dtype='U'),
        'num_gt': np.asarray(num_gt, dtype=np.int64),
        'scores': np.asarray(scores, dtype=np.float64),
        'flags': np.asarray(flags, dtype=bool),
        'offsets': np.asarray(offsets, dtype=np.int64),
    }
    for k, values in ar_by_k.items():
        payload[f'ar_{k}'] = np.asarray(values, dtype=np.float64)
    if meta is not None:
        payload['meta'] = np.asarray(json.dumps(meta))
    np.savez_compressed(path, **payload)


def load_dump(path):
    """Load a per-image dump written by :func:`save_dump`."""
    with np.load(path, allow_pickle=False) as data:
        files = {name: data[name] for name in data.files}
    schema = str(files.get('schema', np.asarray('')))
    if schema != SCHEMA:
        raise ValueError(f'{path}: unexpected dump schema {schema!r}')
    ar = {}
    for name, value in files.items():
        if name.startswith('ar_'):
            ar[int(name[3:])] = np.asarray(value, dtype=np.float64)
    return {
        'schema': schema,
        'image_id': [str(x) for x in files['image_id']],
        'event_id': [str(x) for x in files['event_id']],
        'num_gt': np.asarray(files['num_gt'], dtype=np.int64),
        'ar': ar,
        'scores': np.asarray(files['scores'], dtype=np.float64),
        'flags': np.asarray(files['flags'], dtype=bool),
        'offsets': np.asarray(files['offsets'], dtype=np.int64),
        'meta': json.loads(str(files['meta'])) if 'meta' in files else {},
    }


def align(a, b):
    """Return ``(idx_a, idx_b)`` that pair two dumps by ``image_id``.

    Pairing statistics are only meaningful when both runs scored exactly the
    same images, so a mismatch raises instead of silently dropping tiles.
    """
    ia = {str(x): i for i, x in enumerate(a['image_id'])}
    ib = {str(x): i for i, x in enumerate(b['image_id'])}
    if set(ia) != set(ib):
        only_a = sorted(set(ia) - set(ib))[:5]
        only_b = sorted(set(ib) - set(ia))[:5]
        raise ValueError(
            'image sets differ between the two dumps '
            f'(a-only={only_a}, b-only={only_b})')
    order = sorted(ia)
    return (np.asarray([ia[k] for k in order], dtype=np.int64),
            np.asarray([ib[k] for k in order], dtype=np.int64))


def iter_image_spans(dump, index):
    """Yield ``(lo, hi)`` slices into ``scores``/``flags`` for each image."""
    offsets = dump['offsets']
    for i in index:
        yield int(offsets[i]), int(offsets[i + 1])
