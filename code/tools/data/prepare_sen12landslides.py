# Copyright (c) OpenMMLab. All rights reserved.
"""Convert Sen12Landslides NetCDF patches to M1 multimodal files."""
import argparse
import ast
import json
import os.path as osp
from pathlib import Path

import cv2
import numpy as np
import xarray as xr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_root')
    parser.add_argument('--task', default='S12LS-LD')
    parser.add_argument('--version', default='harmonized')
    parser.add_argument('--out', default='E:/PF-RPN/data/landslide_multimodal')
    parser.add_argument('--splits-file', default=None)
    parser.add_argument('--sar-source', choices=['asc', 'dsc', 'both'],
                        default='asc',
                        help='both stacks asc+dsc into 4ch pre / 4ch post')
    parser.add_argument('--max-per-split', type=int, default=None)
    parser.add_argument('--min-area', type=int, default=4)
    parser.add_argument('--pre-post-rule', choices=['long-baseline', 'auto'],
                        default='long-baseline',
                        help='long-baseline = first/last acquisition of every '
                             'tile (label free, required with hard negatives); '
                             'auto = legacy annotation-derived event window')
    parser.add_argument('--skip-annotated', action='store_true',
                        help='skip entries with pixel_annotated>0 (AD negatives only)')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def _pre_post(attrs, rule='long-baseline'):
    """Return (pre, post) time indices for the change channels.

    ``long-baseline`` (default, recommended) always uses the first and last
    acquisition of the tile's own time series.  It is the only definition that
    is *label free* and therefore identical for annotated (positive) and
    unannotated (negative) tiles -- using the annotation-derived event window
    for positives while negatives must fall back to the full span would give
    the model a trivial shortcut (``event-window channel non-zero => positive``).

    ``auto`` reproduces the legacy behaviour: parse ``pre_post_dates`` (a dict
    for single-event tiles, a tuple of dicts -- the dating module's detected
    NDVI drops sorted by confidence -- for multi-event tiles) and fall back to
    the long baseline when nothing usable is present.
    """
    if rule == 'long-baseline':
        return 0, -1
    raw = attrs.get('pre_post_dates')
    pre, post = 0, -1
    parsed = raw if isinstance(raw, dict) else None
    if parsed is None and isinstance(raw, str) and raw.strip():
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            parsed = None
    if isinstance(parsed, dict):
        pre = int(parsed.get('pre', pre))
        post = int(parsed.get('post', post))
    return pre, post


def _band(ds, name, time_idx):
    return ds[name].isel(time=time_idx).transpose(
        'y', 'x').values.astype(np.float32)


def _topography(dem):
    dem = np.nan_to_num(np.asarray(dem, dtype=np.float32), nan=0.0)
    dem = np.clip(dem, 0.0, 8800.0)
    gy, gx = np.gradient(dem, 10.0, 10.0)
    slope = np.degrees(np.arctan(np.sqrt(gx ** 2 + gy ** 2)))
    aspect = np.degrees(np.arctan2(-gy, gx)) % 360.0
    curvature = -(np.gradient(gx, 10.0, axis=1) +
                  np.gradient(gy, 10.0, axis=0))
    twi = np.log(1.0 / (np.tan(np.deg2rad(np.clip(slope, 0.1, 89.0))) +
                        1e-3))
    return np.stack([dem, slope, aspect, curvature, twi], -1).astype(
        np.float32)


def _load_s2(path, pre_post_rule='long-baseline'):
    with xr.open_dataset(path) as ds:
        ds = ds.sortby('time')
        pre, post = _pre_post(ds.attrs, pre_post_rule)
        pre_rgb = np.stack(
            [_band(ds, 'B04', pre), _band(ds, 'B03', pre),
             _band(ds, 'B02', pre)], -1)
        b04p = _band(ds, 'B04', pre)
        b03p = _band(ds, 'B03', pre)
        b08p = _band(ds, 'B08', pre)
        b12p = _band(ds, 'B12', pre)
        b04q = _band(ds, 'B04', post)
        b03q = _band(ds, 'B03', post)
        b08q = _band(ds, 'B08', post)
        b12q = _band(ds, 'B12', post)
        ndvi_p = (b08p - b04p) / (b08p + b04p + 1e-6)
        ndvi_q = (b08q - b04q) / (b08q + b04q + 1e-6)
        ndwi_p = (b03p - b08p) / (b03p + b08p + 1e-6)
        ndwi_q = (b03q - b08q) / (b03q + b08q + 1e-6)
        nbr_p = (b08p - b12p) / (b08p + b12p + 1e-6)
        nbr_q = (b08q - b12q) / (b08q + b12q + 1e-6)
        change = np.stack([ndvi_q - ndvi_p, ndwi_q - ndwi_p,
                           nbr_q - nbr_p], -1).astype(np.float32)
        topo = _topography(_band(ds, 'DEM', 0))
        mask = (_band(ds, 'MASK', 0) > 0).astype(np.uint8)
    return pre_rgb.astype(np.float32), change, topo, mask


def _boxes(mask, min_area):
    num, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for label in range(1, num):
        if int(stats[label, cv2.CC_STAT_AREA]) < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        boxes.append([float(x), float(y), float(x + w), float(y + h)])
    return boxes
def _load_s1(path, pre_post_rule='long-baseline'):
    with xr.open_dataset(path) as ds:
        ds = ds.sortby('time')
        pre, post = _pre_post(ds.attrs, pre_post_rule)
        pre = np.stack([_band(ds, 'VV', pre), _band(ds, 'VH', pre)], -1)
        post = np.stack([_band(ds, 'VV', post), _band(ds, 'VH', post)], -1)
    return pre.astype(np.float32), post.astype(np.float32)


def _resolve(root, relative):
    return Path(root) / 'data' / 'data_harmonized' / relative


def _splits(args):
    path = (Path(args.dataset_root) / 'tasks' / args.task /
            args.version / 'splits_aligned.json')
    if args.splits_file:
        path = Path(args.splits_file)
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def _process(entry, args, root):
    s2 = _resolve(root, entry['s2'])
    if not s2.exists():
        return None
    optical, change, topo, mask = _load_s2(s2, args.pre_post_rule)
    boxes = _boxes(mask, args.min_area)
    if args.sar_source == 'both':
        sar_keys = ['s1asc', 's1dsc']
    else:
        sar_keys = ['s1asc' if args.sar_source == 'asc' else 's1dsc']
    sar_parts_pre, sar_parts_post = [], []
    for sar_key in sar_keys:
        sar_rel = entry.get(sar_key)
        sar_path = _resolve(root, sar_rel) if sar_rel else None
        if sar_path is not None and sar_path.exists():
            part_pre, part_post = _load_s1(sar_path, args.pre_post_rule)
        elif args.sar_source == 'both':
            # keep a fixed 4ch+4ch layout even when one orbit is absent
            _h, _w = optical.shape[:2]
            part_pre = np.zeros((_h, _w, 2), dtype=np.float32)
            part_post = np.zeros((_h, _w, 2), dtype=np.float32)
        else:
            continue
        sar_parts_pre.append(part_pre)
        sar_parts_post.append(part_post)
    if sar_parts_pre:
        sar_pre = np.concatenate(sar_parts_pre, axis=-1)
        sar_post = np.concatenate(sar_parts_post, axis=-1)
        sar_mask = 1
    else:
        h, w = optical.shape[:2]
        sar_pre = np.zeros((h, w, 2), dtype=np.float32)
        sar_post = np.zeros((h, w, 2), dtype=np.float32)
        sar_mask = 0
    return optical, sar_pre, sar_post, topo, change, mask, boxes, sar_mask


def _entry(event_id, tile_id, arrays, out_root, tile_dir):
    optical, sar_pre, sar_post, topo, change, mask, boxes, sar_mask = arrays
    rel = osp.relpath(tile_dir, out_root).replace('\\', '/')
    annotations = [
        dict(bbox_xyxy=box, category_id=1) for box in boxes
    ]
    return dict(
        sample_id=f'{event_id}_{tile_id}',
        event_id=event_id,
        image_id=tile_id,
        shape=list(optical.shape[:2]),
        modality_mask=[1, sar_mask, 1, 1],
        modalities=dict(
            optical_pre=f'{rel}/optical_pre.npy',
            sar_pre=f'{rel}/sar_pre.npy',
            sar_post=f'{rel}/sar_post.npy',
            topography=f'{rel}/topography.npy',
            change=f'{rel}/change.npy'),
        labels=dict(annotations=annotations, negatives=[]))


def main():
    args = parse_args()
    root = Path(args.dataset_root)
    out_root = Path(args.out)
    for split, entries in _splits(args).items():
        manifest = []
        used = 0
        skipped = 0
        for entry in entries:
            if args.max_per_split is not None and used >= args.max_per_split:
                break
            if args.skip_annotated and int(entry.get('pixel_annotated', 0)) > 0:
                continue
            try:
                arrays = _process(entry, args, root)
            except Exception as exc:  # noqa: BLE001
                print(f'{split}: skip {entry.get("id")}: '
                      f'{type(exc).__name__}: {exc}', flush=True)
                skipped += 1
                continue
            if arrays is None:
                continue
            optical, sar_pre, sar_post, topo, change, mask, _, _ = arrays
            event_id = str(entry.get('inventory', 'unknown'))
            tile_id = str(entry['id'])
            tile_dir = out_root / 'events' / event_id / 'tiles' / tile_id
            if not (tile_dir / 'optical_pre.npy').exists() or args.overwrite:
                tile_dir.mkdir(parents=True, exist_ok=True)
                np.save(tile_dir / 'optical_pre.npy', optical)
                np.save(tile_dir / 'sar_pre.npy', sar_pre)
                np.save(tile_dir / 'sar_post.npy', sar_post)
                np.save(tile_dir / 'topography.npy', topo)
                np.save(tile_dir / 'change.npy', change)
                np.save(tile_dir / 'mask.npy', mask)
            manifest.append(_entry(
                event_id, tile_id, arrays, out_root, tile_dir))
            used += 1
            if used % 25 == 0:
                print(f'{split}: {used} tiles', flush=True)
        manifest_dir = out_root / 'manifests'
        manifest_dir.mkdir(parents=True, exist_ok=True)
        with open(manifest_dir / f'{split}.json', 'w',
                  encoding='utf-8') as file:
            json.dump(manifest, file, indent=2, ensure_ascii=False)
        print(f'{split}: wrote {len(manifest)} tiles, '
              f'skipped {skipped}', flush=True)


if __name__ == '__main__':
    main()
