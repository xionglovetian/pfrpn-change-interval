# Copyright (c) OpenMMLab. All rights reserved.
"""Render the cloud A0-A9 metrics into a Markdown / CSV ablation table."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / 'artifacts' / 'cloud_metrics' / 'summary.json'

ORDER = [
    'A0_rgb', 'A1_optical_topography', 'A2_optical_sar',
    'A3_optical_sar_topography', 'A4_add_change', 'A5_add_modality_token',
    'A6_add_cross_modal_router', 'A7_no_router', 'A8_no_sar', 'A9_no_change',
]
LABELS = {
    'A0_rgb': 'A0 RGB PF-RPN',
    'A1_optical_topography': 'A1 Optical+DEM',
    'A2_optical_sar': 'A2 Optical+SAR',
    'A3_optical_sar_topography': 'A3 Optical+SAR+DEM',
    'A4_add_change': 'A4 +temporal change',
    'A5_add_modality_token': 'A5 +terrain-aware token',
    'A6_add_cross_modal_router': 'A6 +cross-modal router',
    'A7_no_router': 'A7 A6 w/o router',
    'A8_no_sar': 'A8 A6 w/o SAR',
    'A9_no_change': 'A9 A6 w/o change',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', default=str(DEFAULT_SUMMARY))
    parser.add_argument('--out', default=None)
    parser.add_argument('--split-prefix', default='test')
    return parser.parse_args()


def _ar(block, key):
    if not isinstance(block, dict):
        return None
    ar = block.get('AR')
    if not isinstance(ar, dict):
        return None
    value = ar.get(key)
    return value


def main():
    args = parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding='utf-8'))
    lines = [
        '| Run | Split | AR@100 | AR@300 | AR@900 | Recall@1FPPI |',
        '|---|---|---:|---:|---:|---:|',
    ]
    rows = []
    for name in ORDER:
        block = summary.get(name)
        if not isinstance(block, dict):
            continue
        for split in (args.split_prefix,):
            metrics = block.get(split)
            if not isinstance(metrics, dict) or 'AR' not in metrics:
                continue
            fppi = (metrics.get('recall_at_fppi') or {}).get(
                'recall@1.0FPPI')
            row = dict(
                run=LABELS.get(name, name), split=split,
                ar100=_ar(metrics, 'AR@100'), ar300=_ar(metrics, 'AR@300'),
                ar900=_ar(metrics, 'AR@900'), recall_fppi=fppi)
            rows.append(row)

            def fmt(v):
                return 'n/a' if v is None else f'{v:.4f}'

            lines.append(
                f'| {row["run"]} | {split} | {fmt(row["ar100"])} | '
                f'{fmt(row["ar300"])} | {fmt(row["ar900"])} | '
                f'{fmt(row["recall_fppi"])} |')
    text = '\n'.join(lines) + '\n'
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding='utf-8')
    missing = [n for n in ORDER if n not in summary]
    if missing:
        print('missing runs:', missing)


if __name__ == '__main__':
    main()