# Copyright (c) OpenMMLab. All rights reserved.
"""Summarize local M1 ablation dry-run logs."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ABLATION_ROOT = ROOT / 'work_dirs' / 'm2_pf_rpn' / 'ablations'
METRICS = ('loss', 'grad_norm', 'memory', 'time', 'loss_cls',
           'loss_bbox', 'loss_iou', 'loss_ctr')


def _extract(line):
    result = {}
    for name in METRICS:
        match = re.search(rf'{name}: ([0-9.eE+-]+)', line)
        if match:
            result[name] = float(match.group(1))
    return result


def main():
    rows = []
    for log_path in sorted(ABLATION_ROOT.glob('*/train.log')):
        name = log_path.parent.name
        last = None
        for line in log_path.read_text(
                encoding='utf-8', errors='replace').splitlines():
            if 'Epoch(train)' in line and 'grad_norm:' in line:
                last = line
        if last is None:
            rows.append(dict(name=name, status='no-metric'))
            continue
        rows.append(dict(name=name, status='ok', metrics=_extract(last)))

    for row in rows:
        metrics = row.get('metrics', {})
        print(
            f"{row['name']:32s} loss={metrics.get('loss')} "
            f"grad_norm={metrics.get('grad_norm')} "
            f"memory={metrics.get('memory')} "
            f"time={metrics.get('time')}")

    out = ABLATION_ROOT / 'summary.json'
    out.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(f'summary written to {out}')


if __name__ == '__main__':
    main()