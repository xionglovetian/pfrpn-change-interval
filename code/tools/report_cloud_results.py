# Copyright (c) OpenMMLab. All rights reserved.
"""Render cloud A0-A9 metrics into paper-facing tables.

Reads ``artifacts/cloud_metrics/summary.json`` (written by
``tools/eval_cloud_ablations.py``) and emits a Markdown report plus CSV files
with the RESEARCH_ROADMAP.md section 5.9 metrics.
"""
import argparse
import csv
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
    'A0_rgb': 'A0 RGB baseline',
    'A1_optical_topography': 'A1 Optical + DEM',
    'A2_optical_sar': 'A2 Optical + SAR',
    'A3_optical_sar_topography': 'A3 Optical + SAR + DEM',
    'A4_add_change': 'A4 A3 + temporal change',
    'A5_add_modality_token': 'A5 A4 + terrain-aware token',
    'A6_add_cross_modal_router': 'A6 A5 + cross-modal router',
    'A7_no_router': 'A7 A6 w/o router',
    'A8_no_sar': 'A8 A6 w/o SAR',
    'A9_no_change': 'A9 A6 w/o change',
}
DROP_LABELS = {
    'drop_sar': 'w/o SAR',
    'drop_topography': 'w/o DEM/topography',
    'drop_change': 'w/o temporal change',
    'drop_sar_topography': 'w/o SAR + DEM',
    'drop_sar_change': 'w/o SAR + change',
    'drop_topography_change': 'w/o DEM + change',
    'drop_sar_topography_change': 'w/o SAR + DEM + change (optical only)',
    'drop_optical': 'w/o optical',
    'drop_optical_sar': 'w/o optical + SAR',
    'drop_optical_change': 'w/o optical + change',
    'drop_optical_sar_change': 'w/o optical + SAR + change (DEM only)',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', default=str(DEFAULT_SUMMARY))
    parser.add_argument('--out-md',
                        default=str(ROOT / 'docs' / 'cloud_m1_results.md'))
    parser.add_argument('--out-dir', default=str(ROOT / 'artifacts'
                                                 / 'cloud_metrics' / 'tables'))
    parser.add_argument('--protocol-note', default=None,
                        help='override the data/protocol header paragraph')
    parser.add_argument('--ckpt-root',
                        default='work_dirs/m2_pf_rpn/cloud_ablations',
                        help='checkpoint root shown in the repro section')
    parser.add_argument('--seed-note', default='42',
                        help='seed description shown in the repro section')
    parser.add_argument('--split-report',
                        default='data/landslide_multimodal_full/manifests/split_report.json')
    return parser.parse_args()


def _fmt(value, digits=4):
    if value is None:
        return 'n/a'
    if isinstance(value, str):
        return value
    return f'{value:.{digits}f}'


def _get(block, *path, default=None):
    node = block
    for key in path:
        if not isinstance(node, dict):
            return default
        node = node.get(key)
        if node is None:
            return default
    return node


def _row(name, metrics):
    return {
        'run': name,
        'label': LABELS.get(name, name),
        'AR@100': _get(metrics, 'AR', 'AR@100'),
        'AR@300': _get(metrics, 'AR', 'AR@300'),
        'AR@900': _get(metrics, 'AR', 'AR@900'),
        'AR_coco': metrics.get('AR_coco_style'),
        'Recall@0.1FPPI': _get(metrics, 'recall_at_fppi',
                               'recall@0.1FPPI'),
        'Recall@1FPPI': _get(metrics, 'recall_at_fppi', 'recall@1.0FPPI'),
        'Recall@10FPPI': _get(metrics, 'recall_at_fppi', 'recall@10.0FPPI'),
        'recall_small': _get(metrics, 'recall_by_size', 'small'),
        'recall_medium': _get(metrics, 'recall_by_size', 'medium'),
        'recall_large': _get(metrics, 'recall_by_size', 'large'),
        'num_images': metrics.get('num_images'),
        'num_gt': metrics.get('num_gt_boxes'),
    }


def _table(rows, title):
    head = ('| Run | AR@100 | AR@300 | AR@900 | AR(0.5:0.95) | '
            'R@0.1FPPI | R@1FPPI | R@10FPPI | R-small | R-med | R-large |')
    sep = '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|'
    lines = [f'### {title}', '', head, sep]
    for row in rows:
        lines.append(
            '| {label} | {a100} | {a300} | {a900} | {acoco} | {f01} | '
            '{f1} | {f10} | {rs} | {rm} | {rl} |'.format(
                label=row['label'],
                a100=_fmt(row['AR@100']), a300=_fmt(row['AR@300']),
                a900=_fmt(row['AR@900']), acoco=_fmt(row['AR_coco']),
                f01=_fmt(row['Recall@0.1FPPI']),
                f1=_fmt(row['Recall@1FPPI']),
                f10=_fmt(row['Recall@10FPPI']),
                rs=_fmt(row['recall_small']),
                rm=_fmt(row['recall_medium']),
                rl=_fmt(row['recall_large'])))
    lines.append('')
    return lines


def main():
    args = parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding='utf-8'))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    default_protocol = (
        '数据：Sen12Landslides S12LS-LD；事件级 split（balance，seed 42）；'
        '320×320 / batch 8 / lr 2e-5 / AMP(loss_scale=1.0) / 8 epoch / seed 42；'
        '变化模态为长基线 (0,-1)，含 S12LS-AD 难负样本'
        '（train 6813 = 3067 正 + 3746 负）。')
    md = ['# 模块一 M2-PF-RPN 云端实验结果', '',
          args.protocol_note or default_protocol, '']
    fieldnames = None
    for split in ('test', 'val'):
        rows = []
        for name in ORDER:
            block = summary.get(name)
            metrics = block.get(split) if isinstance(block, dict) else None
            if isinstance(metrics, dict) and 'AR' in metrics:
                rows.append(_row(name, metrics))
        if not rows:
            continue
        md += _table(rows, f'{split} 集（事件级）')
        if fieldnames is None:
            fieldnames = list(rows[0])
            with open(out_dir / 'ablations_test.csv', 'w', newline='',
                      encoding='utf-8') as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        else:
            with open(out_dir / 'ablations_val.csv', 'w', newline='',
                      encoding='utf-8') as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    a6 = summary.get('A6_add_cross_modal_router')
    if isinstance(a6, dict):
        drop_rows = []
        for key, label in DROP_LABELS.items():
            metrics = a6.get(key)
            if isinstance(metrics, dict) and 'AR' in metrics:
                row = _row('A6_add_cross_modal_router', metrics)
                row['label'] = label
                drop_rows.append(row)
        if drop_rows:
            md += ['### A6 缺失模态鲁棒性（test 集）', '']
            md += _table(drop_rows, 'A6 缺失模态')[0:-1]
            with open(out_dir / 'missing_modality_test.csv', 'w',
                      newline='', encoding='utf-8') as fh:
                writer = csv.DictWriter(fh, fieldnames=list(drop_rows[0]))
                writer.writeheader()
                writer.writerows(drop_rows)

    md += ['## 复现信息', '',
           '- 配置：`configs/m2-pf-rpn/m2_pf_rpn_cloud_full.py`',
           '- 训练驱动：`tools/run_cloud_ablations.py`（A0–A9 覆盖项见文件）',
           '- 评测：`tools/eval_m1_split.py` / '
           '`tools/eval_cloud_ablations.py`',
           '- 评分口径：IoU 0.5 的 COCO 风格 AR；Recall@FPPI 由置信度阈值'
           '扫描得到；小/中/大按 GT 面积 <32²、32²–96²、≥96² 划分',
           f'- checkpoint：`{args.ckpt_root}/<run>/epoch_8.pth`',
           f'- 随机种子：{args.seed_note}；事件级 split 报告：'
           f'`{args.split_report}`', '']
    Path(args.out_md).write_text('\n'.join(md) + '\n', encoding='utf-8')
    rows_written = sum(1 for line in md if line.startswith('| A'))
    print(f'wrote {args.out_md} ({rows_written} table rows); '
          f'tables in {out_dir}')


if __name__ == '__main__':
    main()