# Copyright (c) OpenMMLab. All rights reserved.
"""Evaluate cloud A0-A9 checkpoints and export the M2-facing proposals.

Runs ``tools/eval_m1_split.py`` for every finished ablation checkpoint on the
event-level val and test splits, then exports ``proposal.v1`` artifacts for the
full multimodal model (A6) with ROI features.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from run_cloud_ablations import ABLATIONS, WORK_ROOT  # noqa: E402

CONFIG = ROOT / 'configs' / 'm2-pf-rpn' / 'm2_pf_rpn_cloud_full.py'
METRIC_ROOT = ROOT / 'artifacts' / 'cloud_metrics'
PROPOSAL_ROOT = ROOT / 'artifacts' / 'proposals' / 'cloud'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='*', default=None)
    parser.add_argument('--splits', nargs='+', default=['val', 'test'])
    parser.add_argument('--export-runs', nargs='+', default=[],
                        help='runs to export proposal.v1 for; empty by default '
                             'because the fused-feature npz dump is CPU bound '
                             'for hours and blocks the GPU queue')
    parser.add_argument('--export-splits', nargs='+', default=['test'])
    parser.add_argument('--max-images', type=int, default=None)
    parser.add_argument('--eval-batch-size', type=int, default=8,
                        help='inference batch size for eval_m1_split')
    parser.add_argument('--config', default=str(CONFIG))
    parser.add_argument('--work-root', default=None)
    parser.add_argument('--metric-root', default=None)
    parser.add_argument('--proposal-root', default=None)
    parser.add_argument('--final-epoch', type=int, default=8)
    parser.add_argument('--cfg-options', nargs='*', default=[],
                        help='extra cfg overrides for every eval/export')
    parser.add_argument('--dump-per-image', action='store_true',
                        help='also write per-image dumps for paired statistics')
    parser.add_argument('--missing-runs', nargs='+', default=['A6_add_cross_modal_router'],
                        help='runs to sweep modality dropout for')
    parser.add_argument('--missing-modality', action='store_true',
                        help='also score the full model with sensors dropped')
    parser.add_argument('--missing-split', default='test')
    parser.add_argument('--include-optical-drops', action='store_true',
                        help='also sweep optical-missing combinations')
    return parser.parse_args()


def main():
    args = parse_args()
    work_root = Path(args.work_root) if args.work_root else WORK_ROOT
    metric_root = Path(args.metric_root) if args.metric_root else METRIC_ROOT
    proposal_root = Path(args.proposal_root) if args.proposal_root else PROPOSAL_ROOT
    selected = set(args.runs)
    summary = {}
    for name, overrides in ABLATIONS.items():
        if selected and name not in selected:
            continue
        ckpt = work_root / name / f'epoch_{args.final_epoch}.pth'
        if not ckpt.exists():
            candidates = sorted((work_root / name).glob('epoch_*.pth'))
            if not candidates:
                print(f'[{name}] no checkpoint, skip', flush=True)
                continue
            ckpt = candidates[-1]
        summary[name] = {'checkpoint': str(ckpt)}
        for split in args.splits:
            out = metric_root / name / f'{split}.json'
            out.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(HERE / 'eval_m1_split.py'), args.config,
                   str(ckpt), '--split', split, '--out', str(out),
                   '--batch-size', str(args.eval_batch_size),
                   '--cfg-options', *overrides, *args.cfg_options]
            if args.max_images:
                cmd += ['--max-images', str(args.max_images)]
            if args.dump_per_image:
                cmd += ['--dump-per-image',
                        str(metric_root / name / f'{split}.per_image.npz')]
            print(f'[{name}/{split}] eval', flush=True)
            proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                                  text=True)
            if proc.returncode != 0:
                print(proc.stdout[-2000:], proc.stderr[-2000:], flush=True)
                summary[name][split] = {'status': 'failed'}
                continue
            summary[name][split] = json.loads(out.read_text())
        if args.missing_modality and name in args.missing_runs:
            drops = [['sar'], ['topography'], ['change'],
                     ['sar', 'topography'], ['sar', 'change'],
                     ['topography', 'change'],
                     # optical-only degradation of the full model
                     ['sar', 'topography', 'change']]
            if args.include_optical_drops:
                drops += [['optical'], ['optical', 'sar'],
                          ['optical', 'change'],
                          ['optical', 'sar', 'change']]
            for drop in drops:
                tag = 'drop_' + '_'.join(drop)
                out = metric_root / name / f'{args.missing_split}.{tag}.json'
                cmd = [sys.executable, str(HERE / 'eval_m1_split.py'),
                       args.config, str(ckpt), '--split', args.missing_split,
                       '--out', str(out),
                       '--batch-size', str(args.eval_batch_size),
                       '--drop-modalities', *drop,
                       '--cfg-options', *overrides, *args.cfg_options]
                if args.dump_per_image:
                    cmd += ['--dump-per-image', str(
                        metric_root / name /
                        f'{args.missing_split}.{tag}.per_image.npz')]
                print(f'[{name}/{args.missing_split}] {tag}', flush=True)
                subprocess.run(cmd, cwd=str(ROOT))
                if out.exists():
                    summary[name][tag] = json.loads(out.read_text())
        if name in args.export_runs:
            for split in args.export_splits:
                out_dir = proposal_root / split
                cmd = [sys.executable, str(HERE / 'inference' / 'export_proposals.py'),
                       args.config, str(ckpt), '--out', str(out_dir),
                       '--split', split, '--cfg-options', *overrides, *args.cfg_options]
                print(f'[{name}/{split}] export proposals', flush=True)
                subprocess.run(cmd, cwd=str(ROOT))
        (metric_root / 'summary.json').write_text(
            json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: {s: (v.get(s) or {}).get('AR')
                          for s in ('val', 'test')}
                      for k, v in summary.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()