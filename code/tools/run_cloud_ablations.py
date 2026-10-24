# Copyright (c) OpenMMLab. All rights reserved.
"""Run M1 A0-A9 ablations on the cloud (event-level split, full data).

The override matrix mirrors ``tools/run_local_ablations.py`` so local and
cloud numbers are directly comparable.  Each run writes ``train.log`` and a
``last_checkpoint`` file into its own work dir; ``summary.json`` collects the
final training metrics and the checkpoint path.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs' / 'm2-pf-rpn' / 'm2_pf_rpn_cloud_full.py'
WORK_ROOT = ROOT / 'work_dirs' / 'm2_pf_rpn' / 'cloud_ablations'

ABLATIONS = {
    'A0_rgb': [
        'model.use_sar=False',
        'model.use_topography=False',
        'model.use_change=False',
        'model.use_modality_tokens=False',
        'model.use_cross_modal_router=False',
        'model.use_router=False',
    ],
    'A1_optical_topography': [
        'model.use_sar=False',
        'model.use_topography=True',
        'model.use_change=False',
        'model.use_modality_tokens=False',
        'model.use_cross_modal_router=False',
        'model.use_router=False',
    ],
    'A2_optical_sar': [
        'model.use_sar=True',
        'model.use_topography=False',
        'model.use_change=False',
        'model.use_modality_tokens=False',
        'model.use_cross_modal_router=False',
        'model.use_router=False',
    ],
    'A3_optical_sar_topography': [
        'model.use_sar=True',
        'model.use_topography=True',
        'model.use_change=False',
        'model.use_modality_tokens=False',
        'model.use_cross_modal_router=False',
        'model.use_router=False',
    ],
    'A4_add_change': [
        'model.use_sar=True',
        'model.use_topography=True',
        'model.use_change=True',
        'model.use_modality_tokens=False',
        'model.use_cross_modal_router=False',
        'model.use_router=False',
    ],
    'A5_add_modality_token': [
        'model.use_sar=True',
        'model.use_topography=True',
        'model.use_change=True',
        'model.use_modality_tokens=True',
        'model.use_cross_modal_router=False',
        'model.use_router=False',
    ],
    'A6_add_cross_modal_router': [
        'model.use_sar=True',
        'model.use_topography=True',
        'model.use_change=True',
        'model.use_modality_tokens=True',
        'model.use_cross_modal_router=True',
        'model.use_router=True',
    ],
    'A7_no_router': [
        'model.use_sar=True',
        'model.use_topography=True',
        'model.use_change=True',
        'model.use_modality_tokens=True',
        'model.use_cross_modal_router=True',
        'model.use_router=False',
    ],
    'A8_no_sar': [
        'model.use_sar=False',
        'model.use_topography=True',
        'model.use_change=True',
        'model.use_modality_tokens=True',
        'model.use_cross_modal_router=True',
        'model.use_router=True',
    ],
    'A9_no_change': [
        'model.use_sar=True',
        'model.use_topography=True',
        'model.use_change=False',
        'model.use_modality_tokens=True',
        'model.use_cross_modal_router=True',
        'model.use_router=True',
    ],
}

_METRIC_RE = re.compile(
    r'Epoch\(train\).*?time: ([0-9.]+).*?memory: ([0-9]+).*?'
    r'grad_norm: ([0-9.eE+-]+).*?loss: ([0-9.eE+-]+)')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='*', default=None,
                        help='Subset of A0-A9 names (default: all).')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--final-epoch', type=int, default=8)
    parser.add_argument('--work-root', default=None,
                        help='override the ablation work root')
    parser.add_argument('--force', action='store_true',
                        help='Re-run even if a checkpoint already exists.')
    parser.add_argument('--cfg-options', nargs='*', default=[],
                        help='Extra --cfg-options applied to every run.')
    return parser.parse_args()


def main():
    args = parse_args()
    work_root = Path(args.work_root) if args.work_root else WORK_ROOT
    work_root.mkdir(parents=True, exist_ok=True)
    selected = set(args.runs)
    summary = {}
    for name, overrides in ABLATIONS.items():
        if selected and name not in selected:
            continue
        work_dir = work_root / name
        work_dir.mkdir(parents=True, exist_ok=True)
        final_ckpt = work_dir / f'epoch_{args.final_epoch}.pth'
        last = work_dir / 'last_checkpoint'
        if not args.force and final_ckpt.exists():
            print(f'[{name}] skip (finished: {final_ckpt.name})', flush=True)
            summary[name] = dict(status='skipped')
            continue
        resume = last.exists() and not args.force
        log_path = work_dir / 'train.log'
        cmd = [
            sys.executable,
            str(ROOT / 'tools' / 'train.py'),
            args.config,
            '--work-dir', str(work_dir),
            '--cfg-options', *overrides, *args.cfg_options,
        ]
        if resume:
            cmd.append('--resume')
            print(f'[{name}] resume from {last.read_text().strip()}', flush=True)
        else:
            print(f'[{name}] start', flush=True)
        last_metrics = None
        with open(log_path, 'w', encoding='utf-8') as log:
            proc = subprocess.Popen(
                cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True,
                encoding='utf-8', errors='replace')
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                match = _METRIC_RE.search(line)
                if match:
                    last_metrics = {
                        'time': float(match.group(1)),
                        'memory': int(match.group(2)),
                        'grad_norm': float(match.group(3)),
                        'loss': float(match.group(4)),
                    }
            code = proc.wait()
        summary[name] = dict(
            status='ok' if code == 0 else f'failed({code})',
            metrics=last_metrics)
        print(f'[{name}] {summary[name]["status"]} {last_metrics}', flush=True)
        (work_root / 'summary.json').write_text(
            json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()