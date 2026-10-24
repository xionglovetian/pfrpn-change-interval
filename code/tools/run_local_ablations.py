# Copyright (c) OpenMMLab. All rights reserved.
"""Run local M1 ablation dry-runs on GPU.

These runs use the official random patch split and are only a pipeline-level
dry-run.  They are not paper results; final experiments require M0 event-level
or region-level splits.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs' / 'm2-pf-rpn' / 'm2_pf_rpn_local_ablation_base.py'
WORK_ROOT = ROOT / 'work_dirs' / 'm2_pf_rpn' / 'ablations'

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

_METRIC_RE = re.compile(r'Epoch\(train\).*loss: ([0-9.eE+-]+).*'
                        r'grad_norm: ([0-9.eE+-]+).*'
                        r'memory: ([0-9]+)')


def _environment():
    env = os.environ.copy()
    prefix = (
        r'C:\Users\ASUS\.conda\envs\pf-rpn;'
        r'C:\Users\ASUS\.conda\envs\pf-rpn\Library\mingw-w64\bin;'
        r'C:\Users\ASUS\.conda\envs\pf-rpn\Library\usr\bin;'
        r'C:\Users\ASUS\.conda\envs\pf-rpn\Library\bin;'
        r'C:\Users\ASUS\.conda\envs\pf-rpn\Scripts;')
    env['PATH'] = prefix + env.get('PATH', '')
    env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    return env


def main():
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    selected = set(sys.argv[1:])
    summary = {}
    for name, overrides in ABLATIONS.items():
        if selected and name not in selected:
            continue
        work_dir = WORK_ROOT / name
        log_path = work_dir / 'train.log'
        work_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(ROOT / 'tools' / 'train.py'),
            str(CONFIG),
            '--work-dir', str(work_dir),
            '--cfg-options', *overrides,
        ]
        print(f'[{name}] start', flush=True)
        last_metrics = None
        with open(log_path, 'w', encoding='utf-8') as log:
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                env=_environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace')
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                match = _METRIC_RE.search(line)
                if match:
                    last_metrics = {
                        'loss': float(match.group(1)),
                        'grad_norm': float(match.group(2)),
                        'memory': int(match.group(3)),
                    }
            code = proc.wait()
        if code != 0:
            raise RuntimeError(f'{name} failed with exit code {code}; '
                               f'see {log_path}')
        summary[name] = last_metrics or {}
        print(f'[{name}] done {summary[name]}', flush=True)

    summary_path = WORK_ROOT / 'summary.json'
    summary_path.write_text(
        json.dumps(summary, indent=2), encoding='utf-8')
    print(f'summary written to {summary_path}', flush=True)


if __name__ == '__main__':
    main()