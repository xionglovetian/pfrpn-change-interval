# Copyright (c) OpenMMLab. All rights reserved.
"""Router usage diagnostic for the A6 ablation (roadmap 19.11).

``cross-modal router`` shows no measurable AR gain.  Before writing that up as
a finding we have to separate

* a genuine null ("the router learns nothing useful here"), from
* a false negative ("the router was never trained effectively").

This tool runs a trained checkpoint over a small slice of a split and reports
how the router actually behaves:

* per-expert usage (softmax of the shared ``nn.Linear(dim, 1)`` probe),
* the normalised entropy of that usage (1.0 == perfectly uniform == no
  specialisation),
* the top-k selection histogram,
* the router weight statistics.

Requires a GPU, so it is queued to run after the training chain.  Run it on a
config with ``use_router=True`` (A6), not on A4/A5/A7.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate

import mmdet.datasets  # noqa: F401
import mmdet.models  # noqa: F401
from mmdet.apis import init_detector
from mmdet.registry import DATASETS
from mmdet.utils import register_all_modules

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_m1_split import _build_eval_dataset_cfg, _parse_cfg_options  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config')
    p.add_argument('checkpoint')
    p.add_argument('--split', default='val')
    p.add_argument('--scale', type=int, default=320)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--max-images', type=int, default=96)
    p.add_argument('--out', default=None)
    p.add_argument('--cfg-options', nargs='*', default=[])
    return p.parse_args()


def main():
    args = parse_args()
    register_all_modules()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(_parse_cfg_options(args.cfg_options))
    dataset_cfg = dict(_build_eval_dataset_cfg(cfg, args.split, args.scale))
    # init_detector() in this fork reads config.test_dataloader.dataset to build
    # its own test pipeline, but the cloud configs leave test_dataloader=None.
    # eval_m1_split.py patches this before init_detector; mirror that here.
    cfg.test_dataloader = dict(dataset=dataset_cfg)
    dataset = DATASETS.build(dataset_cfg)
    model = init_detector(cfg, args.checkpoint, device=args.device)
    model.eval()

    router = getattr(model, 'router', None)
    if router is None:
        raise SystemExit('model has no .router attribute')

    captured = {'logits': []}

    def hook(_module, _inp, out):
        captured['logits'].append(out.detach().float().cpu())

    handle = router.register_forward_hook(hook)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=pseudo_collate)

    n_images = 0
    sel_hist = None
    with torch.no_grad():
        for batch in loader:
            data = model.data_preprocessor(batch, training=False)
            model.predict(data['inputs'], data['data_samples'], rescale=False)
            idx = getattr(router, '_last_idx_selected', None)
            if idx is not None:
                flat = idx.detach().cpu().numpy().reshape(-1)
                if sel_hist is None:
                    sel_hist = np.zeros(int(flat.max()) + 1, dtype=np.int64)
                for v in flat:
                    if v >= sel_hist.size:
                        sel_hist = np.pad(sel_hist, (0, v - sel_hist.size + 1))
                    sel_hist[v] += 1
            n_images += len(data['data_samples'])
            if n_images >= args.max_images:
                break
    handle.remove()
    if not captured['logits']:
        raise SystemExit('router was never called -- is use_router=True?')

    logits = torch.cat(captured['logits'], dim=0).squeeze(-1)  # [N, E]
    num_experts = int(logits.shape[1])
    usage = torch.softmax(logits, dim=1).mean(dim=0).numpy()
    ent = float(-(usage * np.log(usage + 1e-12)).sum())
    max_ent = float(np.log(num_experts))

    modalities = list(getattr(model, 'modality_order', []))
    num_mod = len(modalities)
    num_level = max(num_experts - num_mod, 0)
    names = [f'level_{i}' for i in range(num_level)] + list(modalities)
    names = (names + [f'expert_{i}' for i in range(num_experts)])[:num_experts]

    weight = router.weight.detach().float().cpu().numpy()
    result = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'split': args.split,
        'num_images_diagnosed': n_images,
        'num_experts': num_experts,
        'num_level_experts': num_level,
        'num_modality_experts': num_mod,
        'topk': int(getattr(model, 'topk', 0)),
        'usage': {names[i]: float(usage[i]) for i in range(num_experts)},
        'usage_normalized_entropy': float(ent / max_ent) if max_ent > 0 else 0.0,
        'usage_max_over_uniform': float(usage.max() * num_experts),
        'topk_selection_counts': (sel_hist.tolist() if sel_hist is not None
                                  else None),
        'router_weight_mean': float(weight.mean()),
        'router_weight_std': float(weight.std()),
        'router_weight_norm': float(np.linalg.norm(weight)),
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text, encoding='utf-8')
    print(text)
    print()
    print(f'experts={num_experts} (levels={num_level}, modalities={num_mod}), '
          f'topk={result["topk"]}')
    for i in range(num_experts):
        print(f'  {names[i]:>12}  usage={usage[i]:.4f}  '
              f'x_uniform={usage[i] * num_experts:.2f}')
    print(f'  normalised entropy = {result["usage_normalized_entropy"]:.4f} '
          f'(1.0 = uniform / no specialisation)')


if __name__ == '__main__':
    main()
