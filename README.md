# pfrpn-change-interval

Reproduction package for the manuscript

> **Train on the Event, Deploy on the Baseline: The Change Interval Belongs to the Task Specification, not the Model**
> Luoyang Xiong, Juan Meng\*, Nan Deng, Wenchao Li, Xu Jin, Xiaotian Dong, Yuchai Zhang
> University of Emergency Management, Sanhe 065201, Hebei, China  ·  \*corresponding author

This repository contains the code, data definitions, per-image scores and result artefacts behind every number in the
paper and its supplementary material. It does **not** contain model checkpoints (33 GB) or the Sen12Landslides imagery
(57 GB); both are described below and can be regenerated.

---

## 1. What the paper claims, and where the evidence is

| Claim | Value | Evidence in this repository |
|---|---|---|
| The change interval is fixed by the calendar | 97.9 % exact, 100.0 % within one acquisition, 3,135 tiles / 11 inventories | `results/json/` (`eval_calendar_*`), `data/dates.jsonl`, `scripts/f1_calendar_rule.py` |
| The interval cannot be corrected after training | frozen long-baseline model loses 0.087 AR@900 when given the correct interval | `results/json/frozen_pair_3080ti.json`, `frozen_plus1_3080ti.json` |
| Committing the interval during training is free at deployment | +5.4 % relative AR@900 (0.7094 → 0.7474), three seeds | `results/json/paired_ar.json`, `results/per_image/*.npz` |
| The gain does not replicate across protocols | Indonesia LOEO −0.017, Italy LOEO −0.067 (pre-registered KILL) | `results/json/loeo_ind_*.json`, `B_italy_*.json`, `B_italy_decision.json` |
| A training-free criterion forecasts which route wins | Pearson 0.915, leave-one-fold-out 0.856, 10/10 folds correct sign | `data/features_15f.jsonl`, `scripts/f1_calendar_rule.py`, `results/json/paired_ar_loeo.json` |
| The paradigm gap is bounded once budgets are equalised | 2.13× (four-event split, image-averaged AR@900 at a matched 900-candidate budget), 3.11× (ten folds) | `results/run_configs/*/test_metrics.json` (PF-RPN, macro *and* micro); the external-arm scores (FT-YOLO-World 0.3321 AR@900 / 0.2900 ground-truth-weighted, Grounding DINO 0.1828 / 0.159451) are reported in Section V-A of the manuscript |

---

## 2. Repository map

```
code/                 PF-RPN adaptation (MMDetection 3.3.0 derived): models, configs, tools, tests
data/                 derived-data definitions and per-tile acquisition calendar
  derived_data_definitions.tar.gz    8 arm definitions (config.py + manifests/{train,val,test}.json + norm_stats.json)
  _manifests_base.tgz                base manifests (80 entries)
  dates.jsonl                        per-tile acquisition calendar (used by the calendar rule)
  features_15f.jsonl                 per-tile separability features (criterion inputs)
  mechanics.json, stats_variants.json, repro_check.json
results/
  json/                every metric in the paper (paired bootstrap inputs, fold-level metrics, decisions)
  per_image/           per-image score arrays (.npz, 14 files) - needed to recompute paired CIs
  logs/                training and evaluation logs, plus the exact chain scripts that produced each arm
  run_configs/         per-run config.py + test_metrics.json
scripts/              data builders, calendar rule, criterion (LOFO/baselines), paired bootstrap, fold diagnostics
figures/              paper figures (PNG) and the plotting scripts that generate them
docs/                 compiled manuscript and supplementary material (PDF)
CHECKSUMS.sha256      SHA256 of every tracked file
verify_checksums.py   one-command integrity check
```

---

## 3. Environment

* Python 3.10, PyTorch 2.1.2 + CUDA 12.1, MMCV 2.1.0, MMEngine 0.10.7, MMDetection 3.3.0, NumPy 1.26.4
* A single 24 GB GPU (RTX 4090 / 3080 Ti class) was used for all runs; evaluation-only work runs on CPU
* Data: **Sen12Landslides** (https://huggingface.co/datasets/paulhoehn/Sen12Landslides), converted to `.npy` tiles with
  `code/tools/data/prepare_sen12landslides.py`

```bash
pip install -r code/requirements.txt
pip install -e code          # installs the PF-RPN adaptation as a package
```

---

## 4. Reproducing the numbers

### 4.1 Integrity (seconds, no GPU)

```bash
python verify_checksums.py            # every file matches CHECKSUMS.sha256
```

### 4.2 Paired confidence intervals (minutes, no GPU)

`results/per_image/*.npz` hold the per-image scores of the anchor and of every intervention arm.
`scripts/paired_bootstrap.py` recomputes the paired ΔAR confidence intervals and the sign test:

```bash
python scripts/paired_bootstrap.py --json results/json/paired_ar.json --reps 20000
```

Two estimators are reported in the paper and both are reproducible here: the three-seed average over the
1,759 test images (Table IV) and the paired bootstrap over the 673 images that carry ground truth (§V-D).

### 4.3 The training-free criterion (minutes, no GPU)

```bash
python scripts/f1_calendar_rule.py  --dates data/dates.jsonl --manifests data/_manifests_base.tgz   # 97.9 % / 100.0 %
python scripts/criterion_lofo.py    --features data/features_15f.jsonl                             # r = 0.915, LOO 0.856, 10/10 folds
python scripts/criterion_baselines.py --features data/features_15f.jsonl                           # pointwise and zero-shot baselines
```

`scripts/f1_baselines.py` holds the frame-selector baselines that establish the calendar rule's headroom.

### 4.4 Rebuilding a change-interval arm (needs the dataset)

```bash
# 1) rebuild the arm definition from the calendar
python scripts/build_train_root.py --calendar data/dates.jsonl --out data/data_plus1
# 2) train with the arm config (one 24 GB GPU, ~4 h)
python -m tools.train configs/pf_rpn_plus1.py
# 3) score through the shared export pipeline
python -m tools.eval_export --config data/data_plus1/config.py --checkpoint <ckpt> --out results/json/plus1.json
```

The exact commands that produced the published runs are preserved verbatim in `results/logs/chain*.sh`.

### 4.5 The calendar rule (long form)

```bash
python scripts/f1_calendar_rule.py --dates data/dates.jsonl --manifests data/_manifests_base.tgz
# reproduces 97.9 % exact / 100.0 % within one acquisition over 3,135 tiles
```

---

## 5. Data and checkpoints

| Resource | Size | Where |
|---|---|---|
| Sen12Landslides raw | 57 GB | public, HuggingFace (`paulhoehn/Sen12Landslides`) |
| Derived `.npy` tiles | ~10 GB | rebuild with `code/tools/data/prepare_sen12landslides.py` |
| Model checkpoints (21 baseline + 14 interval-arm runs) | 82 GB | not distributed; rebuilt by the training commands in §4.3 |

No checkpoint is required to verify or recompute any reported metric: every metric is a function of the stored
per-image scores.

---

## 6. Scope and honesty notes

* The interval effect is **scoped**: +5.4 % relative holds on the event-level protocol with an anchor of 0.7094.
  The two cross-protocol folds (Indonesia LOEO, Italy LOEO) did not replicate the gain and are reported as such,
  including the pre-registered kill criterion (`results/json/B_italy_decision.json`).
* The cross-paradigm ratios are quoted on **image-averaged** AR@900 with every arm at a matched 900-candidate budget.
  The ground-truth-weighted (micro) counterparts are recorded in the manuscript tables; mixing the two conventions is
  what produced the earlier 1.99× figure (0.6619 micro ÷ 0.3321 macro), which has been withdrawn in favour of 2.13×
  (0.7059 macro ÷ 0.3321 macro) — the same convention as the 3.11× ten-fold figure.
* The per-image export JSONs of the external detector arms (Grounding DINO, YOLO-World fine-tuned and zero-shot,
  closed-set YOLOv8s) are **not** part of this package, so those arms can be checked against the manuscript's Table I/II
  but not recomputed from this repository alone. Every PF-RPN number, by contrast, recomputes from the shipped artefacts.
* The verification stage described in Section V-E is a **comparison instrument**, not a deployable quality model:
  the paradigm-neutral scorer reaches 3.31 % recall at ten candidates against 22.65 % for a model's own confidence.
* `results/json/tmp_plus1long.json` is an intermediate diagnostic kept for transparency; it is not used in the paper.

---

## 7. License and citation

* Code: MIT (`LICENSE`). Data definitions, results and text: CC BY 4.0.
* Citation metadata: `CITATION.cff`. Until the paper is published, please cite the repository URL.

## 8. Contact

Issues and questions: please open an issue in this repository. Corresponding author: Juan Meng (mengjuan@cidp.edu.cn).
