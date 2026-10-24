# pfrpn-change-interval

Reproduction package for the manuscript

> **Train on the Event, Deploy on the Baseline: Decoupling the Change Interval in Landslide Candidate Discovery**
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
| The paradigm gap is bounded once budgets are equalised | 1.99× (four-event split), 3.11× (ten folds) | `results/json/paired_ar.json`, `results/run_configs/*/test_metrics.json` |

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
scripts/              data builders, calendar rule, paired bootstrap, fold diagnostics
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
`scripts/paired_bootstrap.py` recomputes the ΔAR confidence intervals reported in the paper:

```bash
python scripts/paired_bootstrap.py --json results/json/paired_ar.json --reps 20000
```

The printed values reproduce Table IV and the interval section of the paper.

### 4.3 Rebuilding a change-interval arm (needs the dataset)

```bash
# 1) rebuild the arm definition from the calendar
python scripts/build_train_root.py --calendar data/dates.jsonl --out data/data_plus1
# 2) train with the arm config (one 24 GB GPU, ~4 h)
python -m tools.train configs/pf_rpn_plus1.py
# 3) score through the shared export pipeline
python -m tools.eval_export --config data/data_plus1/config.py --checkpoint <ckpt> --out results/json/plus1.json
```

The exact commands that produced the published runs are preserved verbatim in `results/logs/chain*.sh`.

### 4.4 The calendar rule

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
* The verification stage described in Section V-E is a **comparison instrument**, not a deployable quality model:
  the paradigm-neutral scorer reaches 3.31 % recall at ten candidates against 22.65 % for a model's own confidence.
* `results/json/tmp_plus1long.json` is an intermediate diagnostic kept for transparency; it is not used in the paper.

---

## 7. License and citation

* Code: MIT (`LICENSE`). Data definitions, results and text: CC BY 4.0.
* Citation metadata: `CITATION.cff`. Until the paper is published, please cite the repository URL.

## 8. Contact

Issues and questions: please open an issue in this repository. Corresponding author: Juan Meng (mengjuan@cidp.edu.cn).
