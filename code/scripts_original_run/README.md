# scripts_original_run/ — 原样归档的运行脚本

⚠️ **这一目录不是复现入口。**

这里保存的是**实际产生论文结果时运行的原始脚本**（云端执行、原样留档），用途是**过程追溯证据**：
- 证明结果是怎么跑出来的
- 记录当时的队列编排、参数与看门狗逻辑

**它们内含云端的绝对路径**（`/root/autodl-tmp/...`），因此在别的机器上 **不能直接运行**。

## 复现请走这两处

| 目的 | 去处 |
|---|---|
| **端到端复现** | [`../REPRODUCE.md`](../REPRODUCE.md) —— 已参数化的可移植命令 |
| **核心代码**（路径已清理，可直接跑） | `../tools/`、`../configs/`、`../mmdet/`、`../tests/` |

## 核心代码 vs 归档脚本

| | 路径清洁度 | 用途 |
|---|---|---|
| `tools/ configs/ mmdet/ tests/ schemas/` | ✅ 0 处硬编码路径 | **可移植、可复现** |
| `scripts_original_run/`（本目录） | ⚠️ 含云端绝对路径 | **仅作追溯，勿直接运行** |

## 最有参考价值的归档脚本

`loeo_run.sh`、`loro_run.sh`、`a3_all_ablation.sh`、`post_extra_evals.sh`、
`cloud_pipeline3.sh`、`cloud_adneg.sh`、`cloud_download*.sh`、
`make_loeo_yolo_folds.py`、`make_loro_folds.py`、`pheno_folds.py`、
`run_grounding_dino.py`、`run_yolo_world.py`、`train_yolo_loeo.py`。

> `REPRODUCE.md` 中的命令即由这些脚本整理、参数化而来。
