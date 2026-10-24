# PF-RPN for Landslide Detection — Reproducibility Package

*Prompt-free region proposals for landslide candidate discovery: cross-event and cross-region validation.*

本包包含论文的全部**代码、配置、数据划分定义、结果指标与日志**，用于复现论文中的所有数字。

---

## 快速导航

| 想做什么 | 看这里 |
|---|---|
| **端到端复现全部实验** | [`REPRODUCE.md`](REPRODUCE.md) |
| 核对论文某个表的数字 | [结果文件索引](#结果文件索引) |
| 校验本包未被篡改 | [`CHECKSUMS.sha256`](CHECKSUMS.sha256) |
| 环境与版本 | [`env/`](env) |

---

## 目录结构

```
PF-RPN/                         ← 代码仓库根（所有相对路径以此为准）
├─ mmdet/                       自定义检测模块（PF-RPN 适配）
├─ tools/
│  ├─ data/                     数据构建：NetCDF→npy、事件划分、正负合并、统计
│  ├─ run_cloud_ablations.py    A0–A9 训练驱动
│  ├─ eval_m1_split.py          指标评测（AR@k / 分尺寸 / FPR 等）
│  └─ eval_proposal_metrics.py  外部基线的指标评测
├─ configs/                     训练/评测配置
├─ scripts/                     队列与基线脚本（见下）
├─ tests/                       自检测试
├─ data/                        ★ 划分定义（manifests + norm_stats）
├─ artifacts/                   ★ 主要结果（含逐图 dump）
├─ loeo/  loro/                 ★ 跨事件 / 跨区域结果
├─ baselines/                   ★ 外部基线结果 + 物候分析
├─ docs/                        方法与结果文档
├─ logs/                        运行日志（过程可追溯）
├─ requirements/  setup.py      依赖声明
env/
├─ pip_freeze.txt               精确 pip 环境
├─ conda_env.yml                conda 环境
├─ versions.txt                 关键版本
└─ gpu_info.txt                 硬件信息
CHECKSUMS.sha256                全量 SHA256 校验清单（1199 文件）
```

**`scripts/` 中的关键脚本**
| 脚本 | 作用 |
|---|---|
| `loeo_run.sh` / `loro_run.sh` | LOEO(7 折) / LORO(3 折) 完整流程 |
| `a3_all_ablation.sh` | 机理对照 A3（去变化）10 折 |
| `make_loeo_yolo_folds.py` / `make_loro_folds.py` | 折划分生成 |
| `pheno_folds.py` | 逐折物候/可分性分析 |
| `run_grounding_dino.py` / `run_yolo_world.py` | 开放词汇基线 |
| `train_yolo_loeo.py` / `train_yolov8s_control.py` / `train_yoloworld.py` | 闭集/微调基线 |
| `cloud_download*.sh` | 原始数据下载 |

---

## 结果文件索引

| 论文位置 | 文件 |
|---|---|
| **Table 2** 主消融 A0–A9 | `PF-RPN/artifacts/hn_lb_metrics/<RUN>/{test,val}.json` |
| **Table 3** 跨事件（LOEO 7 折）| `PF-RPN/loeo/A4_loeo_<fold>.json` + `y8_loeo_<fold>_metrics.json` |
| **Table 3** 跨区域（LORO 3 折）| `PF-RPN/loro/A4_loro_<region>.json` + `y8_loro_<region>_metrics.json` |
| **Table 4** 外部基线 | `PF-RPN/baselines/*_metrics.json` |
| **Table 5** 多种子（A3–A7）| `PF-RPN/artifacts/hn_lb_seed{123,777}_metrics/<RUN>/test.json` |
| **Fig 5** 机理（A3 vs 可分性）| `PF-RPN/loeo/A3_loeo_*.json` + `PF-RPN/loro/A3_loro_*.json` + `PF-RPN/baselines/analysis/phenology_folds.json` |
| 配对统计逐图 dump | `PF-RPN/artifacts/hn_lb_paired/<RUN>/*.npz` |
| 缺失模态 / 正负对照 | `PF-RPN/artifacts/{od_lb_metrics,p_lb_metrics}/` |

---

## 关键口径（复现必须一致）

| 项 | 取值 |
|---|---|
| 变化通道 | **长基线 (0,−1)**：每瓦片首帧→末帧，与标注无关（避免捷径） |
| SAR | 升轨（`--sar-source asc`） |
| 归一化统计 | **只用 train** |
| 训练 | 320×320 / batch 8 / lr 2e-5 / 8 epoch / AMP(loss_scale=1.0) |
| 评测 | IoU 0.5 的 COCO 风格 AR@k（k=100/300/900） |
| 划分 | 事件级；LOEO 留一事件(7)；LORO 留一区域(3)，测试事件/区域在训练集中完全缺席 |

---

## 主要结论（供对照）

| 指标 | 数值 |
|---|---|
| 跨事件（LOEO 7 折）PF-RPN vs YOLOv8s | AR@900 **0.6713 ± 0.1593** vs **0.0030 ± 0.0015**（**≈224×**）|
| 跨区域（LORO 3 折）| AR@900 **0.7341 ± 0.0417** vs **0.0027 ± 0.0010**（**≈270×**）|
| 主消融 | A0 0.3533 → A4 0.7167（四模态 **2.03×**）；变化模态 A3→A4 **+0.2485**（3 seed，q=0.00075）|
| 机理（10 折）| 变化模态增益倍数 1.02×–1.93×（均值 1.36×），与像素可分性 **Pearson r = +0.722** |
| 外部基线（去重 5 档）| 最强 YOLO-World 微调 0.0899 |

---

## 已知边界（论文须如实声明）

- 跨事件/跨区域设定**只对比 YOLOv8s 单基线**（强基线仅主 split）
- 跨设定为**单 seed**；主 split 为 3 seed
- 多种子覆盖 **A3–A7（5 个配置）**，A0/A9 无多种子
- 外部基线**去重后 5 档**：`yolov8s_ft_test ≡ y8_50e_test`；`yoloworld_ft_test ≡ yw_50e_test`（同一运行的两种文件名）
- YOLO-World 零样本在 conf=0.01 下 0 候选（该模型在此数据上的最高置信度仅 3.6e-4）；放宽至 conf=1e-6 后 AR@900 = 0.0131 —— **引用时须说明阈值口径**
- 打分动态范围极小 → R@kFPPI 口径失效，故只用 AR@k

---

## 校验

```bash
sha256sum -c CHECKSUMS.sha256
```
无输出即全部通过。
