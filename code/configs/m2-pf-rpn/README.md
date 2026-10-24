# M2-PF-RPN 配置说明

主配置：

```bash
python tools/train.py configs/m2-pf-rpn/m2_pf_rpn_multimodal.py
```

该配置对应 `RESEARCH_ROADMAP.md` 的 A3/A4/A5/A6 全量版本：

- 光学：`optical_pre`
- SAR：`sar_pre` + `sar_post`
- 地形：`topography`
- 时间变化：`change`

`modality_mask` 由数据加载器写入，并在融合与 pseudo-token router 中生效。
缺失模态使用零通道输入，mask 为 0，模型自动降级到可用模态。

## 消融开关

| 编号 | 目标 | 关键覆盖 |
|---|---|---|
| A0 | RGB PF-RPN baseline | 使用原 `configs/pf-rpn/pf-rpn_coco-imagenet.py` 或把 `modality_order` 设为 `('optical',)`，`model.use_sar=False`、`model.use_topography=False`、`model.use_change=False` |
| A1 | Optical + topography | `modality_order=('optical','topography')`，关闭 SAR/change |
| A2 | Optical + SAR | `modality_order=('optical','sar')`，关闭 topography/change |
| A3 | Optical + SAR + topography | `modality_order=('optical','sar','topography')` |
| A4 | A3 + temporal change | 使用默认四模态配置 |
| A5 | A4 + terrain-aware token | `model.use_modality_tokens=True` |
| A6 | A5 + cross-modal router | `model.use_cross_modal_router=True`, `model.use_router=True` |
| A7 | A6 去掉 router | `model.use_router=False` |
| A8 | A6 去掉 SAR | `model.use_sar=False`，并同步修改数据 pipeline |
| A9 | A6 去掉时间变化 | `model.use_change=False`，并同步修改数据 pipeline |

命令行覆盖示例：

```bash
python tools/train.py configs/m2-pf-rpn/m2_pf_rpn_multimodal.py \
  --cfg-options model.use_sar=False model.use_change=False \
  train_dataloader.dataset.pipeline.0.modality_paths.sar=None
```

实际消融必须同步更新：

1. `modality_order`
2. `modality_channels`
3. `model.fusion_cfg.in_channels`
4. `train_pipeline[0].modality_paths`
5. `train_pipeline[0].modality_channels`

建议由 M4 提供统一的配置生成脚本，避免手工覆盖造成输入通道不一致。

## Sen12Landslides 数据

数据集仓库：https://github.com/PaulH97/Sen12Landslides

数据下载到 `E:\Sen12Landslides\data` 后，使用转换器把官方
`splits_aligned.json` 和 NetCDF patch 转成 M1 输入：

```bash
python tools/data/prepare_sen12landslides.py \
  E:/Sen12Landslides \
  --task S12LS-LD \
  --version harmonized \
  --splits-file E:/Sen12Landslides/tasks/S12LS-LD/harmonized/splits_aligned.json \
  --out E:/PF-RPN/data/landslide_multimodal \
  --sar-source asc
```

转换器输出：

```text
E:/PF-RPN/data/landslide_multimodal/
├── events/<event_id>/tiles/<tile_id>/
│   ├── optical_pre.npy
│   ├── sar_pre.npy
│   ├── sar_post.npy
│   ├── topography.npy
│   ├── change.npy
│   └── mask.npy
└── manifests/
    ├── train.json
    ├── val.json
    └── test.json
```

M1 主配置指向 `data/landslide_multimodal`；小样本验证使用
`configs/m2-pf-rpn/m2_pf_rpn_sen12landslides_smoke.py`。

## 最小样例

```bash
python tools/make_minimal_m2_sample.py tests/data/m2_pf_rpn_smoke
python tools/inference/export_proposals.py \
  configs/m2-pf-rpn/m2_pf_rpn_minimal.py \
  checkpoints/m2_pf_rpn.pth \
  --out artifacts/proposals/test
```

该最小样例用于检查：

1. 多模态 `.npy` 通道拼接；
2. `modality_mask` 传递；
3. proposal.v1 JSON + NPZ 导出；
4. schema/特征引用一致性。

## 输出

推理导出：

```bash
python tools/inference/export_proposals.py \
  configs/m2-pf-rpn/m2_pf_rpn_multimodal.py \
  checkpoints/m2_pf_rpn.pth \
  --out artifacts/proposals/test
```

输出遵循 `RESEARCH_ROADMAP.md` 的 `proposal.v1` 约定：

```text
artifacts/proposals/<split>/<event_id>/<image_id>.proposals.json
artifacts/proposals/<split>/<event_id>/<image_id>.features.npz
```