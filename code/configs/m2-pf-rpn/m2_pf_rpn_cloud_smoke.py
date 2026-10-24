_base_ = ['./m2_pf_rpn_gpu_smoke.py']

# Cloud smoke: 223-patch local subset (official random patch-level split).
# Engineering smoke only -- never a paper result.
data_root = 'data/landslide_multimodal/'

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='manifests/train.json'))

work_dir = 'work_dirs/m2_pf_rpn/cloud_smoke'