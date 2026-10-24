_base_ = ['./m2_pf_rpn_multimodal.py']

data_root = 'tests/data/m2_pf_rpn_smoke/'

train_dataloader = dict(
    dataset=dict(data_root=data_root, ann_file='manifest.json'))
val_dataloader = dict(
    dataset=dict(data_root=data_root, ann_file='manifest.json'))
test_dataloader = val_dataloader