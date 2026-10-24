"""Tiny-subset overfit sanity check (not a result, never for the paper).

Verifies the whole training path on the *current* data + protocol: if the
model cannot drive the loss down and snap its boxes onto the GT of 4 tiles,
something in data -> model -> loss -> backward is wrong.
"""
_base_ = ['./m2_pf_rpn_cloud_full.py']

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=False,
    dataset=dict(ann_file='manifests/overfit_train.json'))

optim_wrapper = dict(optimizer=dict(lr=0.0001))

max_epochs = 120
param_scheduler = [
    dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True,
         milestones=[100], gamma=0.1),
]
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1000)
work_dir = '/tmp/overfit_sanity'
randomness = dict(seed=0)