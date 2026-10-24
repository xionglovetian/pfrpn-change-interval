_base_ = ['./m2_pf_rpn_multimodal.py']

data_root = 'data/landslide_multimodal_smoke/'

model = dict(
    encoder=dict(num_cp=0),
    data_preprocessor=dict(
        mean=[
            775.293807, 887.323120, 672.731295,
            -9.083241, -15.453249, -7.738177, -13.610027,
            1404.920732, 18.666528, 134.025566,
            0.000107, 1.758700, -0.080701, 0.073693, -0.031000,
        ],
        std=[
            1179.581708, 1171.757970, 1265.930407,
            2.636395, 2.876795, 2.512657, 2.553452,
            261.651525, 13.643280, 107.975366,
            0.027156, 1.786912, 0.235295, 0.201009, 0.158928,
        ]))

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='manifests/train.json'))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='manifests/val.json'))
test_dataloader = val_dataloader

work_dir = 'work_dirs/m2_pf_rpn/sen12landslides_smoke'