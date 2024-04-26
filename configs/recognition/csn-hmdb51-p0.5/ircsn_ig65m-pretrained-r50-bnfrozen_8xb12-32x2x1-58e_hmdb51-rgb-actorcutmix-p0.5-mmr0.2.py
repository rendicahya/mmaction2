_base_ = [
    '../../_base_/models/ircsn_r152.py', '../../_base_/default_runtime.py'
]

# model settings
model = dict(
    backbone=dict(
        depth=50,
        norm_eval=True,
        bn_frozen=True,
        pretrained='https://download.openmmlab.com/mmaction/recognition/csn/'
        'ircsn_from_scratch_r50_ig65m_20210617-ce545a37.pth'))

dataset_type = 'VideoDataset'
dataset = 'hmdb51'
mix_mode = 'actorcutmix'
detector = 'UniDet'
min_mask_ratio = 0.2
mix_prob = 0.5
num_workers = 16

data_root = 'data/hmdb51/videos'
video_dir = f'data/{dataset}/{detector}/select/{mix_mode}/REPP/mix-0'
data_root_val = f'data/{dataset}/videos'
ann_file_train = f'data/{dataset}/train.txt'
ann_file_val = f'data/{dataset}/test.txt'
ann_file_test = f'data/{dataset}/test.txt'

# file_client_args = dict(
#      io_backend='petrel',
#      path_mapping=dict(
#          {'data/kinetics400': 's3://openmmlab/datasets/action/Kinetics400'}))
file_client_args = dict(io_backend='disk')

train_pipeline = [
    dict(type='ActorCutMix', video_dir=video_dir, mix_prob=0.5, min_mask_ratio=min_mask_ratio),
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=10,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=12,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=58, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=16),
    dict(
        type='MultiStepLR',
        begin=0,
        end=58,
        by_epoch=True,
        milestones=[32, 48],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=5e-4, momentum=0.9, weight_decay=1e-4),
    clip_grad=dict(max_norm=40, norm_type=2))

default_hooks = dict(checkpoint=dict(interval=1000, max_keep_ckpts=1))
find_unused_parameters = True

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (12 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=96)
