_base_ = [
    '../../_base_/models/tsn_r50.py', '../../_base_/schedules/sgd_100e.py',
    '../../_base_/default_runtime.py'
]

dataset_type = 'VideoDataset'
dataset = 'ucf101'
mix_mode = 'actorcutmix'
detector = 'UniDet'
min_mask_ratio = 0.05
mix_prob = 0.5
num_workers = 16
batch_size = 64

video_root = f'data/{dataset}/videos'
class_index = f'data/{dataset}/annotations/classInd.txt'
mix_video_dir = f'data/{dataset}/{detector}/{detection_conf}/{mix_mode}/REPP/mix-0'
video_root_val = video_root
split = 1  # official train/test splits. valid numbers: 1, 2, 3
ann_file_train = f'data/{dataset}/{dataset}_train_split_{split}_videos.txt'
ann_file_val = f'data/{dataset}/{dataset}_val_split_{split}_videos.txt'
ann_file_test = f'data/{dataset}/{dataset}_val_split_{split}_videos.txt'

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='ActorCutMix', mix_video_dir=mix_video_dir, class_index=class_index, mix_prob=mix_prob, min_mask_ratio=min_mask_ratio),
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=3),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=3,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=25,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='TenCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=video_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=video_root_val),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=video_root_val),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

default_hooks = dict(checkpoint=dict(interval=1000, max_keep_ckpts=1))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (32 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=256)

# set training batch size to 4
train_dataloader['batch_size'] = 4

# Save checkpoints every epoch, and only keep the latest checkpoint
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1000, max_keep_ckpts=1))
# Set the maximum number of epochs to 10, and validate the model every 1 epochs
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=64, val_begin=1, val_interval=1)
# adjust learning rate schedule according to 10 epochs
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,
        by_epoch=True,
        milestones=[20, 40],
        gamma=0.1)
]

model = dict(
    cls_head=dict(num_classes=101))
load_from = 'https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_20220906-cd10898e.pth'