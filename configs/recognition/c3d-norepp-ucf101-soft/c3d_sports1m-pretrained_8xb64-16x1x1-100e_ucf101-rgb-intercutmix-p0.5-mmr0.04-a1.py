_base_ = [
    '../../_base_/models/c3d_sports1m_pretrained.py',
    '../../_base_/default_runtime.py',
]
label_mix_alpha = 1
model = dict(cls_head=dict(type='I3DCutMixHead', label_mix_alpha=label_mix_alpha))

dataset_type = 'VideoDataset'
dataset = 'ucf101'
mix_mode = 'intercutmix'
detector = 'UniDet'
detection_conf = 0.5
min_mask_ratio = 0.04
mix_prob = 0.5
relevancy_model = 'all-mpnet-base-v2'
relevancy_thresh = 0.5
num_workers = 16
batch_size = 64
clip_len = 16

video_root = f'data/{dataset}/videos'
class_index = f'data/{dataset}/annotations/classInd.txt'
mix_video_dir = f'data/{dataset}/{detector}/{detection_conf}/{mix_mode}/mix-0/{relevancy_model}/{relevancy_thresh}'
video_root_val = video_root
split = 1  # official train/test splits. valid numbers: 1, 2, 3
ann_file_train = f'data/{dataset}/{dataset}_train_split_{split}_videos.txt'
ann_file_val = f'data/{dataset}/{dataset}_val_split_{split}_videos.txt'
ann_file_test = f'data/{dataset}/{dataset}_val_split_{split}_videos.txt'

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='InterCutMix', mix_video_dir=mix_video_dir, class_index=class_index, mix_prob=mix_prob, min_mask_ratio=min_mask_ratio),
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=clip_len, frame_interval=1, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 128)),
    dict(type='RandomCrop', size=112),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', algorithm_keys=['scene_label', 'mask_ratio']),
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames', clip_len=clip_len, frame_interval=1, num_clips=1, test_mode=True
    ),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 128)),
    dict(type='CenterCrop', crop_size=112),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs'),
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames', clip_len=clip_len, frame_interval=1, num_clips=10, test_mode=True
    ),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 128)),
    dict(type='CenterCrop', crop_size=112),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs'),
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
        pipeline=train_pipeline,
    ),
)
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
        test_mode=True,
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=video_root_val),
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=45,
        by_epoch=True,
        milestones=[20, 40],
        gamma=0.1,
    )
]

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005),
    clip_grad=dict(max_norm=40, norm_type=2),
)

default_hooks = dict(checkpoint=dict(interval=1000, max_keep_ckpts=1))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (30 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=240)
