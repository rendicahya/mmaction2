model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='C3D',
        pretrained=
        'https://download.openmmlab.com/mmaction/recognition/c3d/c3d_sports1m_pretrain_20201016-dcc47ddc.pth',
        style='pytorch',
        conv_cfg=dict(type='Conv3d'),
        norm_cfg=None,
        act_cfg=dict(type='ReLU'),
        dropout_ratio=0.5,
        init_std=0.005),
    cls_head=dict(
        type='I3DHead',
        num_classes=101,
        in_channels=4096,
        spatial_type=None,
        dropout_ratio=0.5,
        init_std=0.01,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[104, 117, 128],
        std=[1, 1, 1],
        format_shape='NCTHW'),
    train_cfg=None,
    test_cfg=None)
default_scope = 'mmaction'
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[dict(type='LocalVisBackend')])
log_level = 'INFO'
load_from = None
resume = False
dataset_type = 'VideoDataset'
data_root = 'data/ucf101/videos'
data_root_val = 'data/ucf101/videos'
split = 1
ann_file_train = 'data/ucf101/ucf101_train_split_1_videos.txt'
ann_file_val = 'data/ucf101/ucf101_val_split_1_videos.txt'
ann_file_test = 'data/ucf101/ucf101_val_split_1_videos.txt'
file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='DecordInit', io_backend='disk'),
    dict(type='SampleFrames', clip_len=16, frame_interval=1, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 128)),
    dict(type='RandomCrop', size=112),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', io_backend='disk'),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 128)),
    dict(type='CenterCrop', crop_size=112),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', io_backend='disk'),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=1,
        num_clips=10,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 128)),
    dict(type='CenterCrop', crop_size=112),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
train_dataloader = dict(
    batch_size=30,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='VideoDataset',
        ann_file='data/ucf101/ucf101_train_split_1_videos.txt',
        data_prefix=dict(video='data/ucf101/videos'),
        pipeline=[
            dict(type='DecordInit', io_backend='disk'),
            dict(
                type='SampleFrames',
                clip_len=16,
                frame_interval=1,
                num_clips=1),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 128)),
            dict(type='RandomCrop', size=112),
            dict(type='Flip', flip_ratio=0.5),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='PackActionInputs')
        ]))
val_dataloader = dict(
    batch_size=30,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='VideoDataset',
        ann_file='data/ucf101/ucf101_val_split_1_videos.txt',
        data_prefix=dict(video='data/ucf101/videos'),
        pipeline=[
            dict(type='DecordInit', io_backend='disk'),
            dict(
                type='SampleFrames',
                clip_len=16,
                frame_interval=1,
                num_clips=1,
                test_mode=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 128)),
            dict(type='CenterCrop', crop_size=112),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='PackActionInputs')
        ],
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='VideoDataset',
        ann_file='data/ucf101/ucf101_val_split_1_videos.txt',
        data_prefix=dict(video='data/ucf101/videos'),
        pipeline=[
            dict(type='DecordInit', io_backend='disk'),
            dict(
                type='SampleFrames',
                clip_len=16,
                frame_interval=1,
                num_clips=10,
                test_mode=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 128)),
            dict(type='CenterCrop', crop_size=112),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='PackActionInputs')
        ],
        test_mode=True))
val_evaluator = dict(type='AccMetric')
test_evaluator = dict(type='AccMetric')
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=45, val_begin=1, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=45,
        by_epoch=True,
        milestones=[20, 40],
        gamma=0.1)
]
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005),
    clip_grad=dict(max_norm=40, norm_type=2))
auto_scale_lr = dict(enable=False, base_batch_size=240)
