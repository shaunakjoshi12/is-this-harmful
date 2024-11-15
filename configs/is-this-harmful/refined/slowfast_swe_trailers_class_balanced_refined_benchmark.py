label_as_distribution = False

model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowFast',
        pretrained=None,
        resample_rate=4,  # tau
        speed_ratio=4,  # alpha
        channel_ratio=8,  # beta_inv
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=True,
            fusion_kernel=7,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            norm_eval=False),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            norm_eval=False)),
    cls_head=dict(
        type='SlowFastHead',
        in_channels=2304,  # 2048+256
        num_classes=2,
        spatial_type='avg',
        dropout_ratio=0.5,
        label_as_distribution=label_as_distribution,
        loss_cls=dict(type='CrossEntropyLoss')))
train_cfg = None
test_cfg = dict(average_clips='prob')
dataset_type = 'SweTrailersBenchmarkDataset'
data_root = '/home/shaunaks/Vidharm/data/clips_processed'
data_root_val = '/home/shaunaks/Vidharm/data/clips_processed'
#data_root_test = 'data/swe_trailers/data'
ann_file_train = '/home/shaunaks/Vidharm/data/annotations/annotated_dataset_train.json'
ann_file_val = '/home/shaunaks/Vidharm/data/annotations/annotated_dataset_val.json'
#ann_file_test = 'data/swe_trailers/refined_test.json'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=7, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=7,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=7,
        num_clips=2,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
#    dict(type='ThreeCrop', crop_size=256),
#    dict(type='Flip', flip_ratio=0),

data = dict(
    videos_per_gpu=7,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
        label_as_distribution=label_as_distribution,
        sample_by_class=True),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        label_as_distribution=label_as_distribution,
        sample_by_class=True)
)
# optimizer
optimizer = dict(
    type='SGD', lr=0.001, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 1 gpu
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-3)
total_epochs = 30
checkpoint_config = dict(interval=1)
workflow = [('train', 1),('val', 1)]#
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy','mean_average_precision',
            'mmit_mean_average_precision'])
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
exp_name = 'slowfast_swe_trailers_class_balanced_refined_benchmark'
work_dir = './work_dirs/'+exp_name
load_from="https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb/slowfast_r50_8x8x1_256e_kinetics400_rgb_20200716-73547d2b.pth"
resume_from = None
find_unused_parameters = False
