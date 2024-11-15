label_as_distribution = False
# model settings
model = dict(
    type='AudioRecognizer',
    backbone=dict(type='ResNet', depth=18, in_channels=1, norm_eval=False),
    cls_head=dict(
        type='AudioTSNHead',
        num_classes=2,
        in_channels=512,
        dropout_ratio=0.5,
        init_std=0.01,
        label_as_distribution=label_as_distribution,
    loss_cls=dict(type='CrossEntropyLoss')))

# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips='prob')
# dataset settings
dataset_type = 'SweTrailersBenchmarkDataset'
data_root = '/home/shaunaks/Vidharm/data/clips_processed'
data_root_val = '/home/shaunaks/Vidharm/data/clips_processed'
#data_root_test = 'data/swe_trailers/data'
ann_file_train = '/home/shaunaks/Vidharm/data/annotations/annotated_dataset_train.json'
ann_file_val = '/home/shaunaks/Vidharm/data/annotations/annotated_dataset_val.json'
train_pipeline = [
    dict(type='LoadAudioFeature'),
    dict(type='SampleFrames', clip_len=128, frame_interval=1, num_clips=1),
    dict(type='AudioFeatureSelector',fixed_length=300),
    dict(type='AudioNormalize'),
    dict(type='SpecAugment'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='Collect', keys=['audios', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['audios'])
]
val_pipeline = [
    dict(type='LoadAudioFeature'),
    dict(
        type='SampleFrames',
        clip_len=200,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='AudioFeatureSelector',fixed_length=500),
    dict(type='AudioNormalize'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='Collect', keys=['audios', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['audios'])
]
test_pipeline = [
    dict(type='LoadAudioFeature'),
    dict(
        type='SampleFrames',
        clip_len=240,
        frame_interval=1,
        num_clips=2,
        test_mode=True),
    dict(type='AudioFeatureSelector',fixed_length=600),
    dict(type='AudioNormalize'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='Collect', keys=['audios', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['audios'])
]
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
        sample_by_class=True))
# optimizer
optimizer = dict(
    type='SGD', lr=0.001, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=1e-3)
total_epochs = 30
checkpoint_config = dict(interval=1)
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy','mean_average_precision',
            'mmit_mean_average_precision'])
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
exp_name = 'tsn_r18_swe_trailers_audio_feature_class_balanced_refined_benchmark'
work_dir = './work_dirs/'+exp_name
load_from = "https://download.openmmlab.com/mmaction/recognition/audio_recognition/tsn_r18_64x1x1_100e_kinetics400_audio_feature/tsn_r18_64x1x1_100e_kinetics400_audio_feature_20201012-bf34df6c.pth"
resume_from = None
workflow = [('train', 1),('val',1)]
