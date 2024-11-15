CUDA_VISIBLE_DEVICES=0 python ../tools/val_evaluate.py ../configs/is-this-harmful/refined/fusion/fuse_MLP_1.py \
work_dirs/refined_fusion_mlp_slowfast_epoch_8/epoch_9.pth \
--eval mean_class_accuracy top_k_accuracy micro_f1_score --out