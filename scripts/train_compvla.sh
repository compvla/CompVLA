export TOKENIZERS_PARALLELISM=false
# Uncomment if you hit CUDA OOM from memory fragmentation
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

TASK=Erase

python train.py \
    --config_path=configs/config_compvla.yaml \
    --dataset.repo_id=${TASK} \
    --dataset.root=datasets/lerobot/${TASK} \
    --policy.freeze_vision_encoder=false \
    --policy.train_model_list="[true, true, true]" \
    --policy.use_joint_in_training=true \
    --policy.use_compliance_expert=true \
    --wandb.enable=false \
    --output_dir=train_results/compvla/${TASK} \
    --job_name=compvla_${TASK}
