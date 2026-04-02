export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# [Optional] To use Weights & Biases logging, run `wandb login` first.
# To disable wandb logging, uncomment the line below:
# export WANDB_MODE=disabled

set -x

MODEL_PATH=pretrained_models/Qwen3-VL-8B-Instruct
RUN_NAME=StAR_Qwen3VL8B_$(date +%Y%m%d_%H%M%S)

# Set the path to your Stage 1 LoRA checkpoint
STAGE1_LORA_CHECKPOINT=YOUR_STAGE1_CHECKPOINT_PATH  # e.g., star_workdir/StAR_Qwen3VL8B_.../global_step_XXX/actor/lora_adapter

python3 -m verl.trainer.main \
    algorithm.adv_estimator=grpo \
    worker.actor.loss_mode=grpo \
    config=training_scripts/visionreasoner_7b.yaml \
    data.train_files=data/ReasonSegX_train \
    data.seed=42 \
    data.val_files=None \
    data.rollout_batch_size=16 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.lora_type=lora \
    worker.actor.model.lora_rank=64 \
    worker.actor.model.lora_alpha=64 \
    worker.actor.model.target_modules=all-linear \
    worker.actor.model.exclude_modules='.*visual.*' \
    worker.actor.model.enable_gradient_checkpointing=true \
    worker.actor.lora_checkpoint_path=${STAGE1_LORA_CHECKPOINT} \
    worker.actor.use_kl_loss=false \
    worker.actor.kl_loss_coef=0 \
    worker.actor.entropy_coeff=0 \
    worker.actor.optim.lr=5.0e-6 \
    worker.actor.optim.weight_decay=0.001 \
    worker.actor.global_batch_size=16 \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=2 \
    worker.actor.offload.param_offload=false \
    worker.rollout.use_self_correction=false \
    worker.rollout.tensor_parallel_size=2 \
    worker.rollout.gpu_memory_utilization=0.45 \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.enforce_eager=false \
    worker.rollout.free_cache_engine=false \
    worker.rollout.n=64 \
    worker.rollout.m=16 \
    worker.rollout.layered_summon=false \
    worker.reward.compute_score=star_s2 \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.total_episodes=10 \
    trainer.save_checkpoint_path=star_workdir/${RUN_NAME}
