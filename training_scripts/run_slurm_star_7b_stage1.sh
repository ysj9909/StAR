#!/bin/bash
#SBATCH --job-name=star-7b-s1
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --partition=YOUR_PARTITION
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --error=outputs/slurm-%j.err

# --- User Settings ---
port=6379
verl_workdir=YOUR_WORKDIR_PATH  # e.g., /path/to/StAR
# module load CUDA/12.6  # Uncomment and adjust if needed
# export WANDB_API_KEY=your_wandb_api_key  # Optional: for wandb logging
# ------------------------

# 0) Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$(printf "0"; for ((j=1; j<SLURM_GPUS_PER_NODE; j++)); do printf ",%d" "$j"; done)

# 1) Conda activation
source $(conda info --base)/etc/profile.d/conda.sh
conda activate star_qwen2_5

# 2) Move to workdir
cd "$verl_workdir"

# 3) Collect hostnames
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node=${nodes[0]}

# 4) Head node IP
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
if [[ "$head_node_ip" == *" "* ]]; then
  IFS=' ' read -ra A <<<"$head_node_ip"
  [[ ${#A[0]} -gt 16 ]] && head_node_ip=${A[1]} || head_node_ip=${A[0]}
fi

export ip_head="${head_node_ip}:${port}"
echo "Ray HEAD -> ${head_node} @ ${ip_head}"

# 5) Start Head
head_tmp="/tmp/ray_${USER}_${SLURM_JOB_ID}_head"
mkdir -p "$head_tmp"

srun --export=ALL --nodes=1 --ntasks=1 --gpus-per-task="${SLURM_GPUS_PER_NODE}" --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
     -w "$head_node" \
  ray start --head \
            --node-ip-address="$head_node_ip" \
            --port=$port \
            --num-cpus="${SLURM_CPUS_PER_TASK}" \
            --num-gpus="${SLURM_GPUS_PER_NODE}" \
            --temp-dir="$head_tmp" \
            --block &

sleep 10

# 6) Start Workers
worker_num=$(( SLURM_NNODES - 1 ))
for ((i=1; i<=worker_num; i++)); do
  node_i=${nodes[$i]}
  worker_tmp="/tmp/ray_${USER}_${SLURM_JOB_ID}_worker_${i}"
  mkdir -p "$worker_tmp"

  echo "Ray WORKER $i -> $node_i"
  srun --export=ALL --nodes=1 --ntasks=1 --gpus-per-task="${SLURM_GPUS_PER_NODE}" --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
       -w "$node_i" \
    ray start --address "$ip_head" \
              --num-cpus="${SLURM_CPUS_PER_TASK}" \
              --num-gpus="${SLURM_GPUS_PER_NODE}" \
              --temp-dir="$worker_tmp" \
              --block &
  sleep 5
done

MODEL_PATH=pretrained_models/Qwen2.5-VL-7B-Instruct
RUN_NAME=StAR_Qwen2.5VL7B_$(date +%Y%m%d_%H%M%S)

export RAY_ADDRESS=$ip_head
export MODEL_PATH RUN_NAME

# 7) Training
set -x
python3 -m verl.trainer.main \
    algorithm.adv_estimator=grpo \
    worker.actor.loss_mode=grpo \
    config=training_scripts/visionreasoner_7b.yaml \
    data.train_files=data/visionreasoner_multi_object_refcocolvis_masks_840 \
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
    worker.actor.use_kl_loss=false \
    worker.actor.kl_loss_coef=0 \
    worker.actor.entropy_coeff=0 \
    worker.actor.optim.lr=1.0e-5 \
    worker.actor.optim.weight_decay=0.001 \
    worker.actor.global_batch_size=16 \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=2 \
    worker.actor.offload.param_offload=false \
    worker.rollout.tensor_parallel_size=2 \
    worker.rollout.gpu_memory_utilization=0.6 \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.enforce_eager=false \
    worker.rollout.free_cache_engine=false \
    worker.rollout.n=16 \
    worker.rollout.layered_summon=false \
    worker.reward.compute_score=star \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=${SLURM_GPUS_PER_NODE} \
    trainer.nnodes=${SLURM_NNODES} \
    trainer.save_freq=100 \
    trainer.total_episodes=1 \
    trainer.save_checkpoint_path=star_workdir/${RUN_NAME} \
  2>&1 | tee ${RUN_NAME}.log
