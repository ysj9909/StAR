# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import json
import os
import ray
import getpass
from omegaconf import OmegaConf

from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.config import PPOConfig
# from verl.trainer.sp_ray_trainer import SPRayPPOTrainer
from verl.trainer.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from verl.utils import get_processor, get_tokenizer
from verl.workers.fsdp_workers import FSDPWorker
from verl.workers.sam_worker import SAMPredictorWorker


def main():
    cli_args = OmegaConf.from_cli()
    file_config = OmegaConf.load(cli_args.config)
    del cli_args.config

    default_config = OmegaConf.structured(PPOConfig())
    ppo_config = OmegaConf.merge(default_config, file_config, cli_args)
    ppo_config = OmegaConf.to_object(ppo_config)

    if not ray.is_initialized():
        # this is for local ray cluster
        # temp_ray_dir = f"/tmp/ray_{getpass.getuser()}_{os.getpid()}"
        # os.makedirs(temp_ray_dir, exist_ok=True)
        ray.init(
            # _temp_dir=temp_ray_dir,
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}}
            )

    ray.get(main_task.remote(ppo_config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
def main_task(config: PPOConfig):
    config.deep_post_init()
    print(json.dumps(config.to_dict(), indent=2))
    # instantiate tokenizer
    tokenizer = get_tokenizer(config.worker.actor.model.model_path)
    processor = get_processor(config.worker.actor.model.model_path, use_fast=True)

    # define worker classes
    ray_worker_group_cls = RayWorkerGroup
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(FSDPWorker),
        Role.Critic: ray.remote(FSDPWorker),
        Role.SAM: ray.remote(SAMPredictorWorker),
    }
    if config.worker.actor.kl_loss_coef > 0:
        role_worker_mapping[Role.RefPolicy] = ray.remote(FSDPWorker)

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }
    
    if "Qwen3VLProcessor" in processor.__class__.__name__:
        is_qwen3 = True
        print("Reward design has been adapted to match Qwen3VL's coordinate prediction scheme.")
    else:
        is_qwen3 = False
        print("Reward design unchanged.")
    
    from verl.workers.reward import CustomRewardManager
    reward_fn = CustomRewardManager(
        tokenizer=tokenizer,
        num_examine=3,
        compute_score=config.worker.reward.compute_score,
        rollout_n=config.worker.rollout.n,
        sam_actor=None,
        is_qwen3=is_qwen3,
        self_correction_bonus_alpha=config.worker.reward.bonus_alpha,
    )
    
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=None,
    )
        
    trainer.init_workers()
    
    if config.worker.reward.compute_score.startswith("star"):
        reward_fn.sam_actor = trainer.sam_wg
    
    trainer.fit()


if __name__ == "__main__":
    main()
