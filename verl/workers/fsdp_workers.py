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
The main entry point to run the PPO algorithm
"""

from typing import Literal
import os
import re
import json
import logging
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.distributed as dist
from accelerate import init_empty_weights
from codetiming import Timer
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    GenerationConfig,
    PreTrainedModel,
)
from transformers.modeling_utils import no_init_weights

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import get_tokenizer, get_processor
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_fn,
    load_fsdp_model,
    load_fsdp_optimizer,
    offload_fsdp_model,
    offload_fsdp_optimizer,
    fsdp_version,
    init_fn,
    layered_summon_lora_params,
)
from verl.utils.model_utils import print_model_size
from verl.utils.model import update_model_config
from verl.utils.performance import log_gpu_memory_usage
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import get_constant_schedule_with_warmup
from verl.workers.actor import DataParallelPPOActor
from verl.workers.config import FSDPConfig, ModelConfig, OptimConfig, WorkerConfig
from verl.workers.critic import DataParallelPPOCritic
from verl.workers.rollout.vllm_rollout import vLLMRollout, vLLMSCRollout
from verl.workers.sharding_manager import FSDPVLLMShardingManager
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

import peft
from peft import TaskType
from verl.peft.get_peft_model import get_peft_model
from verl.peft.tuners.config import LoraConfig
from verl.peft.tuners.layer import Linear as CustomLinear
# from peft.tuners.lora import LoraModel
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.device import get_device_id, get_device_name, get_nccl_backend, get_torch_device, is_cuda_available, is_npu_available
from safetensors.torch import load_file, save_file 

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def process_state_dict(state_dict):
    """
    Modify LoRA weight keys to match the model's expected format.
    Specifically, add '.default' to keys like 'lora_A.weight' or 'lora_B.weight'.
    """
    keys_to_modify = [k for k in state_dict.keys() if re.search(r"\.lora_[AB]\.weight$", k)]
    for old_key in keys_to_modify:
        new_key = old_key.replace(".weight", ".default.weight")
        state_dict[new_key] = state_dict.pop(old_key)


class FSDPWorker(Worker):
    def __init__(
        self,
        config: WorkerConfig,
        role: Literal["actor", "critic", "rollout", "ref", "actor_rollout", "actor_rollout_ref"],
    ):
        super().__init__()
        self.config = config

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        # build device mesh for FSDP
        # TODO: support FSDP hybrid shard for larger model
        world_size = dist.get_world_size()
        self.device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"])

        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_sequence_parallel_size = self.config.actor.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(world_size // self.ulysses_sequence_parallel_size, self.ulysses_sequence_parallel_size),
                mesh_dim_names=["dp", "sp"],
            )
        else:
            self.ulysses_device_mesh = None

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self.lora_type = self.config.actor.model.lora_type
        self._lora_rank = self.config.actor.model.lora_rank
        self._is_lora = self._lora_rank > 0

        self.role = role
        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_critic = self.role == "critic"
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        self._use_param_offload = False
        self._use_optimizer_offload = False
        if self._is_actor:
            self._use_param_offload = self.config.actor.offload.param_offload
            self._use_optimizer_offload = self.config.actor.offload.optimizer_offload
        elif self._is_critic:
            self._use_param_offload = self.config.critic.offload.param_offload
            self._use_optimizer_offload = self.config.critic.offload.optimizer_offload
        elif self._is_ref:
            # NOTE: it seems that manual offload is slowly than FSDP offload
            self._use_param_offload = self.config.ref.offload.param_offload

        # normalize config
        if self._is_actor:
            if self.config.rollout.m > 0:
                self.config.actor.global_batch_size *= self.config.rollout.m
            else:
                self.config.actor.global_batch_size *= self.config.rollout.n
            self.config.actor.global_batch_size_per_device = (
                self.config.actor.global_batch_size // self.device_mesh.shape[0] * self.ulysses_sequence_parallel_size
            )
            assert (
                self.config.actor.global_batch_size_per_device
                % self.config.actor.micro_batch_size_per_device_for_update
                == 0
            )
        elif self._is_critic:
            self.config.critic.global_batch_size *= self.config.rollout.n
            self.config.critic.global_batch_size_per_device = (
                self.config.critic.global_batch_size // self.device_mesh.shape[0] * self.ulysses_sequence_parallel_size
            )
            assert (
                self.config.critic.global_batch_size_per_device
                % self.config.critic.micro_batch_size_per_device_for_update
                == 0
            )

    def _build_model_optimizer(
        self,
        model_config: ModelConfig,
        fsdp_config: FSDPConfig,
        optim_config: OptimConfig,
        # override_model_config,
        padding_free: bool = False,
    ) -> None:
        self.tokenizer = get_tokenizer(model_config.tokenizer_path, trust_remote_code=model_config.trust_remote_code)
        self.processor = get_processor(model_config.tokenizer_path)
        self.model_config = AutoConfig.from_pretrained(
            model_config.model_path,
            trust_remote_code=model_config.trust_remote_code,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_config.override_config,
        )

        try:
            self.generation_config = GenerationConfig.from_pretrained(model_config.model_path)
        except Exception:
            self.generation_config = GenerationConfig.from_model_config(self.model_config)

        self.print_rank0(f"Model config: {self.model_config}")

        if padding_free:
            raise NotImplementedError("Padding free is not implemented yet.")

        if fsdp_config.torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor or self._is_critic else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(fsdp_config.torch_dtype)

        actor_model_config = AutoConfig.from_pretrained(
            model_config.model_path,
            trust_remote_code=model_config.trust_remote_code,
            attn_implementation="flash_attention_2"
        )
        
        # patch for kimi-vl
        # if getattr(actor_model_config, "model_type", None) == "kimi_vl":
        #     actor_model_config.text_config.topk_method = "greedy"
        
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        # override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        self.actor_model_config = actor_model_config
        
        if self._is_critic:
            auto_class = AutoModelForTokenClassification
        elif type(self.model_config) in AutoModelForVision2Seq._model_mapping.keys():
        # elif type(self.model_config) in AutoModelForImageTextToText._model_mapping.keys():
            auto_class = AutoModelForVision2Seq
            # auto_class = AutoModelForImageTextToText
        else:
            auto_class = AutoModelForCausalLM

        if self.rank == 0:
            model = auto_class.from_pretrained(
                model_config.model_path,
                config=self.model_config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=model_config.trust_remote_code,
            )
        else:
            with no_init_weights(), init_empty_weights():
                model = auto_class.from_config(
                    self.model_config,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=model_config.trust_remote_code,
                )

        assert isinstance(model, PreTrainedModel)  # lint
        model.tie_weights()  # avoid hanging
        model = model.to(torch_dtype)
        if model_config.enable_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if self._is_lora:
            def filter_lora_target_modules(model, exclude_keywords=["visual", "lm_head"]):
                """
                Return a list of linear module names that do NOT contain excluded keywords.
                """
                target_modules = []

                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        if not any(keyword in name.lower() for keyword in exclude_keywords):
                            target_modules.append(name)

                return target_modules
            
            if self.config.actor.model.target_modules == "all-linear":
                filtered_modules = filter_lora_target_modules(model, exclude_keywords=["visual", "lm_head"])
            else:
                filtered_modules = self.config.actor.model.target_modules  # already list
            
            self.print_rank0("Applying LoRA to actor module")
            model.enable_input_require_grads()
            # Convert config to regular Python types before creating PEFT model
            lora_config = {"task_type": TaskType.CAUSAL_LM,
                           "r": self.config.actor.model.lora_rank,
                           "lora_alpha": self.config.actor.model.lora_alpha,
                           "target_modules": convert_to_regular_types(filtered_modules),
                           "exclude_modules": convert_to_regular_types(self.config.actor.model.exclude_modules),
                           "bias": "none"}
            
            adapter_type = self.config.actor.model.lora_type.lower()
            if adapter_type == "lora":
                model = get_peft_model(model, LoraConfig(**lora_config))
            elif adapter_type == "hira":
                lora_config["is_lora_augmented"] = self.config.actor.model.is_lora_augmented
                if self.config.actor.model.is_lora_augmented:
                    # lora_config["r"] = lora_config["r"] - self.config.actor.model.lora_rank2
                    lora_config["r2"] = self.config.actor.model.lora_rank2
                
                lora_config = LoraConfig(use_hira=True, **lora_config)
                # lora_config._register_custom_module({nn.Linear: CustomLinear})
                model = get_peft_model(model, lora_config)
            elif adapter_type == "shira":
                lora_config["shira_sparsity"] = self.config.actor.model.sparsity_level
                lora_config["mask_multiplier"] = self.config.actor.model.mask_multiplier
                
                lora_config = LoraConfig(use_shira=True, **lora_config)
                # lora_config._register_custom_module({nn.Linear: CustomLinear})
                model = get_peft_model(model, lora_config)
            self.print_rank0(f"Applied LoRA variant '{adapter_type}' with config: {lora_config}")
            
            if self.config.actor.lora_checkpoint_path is not None:
                state_dict = load_file(f"{self.config.actor.lora_checkpoint_path}/adapter_model.safetensors")
                process_state_dict(state_dict)
                load_result = model.load_state_dict(state_dict, strict=False)
                self.print_rank0("Missing keys:", load_result.missing_keys)
                self.print_rank0("Unexpected keys:", load_result.unexpected_keys)
            
            model = model.to(dtype=PrecisionType.to_dtype(fsdp_config.mp_param_dtype))
            
            total_params = 0
            trainable_params = 0

            self.print_rank0(f"{'Parameter Name':<60} {'Dtype':<15} {'Trainable':<10} {'#Params':>10}")
            self.print_rank0("=" * 100)

            for n, p in model.named_parameters():
                numel = p.numel()
                total_params += numel
                if p.requires_grad:
                    trainable_params += numel
                self.print_rank0(f"{n:<60} {str(p.dtype):<15} {str(p.requires_grad):<10} {numel:>10,}")

            self.print_rank0("=" * 100)
            self.print_rank0(f"Total parameters:     {total_params:,}")
            self.print_rank0(f"Trainable parameters: {trainable_params:,}")

        dist.barrier()
        if self.rank == 0:
            print_model_size(model)

        log_gpu_memory_usage("After init from huggingface AutoModel")
        mixed_precision = MixedPrecision(
            param_dtype=PrecisionType.to_dtype(fsdp_config.mp_param_dtype),
            reduce_dtype=PrecisionType.to_dtype(fsdp_config.mp_reduce_dtype),
            buffer_dtype=PrecisionType.to_dtype(fsdp_config.mp_buffer_dtype),
        )
        # auto_wrap_policy = get_fsdp_wrap_policy(model)
        auto_wrap_policy = get_fsdp_wrap_policy(module=model, config=None, is_lora=self._is_lora)
        
        if fsdp_config.enable_full_shard:
            sharding_strategy = ShardingStrategy.FULL_SHARD
        else:
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP

        if fsdp_config.param_offload or fsdp_config.optimizer_offload:
            cpu_offload = CPUOffload(offload_params=fsdp_config.param_offload)
        else:
            cpu_offload = None

        if self.rank == 0:
            print(f"FSDP wrap policy: {auto_wrap_policy}.")

        self.fsdp_module = FSDP(
            model,
            cpu_offload=cpu_offload,
            param_init_fn=init_fn,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            # param_init_fn=get_init_fn(model, device="cuda") if self.rank != 0 else None,
            sync_module_states=True,
            device_mesh=self.device_mesh,
            use_orig_params=True,
            forward_prefetch=False,
        )
        log_gpu_memory_usage("After Actor FSDP init")

        print(dict(self.fsdp_module._fsdp_wrapped_module.named_parameters()).keys())

        if self._is_actor or self._is_critic:
            self.optimizer = torch.optim.AdamW(
                self.fsdp_module.parameters(),
                lr=optim_config.lr,
                betas=optim_config.betas,
                weight_decay=optim_config.weight_decay,
            )
            num_warmup_steps = int(optim_config.lr_warmup_steps_ratio * optim_config.training_steps)
            self.lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps
            )
        else:
            self.optimizer, self.lr_scheduler = None, None

        log_gpu_memory_usage("After actor optimizer init")

    def _build_rollout(self) -> None:
        from torch.distributed.device_mesh import init_device_mesh
        
        # TODO(sgm): support FSDP hybrid shard for larger model
        tp_size = self.config.rollout.tensor_parallel_size
        dp_size = self.world_size // tp_size
        assert self.world_size % tp_size == 0, (
            f"rollout world_size: {self.world_size} is not divisible by tp_size: {tp_size}"
        )
        rollout_device_mesh = init_device_mesh("cuda", mesh_shape=(dp_size, tp_size), mesh_dim_names=["dp", "tp"])
        log_gpu_memory_usage("Before building vllm rollout")
        lora_kwargs = {"lora_kwargs": {"enable_lora": True, "max_loras": 1, "max_lora_rank": self._lora_rank}} if self._is_lora else {}
        
        if self.config.rollout.use_self_correction:
            Rollout = vLLMSCRollout
        else:
            Rollout = vLLMRollout
        
        self.rollout = Rollout(
            model_path=self.config.actor.model.model_path,
            config=self.config.rollout,
            tokenizer=self.tokenizer,
            # model_hf_config=self.actor_model_config,
            # device_mesh=rollout_device_mesh,
            # trust_remote_code=self.model_config.trust_remote_code,
            **lora_kwargs
        )
        
        log_gpu_memory_usage("After building vllm rollout")
        
        if "Qwen3VLProcessor" in self.processor.__class__.__name__:
            is_qwen3 = True
        else:
            is_qwen3 = False
        
        full_params = torch.distributed.get_world_size() == 1
        self.rollout_sharding_manager = FSDPVLLMShardingManager(
            module=self.fsdp_module,
            inference_engine=self.rollout.inference_engine,
            # model_config=self.actor_model_config,
            rollout_config=self.config.rollout,
            full_params=full_params,
            device_mesh=rollout_device_mesh,
            offload_param=self._use_param_offload,
            load_format=self.config.rollout.load_format,
            layered_summon=self.config.rollout.layered_summon,
            is_qwen3=is_qwen3,
        )
        log_gpu_memory_usage("After building sharding manager")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self._is_critic:
            model_config = self.config.critic.model
            fsdp_config = self.config.critic.fsdp
            optim_config = self.config.critic.optim
            padding_free = self.config.critic.padding_free
        else:
            model_config = self.config.actor.model
            fsdp_config = self.config.actor.fsdp
            optim_config = self.config.actor.optim
            padding_free = self.config.actor.padding_free

        if self._is_actor or self._is_critic or self._is_ref:
            self._build_model_optimizer(
                model_config=model_config,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                padding_free=padding_free,
            )
            # get the original unwrapped module
            self.unwrapped_model = self.fsdp_module._fsdp_wrapped_module
            if self._use_optimizer_offload and not self._is_critic:
                offload_fsdp_optimizer(optimizer=self.optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init")

        if self._is_actor:
            self.actor = DataParallelPPOActor(
                config=self.config.actor,
                actor_module=self.fsdp_module,
                actor_optimizer=self.optimizer,
            )

        if self._is_critic:
            self.critic = DataParallelPPOCritic(
                config=self.config,
                critic_module=self.fsdp_module,
                critic_optimizer=self.optimizer,
            )

        if self._is_rollout:
            self._build_rollout()

        if self._is_ref:
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.fsdp_module)

        if self._is_actor or self._is_critic:
            self.flops_counter = FlopsCounter(self.model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.fsdp_module,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                tokenizer=self.tokenizer,
                processor=self.processor
            )

        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path: str, global_step: int = 0, remove_previous_ckpt: bool = False):
        from verl.utils.logger import log_with_rank
        assert self._is_actor or self._is_critic
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path,
            global_step=global_step,
            remove_previous_ckpt=remove_previous_ckpt,
        )
        dist.barrier()
        
        if self._is_lora and hasattr(getattr(self, "actor_module", self.fsdp_module), "peft_config"):
            lora_save_path = os.path.join(local_path, "lora_adapter")
            peft_model = getattr(self, "actor_module", self.fsdp_module)
            peft_config = {}
            
            # model.named_modules()
            # lora_params_names = [name for name in state_dict.keys() if "lora_" in name]
            
            if dist.get_rank() == 0:
                os.makedirs(lora_save_path, exist_ok=True)
                peft_config = asdict(peft_model.peft_config.get("default", {}))
                peft_config["task_type"] = peft_config["task_type"].value
                peft_config["peft_type"] = peft_config["peft_type"].value
                peft_config["target_modules"] = list(peft_config["target_modules"])
                peft_config["exclude_modules"] = ["visual", "lm_head"]
            try:
                if fsdp_version(self.fsdp_module) > 0:
                    self.fsdp_module = self.fsdp_module.to(get_device_name())
                    lora_params = layered_summon_lora_params(self.fsdp_module)
                    if dist.get_rank() == 0:
                        save_file(lora_params, os.path.join(lora_save_path, "adapter_model.safetensors"))
                        with open(os.path.join(lora_save_path, "adapter_config.json"), "w", encoding="utf-8") as f:
                            json.dump(peft_config, f, ensure_ascii=False, indent=4)
            except Exception as e:
                log_with_rank(f"Save LoRA Adapter Error ({e})", rank=dist.get_rank(), logger=logger, log_only_rank_0=True)

            dist.barrier()
            log_with_rank(f"[rank-{self.rank}]: Saved LoRA adapter to: {lora_save_path}", rank=dist.get_rank(), logger=logger, log_only_rank_0=True)
        
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path: str, del_local_after_load: bool = True):
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        self.checkpoint_manager.load_checkpoint(path=path, del_local_after_load=del_local_after_load)
        dist.barrier()
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

    """ActorRolloutRefWorker"""

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        assert self._is_actor

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)

        log_gpu_memory_usage("Before update policy")
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)

            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["mfu/actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size

            lr = self.lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr
            self.lr_scheduler.step()
            log_gpu_memory_usage("After update policy")

            # TODO: here, we should return all metrics
            output = DataProto(meta_info={"metrics": metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to("cpu")

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)

        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        assert self._is_rollout

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        with self.rollout_sharding_manager:
            # after parameters sync with rollout, offload actor model to CPU
            if self._use_param_offload:
                offload_fsdp_model(self.fsdp_module)

            if self._use_optimizer_offload:
                offload_fsdp_optimizer(optimizer=self.optimizer)

            log_gpu_memory_usage("After entering rollout sharding manager")

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)
            log_gpu_memory_usage("After rollout generation")

            output = self.rollout_sharding_manager.postprocess_data(output)

        output = output.to("cpu")
        torch.cuda.empty_cache()  # clear kv cache
        log_gpu_memory_usage("After recompute log prob")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto):
        assert self._is_actor
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)
        
        # Support all hardwares
        from contextlib import nullcontext
        
        is_lora = data.meta_info.get("is_lora", False)
        # adapter_ctx = self.ref_policy.actor_module.disable_adapter() if is_lora else nullcontext()
        adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()
        
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["temperature"] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            with adapter_ctx:
                output = self.actor.compute_log_prob(data=data)
                # if is_lora:
                #     output = self.ref_policy.compute_log_prob(data=data)
                # else:
                #     output = self.actor.compute_log_prob(data=data)
            output = DataProto.from_dict(
                tensors={"old_log_probs": output}, meta_info={"temperature": self.config.rollout.temperature}
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.fsdp_module._handle.reshard(True)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        torch.cuda.empty_cache()
        log_gpu_memory_usage("After compute_log_prob")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        if self._is_lora:
            # if _is_lora, actor without lora applied is the ref
            data.meta_info["is_lora"] = True
            self._is_actor = True
            data = self.compute_log_prob(data)
            # this old_log_probs is in fact ref_log_prob
            data = DataProto.from_dict(tensors={"ref_log_prob": data.batch["old_log_probs"]})
            return data
        assert self._is_ref
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        data.meta_info["temperature"] = self.config.rollout.temperature
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.ref_policy.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={"ref_log_prob": output})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.fsdp_module._handle.reshard(True)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        torch.cuda.empty_cache()
        log_gpu_memory_usage("After compute_ref_log_prob")
        return output

    """CriticWorker"""

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        assert self._is_critic
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={"values": values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        output = output.to("cpu")
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)

            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["mfu/critic"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size

            self.lr_scheduler.step()
            lr = self.lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr

            output = DataProto(batch=None, meta_info={"metrics": metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        output = output.to("cpu")
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)

        torch.cuda.empty_cache()
        return output
