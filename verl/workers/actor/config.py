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
Actor config
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union, List


@dataclass
class ModelConfig:
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    override_config: Dict[str, Any] = field(default_factory=dict)
    enable_gradient_checkpointing: bool = True
    trust_remote_code: bool = True
    
    lora_type: str = "lora"
    sparsity_level: float = 1.0
    mask_multiplier: float = 0.0
    lora_rank: int = 0
    lora_alpha: int = 0
    target_modules: str = "all-linear"
    exclude_modules: str = '.*visual.*'
    
    is_lora_augmented: bool = False
    lora_rank2: int = 0

    def post_init(self):
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path


@dataclass
class SelfPlayConfig:
    use_selfplay: bool = False
    train_proposer: bool = True
    num_pos_refs: int = 10
    acc_gate_threshold: float = 0.5
    n_learnability_samples: int = 8
    pred_data_mix_strategy: str = "max_new"
    save_proposer_outputs: bool = False
    auto_mask_cfg = {}
    candidate_mask_source: str = "dataset"
    mask_dup_iou_thr: float = 0.9
    nested_contain_thr: float = 0.95
    selection_max_k: int = 100
    require_both_selection: bool = True
    max_task_pool: int = 200
    proposer_update_after_step: int = 30
    proposer_update_every: int = 5
    use_learnability_adv_weight: bool = False
    update_iteration: int = 1
    fixed_pos_refs_by_type = {
        "function_purpose" : [
            "Which part of the bicycle allows the rider to change gears and adapt to different terrains?",
            "You plan to record the professor’s voice from the back row. Which device would you set up?",
            "Highlight the object that stores power for portable use.",
            "If the projector screen in this classroom isn’t working, please identify a flat, light-colored surface we can use instead.",
            "Find something that could serve as a temporary seat.",
        ],
        "commonsense" : [
            "Which animal in this image is nocturnal?",
            "Locate the area where mail would be delivered in this scene.",
            "Someone got a small cut. Which product would you apply first to clean it before bandaging?",
            "In this construction site, identify materials that require special disposal due to environmental regulations.",
            "Find the parts of this computer that contain rare earth metals and would be valuable for recycling.",
        ],
        "comparative_relational" : [
            "Which fruit looks the most ripe compared to the rest?",
            "Find the container that is filled to a different level than the others.",
            "Segment two tools with matching heads but different handles.",
            "Find the fastest-moving object.",
            "Please locate the group of people who are interacting most actively in this party scene.",
            "Find the person who is being congratulated by others.",
            "Point out what the photographer is focusing on.",
            "Identify the vehicle that is obstructing the crosswalk, not just parked beside it.",
            "Segment the referee signaling a foul and the player being penalized.",
            "Please identify what caused the floor to get wet.",
        ],
        "compositional" : [
            "Locate something that can hold liquid or store food but is not made of plastic.",
            "Identify people who are wearing glasses and holding a phone.",
            "Segment all the animals that have stripes OR spots.",
            "Find the student who is neither writing nor looking at the board.",
            "Cats have excellent night vision. Identify the part that reflects light and enables this ability.",
        ],
    }

@dataclass
class OptimConfig:
    lr: float = 1e-6
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-2
    lr_warmup_steps_ratio: float = 0.0
    min_lr_ratio: Optional[float] = None
    warmup_style: str = "constant"
    """auto keys"""
    training_steps: int = field(default=-1, init=False)


@dataclass
class FSDPConfig:
    enable_full_shard: bool = True
    param_offload: bool = False
    optimizer_offload: bool = False
    torch_dtype: Optional[str] = None
    mp_param_dtype: str = "bf16"
    mp_reduce_dtype: str = "fp32"
    mp_buffer_dtype: str = "fp32"


@dataclass
class OffloadConfig:
    param_offload: bool = False
    optimizer_offload: bool = False


@dataclass
class ActorConfig:
    strategy: str = "fsdp"
    loss_mode: str = "grpo"
    global_batch_size: int = 256
    micro_batch_size_per_device_for_update: int = field(default=-1, init=False)
    micro_batch_size_per_device_for_experience: int = field(default=-1, init=False)
    lora_checkpoint_path: Optional[str] = None
    max_grad_norm: float = 1.0
    clip_ratio: float = 0.2
    entropy_coeff: float = 1e-3
    use_kl_loss: bool = True
    kl_loss_coef: float = 1e-3
    kl_loss_type: str = "low_var_kl"
    ppo_epochs: int = 1
    padding_free: bool = False
    ulysses_sequence_parallel_size: int = 1
    model: ModelConfig = field(default_factory=ModelConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    """auto keys"""
    global_batch_size_per_device: int = field(default=-1, init=False)

    def post_init(self):
        if self.ppo_epochs != 1:
            raise NotImplementedError


@dataclass
class RefConfig:
    strategy: str = "fsdp"
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    """auto keys"""
    micro_batch_size_per_device_for_experience: int = field(default=-1, init=False)
    padding_free: bool = field(default=False, init=False)
