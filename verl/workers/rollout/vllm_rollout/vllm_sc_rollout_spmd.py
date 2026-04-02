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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
"""

from contextlib import contextmanager
from typing import Any, List, Union
import json
import os
import re
import numpy as np

import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer
from vllm import LLM, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest
from PIL import Image, ImageDraw, ImageFont

from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length  #, get_eos_mask
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.config import RolloutConfig


def _repeat_interleave(features: Union[torch.Tensor, List[Any]], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(features, torch.Tensor):
        return features.repeat_interleave(repeats, dim=0)
    else:
        return [feature for feature in features for _ in range(repeats)]


class vLLMSCRollout(BaseRollout):
    def __init__(self,
                 model_path: str,
                 config: RolloutConfig,
                 tokenizer: PreTrainedTokenizer,
                #  model_hf_config,
                 **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        
        # self-correction (multi-turn) setting
        self.use_self_correction: bool = getattr(config, "use_self_correction", False)
        # only support 2-turn (1st: answer, 2nd: self-correction)
        self.num_turns: int = getattr(config, "num_turns", 2 if self.use_self_correction else 1)
        if self.use_self_correction and self.num_turns != 2:
            raise ValueError(
                f"When use_self_correction=True, current code only supports num_turns=2"
                f"(current num_turns: {self.num_turns})."
            )
        # “bbox” or “mask” 
        self.som_overlay_type: str = getattr(config, "som_overlay_type", "bbox")
        # Debug SoM image save path (If None, do not save)
        self.som_save_dir: str | None = getattr(config, "som_save_dir", None)
        self.som_image_dir: str | None = None
        if self.som_save_dir:
            os.makedirs(self.som_save_dir, exist_ok=True)
            self.som_image_dir = os.path.join(self.som_save_dir, "som_images")
            os.makedirs(self.som_image_dir, exist_ok=True)
        
        self.example_answer = "[{\"label\": \"chair\", \"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110]}, {\"label\": \"train track\", \"bbox_2d\": [225,296,706,786], \"point_2d\": [302,410]}]"
        
        self.sc_instruction_template = "<image>\n" \
                            "Your task is to find the target object(s) that match the query —\"{Question}\"—. " \
                            "In the image, the prediction(s) from the previous answer are shown as {num_bboxes} green bounding box(es). " \
                            "There might be errors due to a lack of comprehensive interpretation of the query and image or insufficient fine-grained analysis of the image. " \
                            "If you find any errors, please correct the error and rewrite the solution.\n" \
                            "Important rules for self-correction:\n" \
                            "- First, you should observe and analyze the image along with the query very carefully, and think about what the query —\"{Question}\"— is actually referring to.\n" \
                            "- Then, you must be sure to identify all {num_bboxes} bbox(es) in the given image, 그리고 각 bbox region 에 대해서 the query 와 match 하는지 전부 검증해.\n" \
                            "- Also, thoroughly examine the image to determine whether there are still missing target object(s) in the image that match or answer the query but are not yet covered by any bounding box.\n" \
                            "- Through the above process, if you find any errors in the previous answer, correct it and provide the final answer; otherwise, return the previous prediction as your final answer.\n" \
                            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. Output the bbox(es) and point(s) inside the interested object(s), along with a short label, in JSON format.\n" \
                            "i.e., <think> thinking process (step-by-step reasoning) here </think> " \
                            "<answer>{Answer}</answer>" 
                        
        
        self.system_prompt = "You are a helpful assistant."
        
        self.user_prompt_template = "Please find \"{Question}\" with bbox(es) and point(s). Also provide a short label for each object. " \
                "First, understand and summarize what the problem is likely referring to (which object or concept). " \
                "Then apply this to the image and find the matched target object(s).\n" 
                    
                
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if not config.enforce_eager and config.free_cache_engine:
            raise ValueError("CUDA graph should be disabled when `free_cache_engine` is True.")
        
        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        vllm_init_kwargs = {}
        if config.limit_images > 0:
            vllm_init_kwargs = {"limit_mm_per_prompt": {"image": config.limit_images}}
            
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        
        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=config.tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=config.prompt_length + config.response_length,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=config.max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            seed=0,
            disable_mm_preprocessor_cache=True,
            **vllm_init_kwargs,
            **lora_kwargs,
        )

        # print(vllm_init_kwargs)
        
        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "n": 1,
            "logprobs": 0,  # can be set to 0 and let actor to recompute
            "max_tokens": config.response_length,
            "detokenize": False}
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)
        
        self.pad_token_id = tokenizer.pad_token_id
    
    # ------------------------------------------------------------------
    # Helper: <answer> ... JSON ... </answer> 에서 bbox/point 들 파싱
    # ------------------------------------------------------------------
    def _parse_bboxes_points_from_response(self, response_str: str):
        boxes = []
        points = []
        try:
            json_match = re.search(
                r'<answer>\s*(.*?)\s*</answer>', response_str, re.DOTALL
            )
            if not json_match:
                return boxes, points
            data = json.loads(json_match.group(1))
            for item in data:
                bbox = item.get("bbox_2d", None)
                pt = item.get("point_2d", None)
                boxes.append(bbox if isinstance(bbox, list) and len(bbox) == 4 else None)
                points.append(pt if isinstance(pt, list) and len(pt) == 2 else None)
        except Exception:
            pass
        return boxes, points

    # ------------------------------------------------------------------
    # Helper: SoM overlay (bbox=초록 선, mask=노란색 fill) + 번호 라벨
    # ------------------------------------------------------------------
    def _draw_som_overlay(
        self,
        base_image: Image.Image,
        boxes: List[list],
        overlay_type: str = "bbox",
    ) -> Image.Image:
        img = base_image.convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        green = (0, 255, 0, 255)
        yellow_fill = (255, 255, 0, 80)
        yellow_edge = (255, 255, 0, 255)

        for idx, box in enumerate(boxes, start=1):
            if box is None:
                continue
            x1, y1, x2, y2 = box
            if not (x1 < x2 and y1 < y2):
                continue
            if overlay_type == "mask":
                draw.rectangle([x1, y1, x2, y2], outline=yellow_edge, fill=yellow_fill, width=2)
            else:
                draw.rectangle([x1, y1, x2, y2], outline=green, width=4)

        out = Image.alpha_composite(img, overlay).convert("RGB")
        return out

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        
        # used to construct attention_mask
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        
        batch_size = input_ids.size(0)
        
        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)],
                dtype=object
            )
        
        raw_prompt_ids_array = non_tensor_batch["raw_prompt_ids"]
        if batch_size != len(raw_prompt_ids_array):
            raise RuntimeError("vllm sharding manager is not work properly.")
        
        # problems = non_tensor_batch["problem"]
        problems = non_tensor_batch.pop("problem")
        
        # numpy -> python list[int]
        raw_prompt_ids_list: List[List[int]] = []
        for i in range(batch_size):
            ids_i = raw_prompt_ids_array[i]
            if isinstance(ids_i, np.ndarray):
                ids_i = ids_i.tolist()
            elif not isinstance(ids_i, list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(ids_i)}"
                )
            raw_prompt_ids_list.append(ids_i)

        images_list = None
        if "images" in non_tensor_batch:
            # Each element is in the form of [PIL.Image] due to collate_fn
            images_list = list(non_tensor_batch["images"])

        do_sample = prompts.meta_info.get("do_sample", True)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "n": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
            }

        lora_requests = None
        if self.lora_kwargs:
            # latest_lora_id = getattr(self.inference_engine, "_active_lora_id", None)
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(
                        lora_name=f"{lora_int_id}",
                        # lora_name="rlhf",
                        lora_int_id=lora_int_id,
                        lora_path="sj_lora_path")
                ] * batch_size

        # --------------------------- 1st turn ---------------------------
        # input configuration for passing to vLLM (1 original image)
        if images_list is not None:
            vllm_inputs_first = [
                {"prompt_token_ids": p, "multi_modal_data": {"image": imgs}}
                for p, imgs in zip(raw_prompt_ids_list, images_list)
            ]
        else:
            vllm_inputs_first = [{"prompt_token_ids": p} for p in raw_prompt_ids_list]
            
        # ensure prompt_token_ids type
        for input_data in vllm_inputs_first:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(
                   f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

        # In the case of self-correction, the response length is split across two turns.
        if self.use_self_correction and self.num_turns == 2:
            per_turn_max_tokens = max(1, self.config.response_length // 2)
        else:
            per_turn_max_tokens = self.config.response_length

        with self.update_sampling_params(**kwargs):
            completions_first: List[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs_first,
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

        # flatten 1st-turn outputs
        response_ids_first: List[List[int]] = []
        rollout_log_probs_first: List[List[float]] = []
        first_responses_text: List[str] = []
        for completion in completions_first:
            for sample_id in range(len(completion.outputs)):
                out = completion.outputs[sample_id]
                resp = out.token_ids
                response_ids_first.append(resp[:per_turn_max_tokens])
                first_responses_text.append(
                    self.tokenizer.decode(resp[:per_turn_max_tokens], skip_special_tokens=True)
                )
                if self.config.calculate_log_probs:
                    lp = []
                    for i_tok, logprob in enumerate(out.logprobs):
                        lp.append(logprob[resp[i_tok]].logprob)
                    rollout_log_probs_first.append(lp)
        
        num_rollouts = self.config.n

        response_ids_first = pad_2d_list_to_length(
            response_ids_first, self.pad_token_id, max_length=per_turn_max_tokens
        ).to(input_ids.device)
        if self.config.calculate_log_probs:
            rollout_log_probs_first = pad_2d_list_to_length(
                rollout_log_probs_first, -1, max_length=per_turn_max_tokens
            ).to(input_ids.device)
            rollout_log_probs_first = rollout_log_probs_first.to(torch.float32)

        # --------------------------- 2nd turn ---------------------------
        if not self.use_self_correction or self.num_turns != 2:
            responses = response_ids_first
            response_mask = get_response_mask(
                response_id=responses, eos_token=eos_token_id, dtype=attention_mask.dtype
            )
            response_mask_turn1 = None
            response_mask_turn2 = None
            rollout_log_probs = rollout_log_probs_first if self.config.calculate_log_probs else None
        else:
            # Based on 1st response, create a SoM image (optionally) and configure
            # the self-correction prompt for the 2nd turn.
            second_prompt_ids: List[List[int]] = []
            second_prompt_texts: List[str] = []
            second_mm_images: List[List[Image.Image]] = []

            for i in range(len(first_responses_text)):
                prompt_idx = i // num_rollouts
                prev_answer = first_responses_text[i]
                
                # Only keep the content inside <answer>...</answer> for the 2nd-turn prompt.
                # If the answer tag is missing (e.g., truncated), use an empty string.
                m = re.search(r"<answer>\s*(.*?)\s*</answer>", prev_answer, re.DOTALL)
                if m:
                    prev_answer = m.group(1).strip()
                else:
                    prev_answer = ""
                prev_answer = "<answer>" + prev_answer + "</answer>"
                
                # Question text (same with first-turn)
                q_raw = problems[prompt_idx]
                question_text = str(q_raw).lower().strip(".")
                    
                sc_instruction = self.sc_instruction_template.format(
                    Question=question_text,
                    Answer=self.example_answer,
                )
                
                # original image
                orig_imgs = images_list[prompt_idx] if images_list is not None else None

                # Set-of-Mark (SoM) Image (Original image with bbox/mask + number overlay)
                som_img = None
                if orig_imgs is not None:
                    base_img = orig_imgs[0]  # [PIL.Image]
                    boxes, _ = self._parse_bboxes_points_from_response(prev_answer)
                    som_img = self._draw_som_overlay(
                        base_image=base_img,
                        boxes=boxes,
                        overlay_type=self.som_overlay_type,
                    )
                
                mm_images = [som_img]
                # # 2nd-turn text configuration
                # if self.use_som_vp and som_img is not None and orig_imgs is not None:
                #     mm_images = [som_img]
                #     # mm_images = orig_imgs + [som_img]  # [original, som]
                # else:
                #     mm_images = orig_imgs  # [original] or None
                
                base_user_prompt = self.user_prompt_template.format(
                    Question=question_text,
                    Answer=self.example_answer,
                )
                
                user_content_second = (
                    base_user_prompt
                    # + "\n\n[Previous Answer]\n"
                    # + prev_answer
                    + "\n[Self-correction Instruction]\n"
                    + sc_instruction
                )
                
                messages = []
                if self.system_prompt is not None:
                    messages.append({"role": "system", "content": self.system_prompt})
                messages.append({"role": "user", "content": user_content_second})
                
                prompt_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                raw_prompt_text = prompt_text.replace(
                    "<image>", "<|vision_start|><|image_pad|><|vision_end|>"
                )

                full_ids = self.tokenizer.encode(
                    raw_prompt_text, add_special_tokens=False
                )
                
                second_prompt_ids.append(full_ids)
                second_prompt_texts.append(
                    self.tokenizer.decode(full_ids, skip_special_tokens=True)
                )

                if mm_images is not None:
                    second_mm_images.append(mm_images)
                else:
                    second_mm_images.append([])

                # SoM image save for debug (som_save_dir/som_images)
                if self.som_image_dir is not None and som_img is not None:
                    try:
                        sample_idx = i // num_rollouts
                        rollout_idx = i % num_rollouts
                        fname = f"som_s{sample_idx:04d}_r{rollout_idx:02d}.png"
                        som_img.save(os.path.join(self.som_image_dir, fname))
                    except Exception as e:
                        print("[SoM Debug] save error:", e)

            non_tensor_batch["prompt_turn2"] = np.array(second_prompt_texts, dtype=object)
            
            # vLLM input (2nd turn, n=1)
            vllm_inputs_second = []
            for p_ids, imgs in zip(second_prompt_ids, second_mm_images):
                if isinstance(p_ids, np.ndarray):
                    p_ids = p_ids.tolist()
                if imgs:
                    vllm_inputs_second.append(
                        {"prompt_token_ids": p_ids, "multi_modal_data": {"image": imgs}}
                    )
                else:
                    vllm_inputs_second.append({"prompt_token_ids": p_ids})

            for input_data in vllm_inputs_second:
                if isinstance(input_data["prompt_token_ids"], np.ndarray):
                    input_data["prompt_token_ids"] = input_data[
                        "prompt_token_ids"
                    ].tolist()

            # To match the number of vllm_inputs and lora_requests
            if lora_requests is not None:
                lora_requests *= num_rollouts
            
            # second turn is always sampled as n=1
            second_kwargs = kwargs.copy()
            second_kwargs["n"] = 1

            with self.update_sampling_params(max_tokens=per_turn_max_tokens, **second_kwargs):
                completions_second: List[RequestOutput] = self.inference_engine.generate(
                    prompts=vllm_inputs_second,
                    sampling_params=self.sampling_params,
                    lora_request=lora_requests,
                    use_tqdm=False,
                )

            response_ids_second: List[List[int]] = []
            rollout_log_probs_second: List[List[float]] = []
            second_responses_text: List[str] = []
            for completion in completions_second:
                out = completion.outputs[0]  # n=1
                resp = out.token_ids
                response_ids_second.append(resp)
                second_responses_text.append(
                    self.tokenizer.decode(resp, skip_special_tokens=True)
                )
                if self.config.calculate_log_probs:
                    lp = []
                    for i_tok, logprob in enumerate(out.logprobs):
                        lp.append(logprob[resp[i_tok]].logprob)
                    rollout_log_probs_second.append(lp)
            
            # for j in range(min(3, len(second_responses_text))):
            #     print(f"[DEBUG second-turn raw length] {len(response_ids_second[j])}")
            #     print(f"[DEBUG second-turn text] {second_responses_text[j][:200]}")
            
            response_ids_second = pad_2d_list_to_length(
                response_ids_second, self.pad_token_id, max_length=per_turn_max_tokens
            ).to(input_ids.device)
            if self.config.calculate_log_probs:
                rollout_log_probs_second = pad_2d_list_to_length(
                    rollout_log_probs_second, -1, max_length=per_turn_max_tokens
                ).to(input_ids.device)
                rollout_log_probs_second = rollout_log_probs_second.to(torch.float32)

            # responses = [turn1 | turn2]
            responses = torch.cat([response_ids_first, response_ids_second], dim=-1)
            response_mask_turn1 = get_response_mask(
                response_id=response_ids_first,
                eos_token=eos_token_id,
                dtype=attention_mask.dtype,
            )
            response_mask_turn2 = get_response_mask(
                response_id=response_ids_second,
                eos_token=eos_token_id,
                dtype=attention_mask.dtype,
            )
            response_mask = torch.cat(
                [response_mask_turn1, response_mask_turn2], dim=-1
            )

            if self.config.calculate_log_probs:
                rollout_log_probs = torch.cat(
                    [rollout_log_probs_first, rollout_log_probs_second], dim=-1
                )
            else:
                rollout_log_probs = None
        
        # rollout.n > 1 인 경우, prompt 쪽 tensor 들을 n번 반복
        if self.config.n > 1 and do_sample:
            batch_size = batch_size * self.config.n
            input_ids = _repeat_interleave(input_ids, self.config.n)
            attention_mask = _repeat_interleave(attention_mask, self.config.n)
            position_ids = _repeat_interleave(position_ids, self.config.n)
            if "pixel_values" in non_tensor_batch.keys():
                non_tensor_batch["pixel_values"] = _repeat_interleave(non_tensor_batch["pixel_values"], self.config.n)
                non_tensor_batch["image_grid_thw"] = _repeat_interleave(
                    non_tensor_batch["image_grid_thw"], self.config.n
                )
        
        sequence_ids = torch.cat([input_ids, responses], dim=-1)
        
        response_length = responses.size(1)
        
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch_dict = {
            "prompts": input_ids,
            "responses": responses,
            "input_ids": sequence_ids,  # here input_ids become the whole sentences
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
        }
        if self.use_self_correction and self.num_turns == 2:
            batch_dict["response_mask_turn1"] = torch.cat(
                [response_mask_turn1, torch.zeros_like(response_mask_turn1)], dim=-1
            )
            batch_dict["response_mask_turn2"] = torch.cat(
                [torch.zeros_like(response_mask_turn2), response_mask_turn2], dim=-1
            )

        batch = TensorDict(batch_dict, batch_size=batch_size)

        if self.config.calculate_log_probs:
            batch["rollout_log_probs"] = rollout_log_probs

        non_tensor_batch.pop("raw_prompt_ids", None)
        non_tensor_batch.pop("images", None)
        
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
