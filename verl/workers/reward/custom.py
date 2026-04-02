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


import os
import torch
import numpy as np
from collections import defaultdict
from typing import Optional
from transformers import PreTrainedTokenizer
from PIL import Image, ImageDraw
from tensordict import TensorDict

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.reward_score import (
    math_compute_score,
    r1v_compute_score,
    seg_compute_score,
    seg_strict_compute_score,
    vision_reasoner_compute_score,
)


class CustomRewardManager:
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 num_examine: int,
                 compute_score: str,
                 rollout_n: int,
                 sam_actor=None,
                 is_qwen3=False,
                 self_correction_bonus_alpha: float = 0.0,
                 som_debug_dir: Optional[str] = None,
                 som_overlay_type: str = "bbox",
                 ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score_type = compute_score
        self.rollout_n = rollout_n
        self.is_qwen3 = is_qwen3
        self.is_stage2 = (compute_score == "star_s2")
        # bonus = alpha * (R_acc(Y2) - R_acc(Y1)) - scaling factor
        self.self_correction_bonus_alpha = self_correction_bonus_alpha
        
        self.som_debug_dir = som_debug_dir
        self.som_overlay_type = som_overlay_type  # "bbox" or "mask"
        self.som_comparison_dir: Optional[str] = None
        if self.som_debug_dir is not None:
            os.makedirs(self.som_debug_dir, exist_ok=True)
            # turn1/turn2 비교 이미지는 하위 comparison_images 폴더에 저장
            self.som_comparison_dir = os.path.join(self.som_debug_dir, "comparison_images")
            os.makedirs(self.som_comparison_dir, exist_ok=True)
        self._som_debug_idx = 0
        
        if compute_score == "math":
            self.compute_score = math_compute_score
        elif compute_score == "r1v":
            self.compute_score = r1v_compute_score
        elif compute_score == "seg":
            self.compute_score = seg_compute_score
        elif compute_score == "seg_strict":
            self.compute_score = seg_strict_compute_score
        elif compute_score == "vision_reasoner":
            self.compute_score = vision_reasoner_compute_score
        elif compute_score.startswith("star"):
            print("compute score: segment anything reasoner")
            self.sam_actor = sam_actor
        else:
            raise NotImplementedError()

    def __call__(self, data: DataProto) -> torch.Tensor:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_details = defaultdict(list)
        already_print = 0
        
        # --------------------------------------------------------------
        # SoM overlay Utils
        # --------------------------------------------------------------
        def _parse_bboxes_from_response(resp_str: str):
            import json, re

            boxes = []
            try:
                m = re.search(r"<answer>\s*(.*?)\s*</answer>", resp_str, re.DOTALL)
                if not m:
                    return boxes
                data_json = json.loads(m.group(1))
                for item in data_json:
                    bbox = item.get("bbox_2d", None)
                    if isinstance(bbox, list) and len(bbox) == 4:
                        boxes.append(bbox)
            except Exception:
                pass
            return boxes

        def _draw_som_overlay(image: Image.Image, boxes, overlay_type: str):
            img = image.convert("RGBA")
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            green = (0, 255, 0, 255)
            yellow_fill = (255, 255, 0, 80)
            yellow_edge = (255, 255, 0, 255)

            for idx, box in enumerate(boxes, start=1):
                if box is None:
                    continue
                x1, y1, x2, y2 = box
                if overlay_type == "mask":
                    draw.rectangle([x1, y1, x2, y2], outline=yellow_edge, fill=yellow_fill, width=2)
                else:
                    draw.rectangle([x1, y1, x2, y2], outline=green, width=3)
                # 번호
                tx, ty = x1 + 4, y1 + 4
                draw.rectangle(
                    [tx - 2, ty - 2, tx + 14, ty + 14],
                    fill=(0, 0, 0, 160),
                )
                draw.text((tx, ty), str(idx), fill=(255, 255, 255, 255))
            return Image.alpha_composite(img, overlay).convert("RGB")
        
        def _draw_mask_overlay(image: Image.Image, masks, per_instance_ious=None):
            """
            sam_worker 가 반환한 predicted mask 들을 노란색으로 overlay.
            각 instance 에 대해서는 "index:iou" 형식으로 라벨을 붙인다.
            """
            base = image.convert("RGBA")
            overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))

            for idx, mask in enumerate(masks):
                if mask is None:
                    continue

                # mask 를 0/1 uint8 로 변환
                if isinstance(mask, Image.Image):
                    mask_img = mask.convert("L")
                else:
                    if hasattr(mask, "cpu"):
                        mask_arr = mask.cpu().numpy()
                    else:
                        mask_arr = np.array(mask)
                    if mask_arr.ndim == 3:
                        mask_arr = mask_arr[..., 0]
                    mask_img = Image.fromarray(
                        (mask_arr > 0).astype(np.uint8) * 255, mode="L"
                    )

                yellow = Image.new("RGBA", base.size, (255, 255, 0, 80))
                overlay = Image.composite(yellow, overlay, mask_img)

                # 중심 위치 계산
                mask_np = np.array(mask_img)
                ys, xs = np.where(mask_np > 0)
                if len(xs) > 0:
                    cx, cy = float(xs.mean()), float(ys.mean())
                else:
                    cx, cy = 5.0, 5.0

                text = f"{idx + 1}"

                draw = ImageDraw.Draw(overlay)
                draw.rectangle(
                    [cx - 2, cy - 2, cx + 40, cy + 14],
                    fill=(0, 0, 0, 160),
                )
                draw.text((cx, cy), text, fill=(255, 255, 255, 255))
                
            return Image.alpha_composite(base, overlay).convert("RGB")
        
        def _add_global_iou_text(image: Image.Image, iou_value: float, prefix: str):
            """이미지 하단에 전체 mask_iou (mIoU) 텍스트를 추가."""
            base = image.convert("RGB")
            draw = ImageDraw.Draw(base)
            font = ImageFont.load_default()
            w, h = base.size
            text = f"{prefix} mIoU = {iou_value:.3f}"
            tw, th = draw.textsize(text, font=font)
            x0, y0 = 5, h - th - 6
            draw.rectangle([x0 - 2, y0 - 2, x0 + tw + 2, y0 + th + 2], fill=(0, 0, 0, 160))
            draw.text((x0, y0), text, fill=(255, 255, 255, 255), font=font)
            return base

        # ------------------------------------------------------------------
        # Stage2 + multi-turn (self-correction):
        #   - for each row, 1st / 2nd turn responses are concatenated,
        #   - and decomposed using response_mask_turn1 / response_mask_turn2
        # ------------------------------------------------------------------
        
        is_multi_turn_stage2 = (
            self.is_stage2
            and "response_mask_turn1" in data.batch.keys()
            and "response_mask_turn2" in data.batch.keys()
        )
        
        if is_multi_turn_stage2:
            # responses = [turn1, turn2] concat
            total_resp_len = data.batch["responses"].size(-1)
            per_turn_len = total_resp_len // 2  # current implementation only supports num_turns=2

            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem

                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]

                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                all_responses = data_item.batch["responses"]
                mask_turn1 = data_item.batch["response_mask_turn1"]
                mask_turn2 = data_item.batch["response_mask_turn2"]

                # 1st turn segment
                seg1_ids = all_responses[:per_turn_len]
                seg1_mask = mask_turn1[:per_turn_len].bool()
                valid_len1 = int(seg1_mask.sum().item())
                if valid_len1 > 0:
                    resp1_ids = seg1_ids[:valid_len1]
                    response_str1 = self.tokenizer.decode(resp1_ids, skip_special_tokens=True)
                else:
                    response_str1 = ""

                # 2nd turn segment
                seg2_ids = all_responses[per_turn_len : per_turn_len * 2]
                seg2_mask = mask_turn2[per_turn_len : per_turn_len * 2].bool()
                valid_len2 = int(seg2_mask.sum().item())
                if valid_len2 > 0:
                    resp2_ids = seg2_ids[:valid_len2]
                    response_str2 = self.tokenizer.decode(resp2_ids, skip_special_tokens=True)
                else:
                    response_str2 = ""

                # Stage2 uses only mask + image instead of ground_truth string (bbox/point)
                ground_truth = None
                solution_mask = data_item.non_tensor_batch["solution_mask"]
                image = data_item.non_tensor_batch["image"]

                # 1st turn reward (rs prediction)
                score_out1 = self.sam_actor.compute_reward(
                    response_str1,
                    ground_truth,
                    solution_mask,
                    image,
                    self.is_qwen3,
                    True,
                )
                if isinstance(score_out1, tuple):
                    score1, details1 = score_out1
                else:
                    score1, details1 = score_out1, {}

                # 2nd turn reward (rs prediction + self-correction)
                score_out2 = self.sam_actor.compute_reward(
                    response_str2,
                    ground_truth,
                    solution_mask,
                    image,
                    self.is_qwen3,
                    True,
                )
                if isinstance(score_out2, tuple):
                    score2, details2 = score_out2
                else:
                    score2, details2 = score_out2, {}

                acc1 = float(details1.get("accuracy", 0.0))
                acc2 = float(details2.get("accuracy", 0.0))
                iou1 = float(details1.get("mask_iou", 0.0))
                iou2 = float(details2.get("mask_iou", 0.0))
                delta_iou = iou2 - iou1

                bonus = self.self_correction_bonus_alpha * (acc2 - acc1)
                
                final_score1 = score1
                final_score2 = score2 + bonus

                # token-level reward
                if valid_len1 > 0:
                    reward_tensor[i, valid_len1 - 1] = final_score1
                if valid_len2 > 0:
                    reward_tensor[i, per_turn_len + valid_len2 - 1] = final_score2

                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                prompt_turn2 = data_item.non_tensor_batch["prompt_turn2"]

                if already_print < self.num_examine:
                    already_print += 1
                    print("[prompt_turn1]", prompt_str)
                    print("[response_turn1]", response_str1)
                    print("[prompt_turn2]", prompt_turn2)
                    print("[response_turn2]", response_str2)
                    print(
                        "[acc1]", acc1,
                        "[acc2]", acc2,
                        "[bonus]", bonus,
                        "[delta_iou]", delta_iou,
                    )
                    # print("[acc1]", acc1, "[acc2]", acc2, "[bonus]", bonus)

                # SoM Debug Image Storage
                # Left: first-turn SAM masks, Right: second-turn SAM masks
                if self.som_debug_dir is not None and hasattr(
                    data_item.non_tensor_batch, "get"
                ):
                    try:
                        src_img = data_item.non_tensor_batch["image"]
                        if isinstance(src_img, Image.Image):
                            masks1 = details1.get("pred_masks", None)
                            masks2 = details2.get("pred_masks", None)

                            if masks1 is not None and masks2 is not None:
                                img1 = _draw_mask_overlay(src_img, masks1)
                                img2 = _draw_mask_overlay(src_img, masks2)
                            else:
                                # fallback: bbox overlay
                                boxes1 = _parse_bboxes_from_response(response_str1)
                                boxes2 = _parse_bboxes_from_response(response_str2)
                                img1 = _draw_som_overlay(src_img, boxes1, self.som_overlay_type)
                                img2 = _draw_som_overlay(src_img, boxes2, self.som_overlay_type)

                            # 각 이미지 하단에 평균 mask_iou 를 표시
                            img1 = _add_global_iou_text(img1, iou1, "T1")
                            img2 = _add_global_iou_text(img2, iou2, "T2")

                            w, h = img1.size
                            cat_img = Image.new("RGB", (w * 2, h), (255, 255, 255))
                            cat_img.paste(img1, (0, 0))
                            cat_img.paste(img2, (w, 0))

                            save_path = os.path.join(
                                self.som_comparison_dir,
                                f"comparison_{self._som_debug_idx:06d}.png",
                            )
                            cat_img.save(save_path)
                            self._som_debug_idx += 1
                    except Exception as e:
                        print("[SoM debug save error]", e)

                # logging metrics
                for k, v in details1.items():
                    if k != "pred_masks":
                        reward_details[f"turn1/{k}"].append(v)
                for k, v in details2.items():
                    if k != "pred_masks":
                        reward_details[f"turn2/{k}"].append(v)
                reward_details["self_correction/bonus"].append(bonus)
                reward_details["self_correction/delta_mask_iou"].append(delta_iou)

            ious = np.array(reward_details["mask_iou"])
            corrects = (ious >= 2.0).astype(float)
            per_sample_matrix = corrects.reshape(-1, self.rollout_n)
            per_sample_acc = per_sample_matrix.mean(axis=1)
            print(f"[Sample-wise Rollout Acc] {per_sample_acc.tolist()}")
            reward_details["acc"] = corrects.tolist()
            
            reward_stats = {k: sum(v) / len(v) for k, v in reward_details.items()}
            return reward_tensor, reward_stats, ious
        
        # ---------- 2) STAR 계열: SAM2 + multi-GPU DP_COMPUTE_PROTO ----------
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_details = defaultdict(list)

        predict_strs = []
        ground_truths = []
        solution_masks = []
        images = []
        last_token_indices = []

        already_print = 0
        for i in range(len(data)): 
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            ground_truth = data_item.non_tensor_batch["solution"] if not self.is_stage2 else None
            
            predict_strs.append(response_str)
            ground_truths.append(ground_truth)
            solution_masks.append(data_item.non_tensor_batch["solution_mask"])
            images.append(data_item.non_tensor_batch["image"])
            last_token_indices.append(int(valid_response_length - 1))

            if already_print < self.num_examine:
                already_print += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)

        # ---- SAM worker에 넘길 DataProto 생성 (meta_info로 플래그 전달) ----
        num = len(predict_strs)
        batch_td = TensorDict(
            {
                "images": images
            }, 
            batch_size=(num,)
        )
        sam_input = DataProto(
            # batch=None,
            batch=batch_td,
            non_tensor_batch={
                "predict_str": np.asarray(predict_strs, dtype=object),
                "ground_truth": np.asarray(ground_truths, dtype=object),
                "solution_mask": np.asarray(solution_masks, dtype=object),
                # "image": np.asarray(images, dtype=object),
            },
            meta_info={
                "is_qwen3": self.is_qwen3,
                "is_stage2": self.is_stage2,
            },
        )

        # world_size로 나누어떨어지도록 패딩 (DP_COMPUTE_PROTO의 chunk() 제약)
        # sam_input_padded, pad_size = pad_dataproto_to_divisor(sam_input, self.sam_actor.world_size)
        sam_output: DataProto = self.sam_actor.compute_reward_batch(sam_input).get()
        # sam_output: DataProto = unpad_dataproto(sam_output_padded, pad_size)

        scores = sam_output.batch["scores"].cpu()
        assert scores.shape[0] == len(data), \
            f"Score length {scores.shape[0]} != batch size {len(data)}"

        for i, last_idx in enumerate(last_token_indices):
            reward_tensor[i, last_idx] = scores[i]

        # metric들 수집
        for k, v in sam_output.non_tensor_batch.items():
            if isinstance(v, np.ndarray):
                reward_details[k] = v.tolist()
            else:
                reward_details[k] = list(v)

        ious = np.array(reward_details["mask_iou"])
        corrects = (ious >= 2.0).astype(float)
        per_sample_matrix = corrects.reshape(-1, self.rollout_n)
        per_sample_acc = per_sample_matrix.mean(axis=1)
        print(f"[Sample-wise Rollout Acc] {per_sample_acc.tolist()}")
        reward_details["acc"] = corrects.tolist()
        
        reward_stats = {k: sum(v) / len(v) for k, v in reward_details.items() if k != "pred_masks"}
                        
        return reward_tensor, reward_stats, ious
   