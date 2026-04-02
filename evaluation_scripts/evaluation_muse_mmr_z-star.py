import json
import random
import argparse
from PIL import Image as PILImage
from tqdm import tqdm
import pdb
import os
import re
import numpy as np
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
from qwen_vl_utils import process_vision_info

import torch
import torch.nn.functional as F
from dataclasses import asdict
from datasets import load_from_disk, load_dataset
from safetensors.torch import load_file, save_file

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def process_state_dict(state_dict):
    """
    Modify LoRA weight keys to match the model's expected format.
    Specifically, add '.default' to keys like 'lora_A.weight' or 'lora_B.weight'.
    """
    keys_to_modify = [k for k in state_dict.keys() if re.search(r"\.lora_[AB]\.weight$", k)]
    for old_key in keys_to_modify:
        new_key = old_key.replace(".weight", ".default.weight")
        state_dict[new_key] = state_dict.pop(old_key)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning_model_path", type=str, default="Ricky06662/Seg-Zero-7B")
    parser.add_argument("--segmentation_model_path", type=str, default="facebook/sam2-hiera-large")
    parser.add_argument("--use_lora", type=bool, default=True)
    parser.add_argument("--visualization", type=bool, default=False)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--num_parts", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--load_dataset", type=bool, default=False)
    return parser.parse_args()

# def extract_bbox_points_think(output_text, x_factor, y_factor):
#     think_pattern = r'<think>\s*(.*?)\s*</think>'
#     think_text = ""
#     think_match = re.search(think_pattern, output_text, re.DOTALL)
#     if think_match:
#         think_text = think_match.group(1)
        
#     json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text, re.DOTALL)
#     if not json_match:
#         pred_bboxes, pred_points = [], []
#         label_texts = []
#         return pred_bboxes, pred_points, think_text, label_texts

#     raw = json_match.group(1)
#     data = json.loads(raw)
#     if raw == "" or (isinstance(data, list) and len(data) == 0) or (isinstance(data, list) and len(data) == 1 and len(data[0]) == 0):
#         pred_bboxes, pred_points = [], []
#         label_texts = []
#         return pred_bboxes, pred_points, think_text, label_texts
    
#     if json_match:
#         # data = json.loads(json_match.group(1))
#         pred_bboxes = [[
#             int(item['bbox_2d'][0] * x_factor + 0.5),
#             int(item['bbox_2d'][1] * y_factor + 0.5),
#             int(item['bbox_2d'][2] * x_factor + 0.5),
#             int(item['bbox_2d'][3] * y_factor + 0.5)
#         ] for item in data]
#         pred_points = [[
#             int(item['point_2d'][0] * x_factor + 0.5),
#             int(item['point_2d'][1] * y_factor + 0.5)
#         ] for item in data]
#         label_texts = [item["label"] for item in data]
    
#     return pred_bboxes, pred_points, think_text, label_texts


def extract_bbox_points_think(output_text, x_factor, y_factor):
    think_pattern = r'<think>\s*(.*?)\s*</think>'
    think_text = ""
    think_match = re.search(think_pattern, output_text, re.DOTALL)
    if think_match:
        think_text = think_match.group(1)
        
    json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text, re.DOTALL)
    if not json_match:
        pred_bboxes, pred_points = [], []
        label_texts = []
        return pred_bboxes, pred_points, think_text, label_texts

    raw = json_match.group(1)
    try:
        data = json.loads(raw)
    except Exception:
        # 예외 케이스: <answer> 내부에 JSON 배열이 여러 개 연속 배치된 경우
        # 예) [ {...}, {...} ] \n [ {...} ]
        dec = json.JSONDecoder()
        s = raw.strip()
        values = []
        idx = 0
        while True:
            # 공백 스킵
            while idx < len(s) and s[idx].isspace():
                idx += 1
            if idx >= len(s):
                break
            try:
                val, end = dec.raw_decode(s[idx:])
            except json.JSONDecodeError:
                break
            values.append(val)
            idx += end
        # 여러 값 병합: 리스트는 이어붙이고, dict는 단일 항목으로 취급
        combined = []
        for v in values:
            if isinstance(v, list):
                combined.extend(v)
            elif isinstance(v, dict):
                combined.append(v)
        data = combined
        
    if raw == "" or (isinstance(data, list) and len(data) == 0) or (isinstance(data, list) and len(data) == 1 and len(data[0]) == 0):
        pred_bboxes, pred_points = [], []
        label_texts = []
        return pred_bboxes, pred_points, think_text, label_texts
    
    if json_match:
        # data = json.loads(json_match.group(1))
        pred_bboxes = [[
            int(item['bbox_2d'][0] * x_factor + 0.5),
            int(item['bbox_2d'][1] * y_factor + 0.5),
            int(item['bbox_2d'][2] * x_factor + 0.5),
            int(item['bbox_2d'][3] * y_factor + 0.5)
        ] for item in data]
        pred_points = [[
            int(item['point_2d'][0] * x_factor + 0.5),
            int(item['point_2d'][1] * y_factor + 0.5)
        ] for item in data]
        label_texts = [item["label"] for item in data]
    
    return pred_bboxes, pred_points, think_text, label_texts


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    # if union == 0:
    #     return 0
    return intersection, union

def compute_bbox_iou(bbox1, bbox2):
    # 计算两个bbox的交集区域
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    # 计算交集面积
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 计算两个bbox的面积
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    # 计算并集面积
    union = area1 + area2 - intersection
    
    # 避免除以0
    if union == 0:
        return 0
    
    return intersection / union


def process_safetensors_files(folder_path: str):
    if not is_main_process():
        return  # Skip for non-main ranks
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".safetensors"):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing: {file_path}")

            # Load tensors from safetensors file
            tensors = load_file(file_path)
            new_tensors = {}

            for key, value in tensors.items():
                # Skip keys containing "lora"
                # if "lora" in key:
                #     continue

                # Remove substrings from key
                new_key = key.replace("base_model.model.", "")
                # new_key = new_key.replace(".base_layer", "")

                new_tensors[new_key] = value

            # Save to new file (overwrite original)
            save_file(new_tensors, file_path)
            print(f"Saved: {file_path}")


def main():
    args = parse_args()
    #We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        # args.reasoning_model_path,
        args.reasoning_model_path if not args.use_lora else "pretrained_models/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    # segmentation_model = SAM2ImagePredictor.from_pretrained(args.segmentation_model_path)
    checkpoint = "./sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    segmentation_model = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
    
    if args.use_lora:
        # 1) Load your trained LoRA adapter and fuse (merge) it into the base model
        print("Fusing LoRA Adapter!")
        from peft import LoraConfig, get_peft_model
        
        adapter_path = args.reasoning_model_path[:-len("huggingface")] + "lora_adapter"
        lora_config = LoraConfig.from_pretrained(adapter_path)
        peft_model = get_peft_model(reasoning_model, lora_config)
        
        state_dict = load_file(f"{adapter_path}/adapter_model.safetensors")
        process_state_dict(state_dict)
        load_result = peft_model.load_state_dict(state_dict, strict=False)
        # print("Missing keys:", load_result.missing_keys)
        # print("Unexpected keys:", load_result.unexpected_keys)
        
        reasoning_model = peft_model.to("cuda")
        
        # Merge LoRA deltas into the base Qwen2.5‐VL weights and unload the wrapper.
        # After this, `reasoning_model` is a plain Qwen2_5_VLForConditionalGeneration
        # with no adapter overhead at inference time.
        # reasoning_model = reasoning_model.merge_and_unload()
        
    reasoning_model.eval()  # just to be explicit

    # default processer
    processor = AutoProcessor.from_pretrained(args.reasoning_model_path, padding_side="left")
    
    resize_size = 840
    if args.load_dataset:
        dataset = load_dataset(args.test_data_path, split='test')
    else:
        dataset = load_from_disk(args.test_data_path)
    total_len = len(dataset)
    
    part_size = total_len // args.num_parts
    start_idx = args.idx * part_size
    end_idx = start_idx + part_size if args.idx < args.num_parts - 1 else total_len
    
    # pdb.set_trace()
    dataset = dataset.select(range(start_idx, end_idx))
    
    if args.visualization:
        vis_dir = os.path.join(args.output_path, "visualization")
        os.makedirs(vis_dir, exist_ok=True)
    
    if 'bbox' in dataset[0]:
        has_bbox = True
    else:
        has_bbox = False
    
    # QUESTION_TEMPLATE = \
    #     "Please find \"{Question}\" with bbox(es) and point(s). " \
    #     "Also provide a short label for each object. " \
    #     "Compare the difference between object(s) and find the most closely matched object(s). " \
    #     "Return ALL matching instances; double-check none are missed. " \
    #     "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. " \
    #     "Output the bbox(es) and point(s) inside the interested object(s), along with a short label, in JSON format. " \
    #     "i.e., <think> thinking process (step-by-step reasoning) here </think> " \
    #     "<answer>{Answer}</answer>"
    
    QUESTION_TEMPLATE = \
        "Please find \"{Question}\" with bbox(es) and point(s). " \
        "Also provide a short label for each object. " \
        "First, understand and summarize what the problem —\"{Question}\"— is likely referring to (which object or concept). " \
        "Then apply this to the image and find the matched target object(s). " \
        "Return ALL matching instances; double-check none are missed. " \
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. " \
        "Output the bbox(es) and point(s) inside the interested object(s), along with a short label, in JSON format. " \
        "i.e., <think> thinking process (step-by-step reasoning) here </think> " \
        "<answer>{Answer}</answer>"
    
    messages = []
    id_list = []
    for item in dataset:
        image = item["image"].convert("RGB")
        message = [{
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": image.resize((resize_size, resize_size), PILImage.BILINEAR)
                },
                {   
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(
                        Question=item["text"].lower().strip(".\"?!"),
                        Answer="[{\"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110], \"label\": \"chair\"}, {\"bbox_2d\": [225,296,706,786], \"point_2d\": [302,410], \"label\": \"train track\"}]" 
                    )    
                }
            ]
        }]
        messages.append(message)
        id_list.append({
            "image_id": item["image_id"],
            "ann_id": item["ann_id"],
            "text": item["text"],
            "image": image,
            "mask": item["mask"],
            "img_height": item["img_height"],
            "img_width": item["img_width"],
            "bbox": item["bbox"] if has_bbox else None
        })

    all_outputs = []
    for i in tqdm(range(0, len(messages), args.batch_size)):
        batch_messages = messages[i:i + args.batch_size]
        batch_id_list = id_list[i:i + args.batch_size]
        
        # Preparation for inference
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # Inference: Generation of the output
        generated_ids = reasoning_model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        
        # pdb.set_trace()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for id_idx in range(len(batch_output_text)):
                try:
                    bboxes, points, think, label = extract_bbox_points_think(
                                            batch_output_text[id_idx], 
                                            batch_id_list[id_idx]["img_width"]/resize_size, 
                                            batch_id_list[id_idx]["img_height"]/resize_size
                                        )
                except Exception as e:
                    # add penalty in this situation
                    print("Reasoning error: ", e, "Text: ", batch_output_text[id_idx], "ID: ", batch_id_list[id_idx]["image_id"])
                    think = ""
                    label = ""
                    intersection = 0
                    union = np.array(batch_id_list[id_idx]["mask"]).sum()
                    bbox_iou = 0.0
                    all_outputs.append({
                        "image_id": batch_id_list[id_idx]["image_id"],
                        "ann_id": batch_id_list[id_idx]["ann_id"],
                        "think": think,
                        "label": label,
                        "intersection": int(intersection),
                        "union": int(union),
                        "bbox_iou": bbox_iou
                    })
                    continue
                try:
                    segmentation_model.set_image(batch_id_list[id_idx]["image"])
                    mask_all = np.zeros((batch_id_list[id_idx]["img_height"], batch_id_list[id_idx]["img_width"]), dtype=bool)
                except Exception as e:
                    print("Set image error: ", e, batch_id_list[id_idx]["image_id"], batch_id_list[id_idx]["ann_id"])
                    # skip this because the image or mask is not correct
                    continue
                try:
                    for bbox, point in zip(bboxes, points):
                        masks, scores, _ = segmentation_model.predict(
                            point_coords=[point],
                            point_labels=[1],
                            box=bbox
                        )
                        sorted_ind = np.argsort(scores)[::-1]
                        masks = masks[sorted_ind]
                        mask = masks[0].astype(bool)
                        mask_all = np.logical_or(mask_all, mask)
                    gt_mask = np.array(batch_id_list[id_idx]["mask"])
                except Exception as e:
                    print("Segmentation error: ", e, batch_id_list[id_idx]["image_id"], batch_id_list[id_idx]["ann_id"])
                    # skip this because the image or mask is not correct
                    continue
                try:
                    intersection, union = compute_iou(mask_all, gt_mask)
                except Exception as e:
                    print("Image error: ", e)
                    # skip this because the image or mask is not correct
                    continue 
                
                if args.visualization:
                    try:
                        img = batch_id_list[id_idx]["image"]
                        img_id = batch_id_list[id_idx]["image_id"]
                        query_text = batch_id_list[id_idx].get("text", "")
                        # 좌/우 패널 준비
                        fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=150)
                        # 좌: GT overlay
                        axes[0].imshow(img)
                        H_gt, W_gt = gt_mask.shape
                        gt_overlay = np.zeros((H_gt, W_gt, 4), dtype=np.float32)
                        # 반투명 빨강
                        gt_overlay[gt_mask] = [1.0, 0.0, 0.0, 0.4]
                        axes[0].imshow(gt_overlay)
                        axes[0].set_title("GT Overlay")
                        axes[0].axis("off")
                        if not gt_mask.any():
                            # 좌측 상단 위치를 이미지 좌표로 계산 (margin 10px 정도)
                            star_x = 10
                            star_y = 10
                            axes[0].plot(
                                star_x, star_y,
                                marker="*", markersize=15,
                                markeredgecolor="black", markerfacecolor="black"
                            )
                        # 우: Pred overlay ( bbox / points)
                        axes[1].imshow(img)
                        H_pr, W_pr = mask_all.shape
                        pred_overlay = np.zeros((H_pr, W_pr, 4), dtype=np.float32)
                        # 연한 초록색
                        pred_overlay[mask_all] = [0.0, 1.0, 0.0, 0.4]
                        axes[1].imshow(pred_overlay)
                        # bbox & points
                        for bbox, point in zip(bboxes, points):
                            x1, y1, x2, y2 = bbox
                            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=1.5)
                            axes[1].add_patch(rect)
                            # pos: 파란색 별
                            axes[1].plot(point[0], point[1], marker="*", markersize=7, markeredgecolor="blue", markerfacecolor="blue")
                        axes[1].set_title("Pred Overlay")
                        axes[1].axis("off")
                        # 제목/Label/IoU 텍스트
                        fig.suptitle(query_text, fontsize=7)
                        label_str = ", ".join(label)
                        fig.text(0.5, 0.05, f"Label : {label_str}", ha="center", va="bottom", fontsize=6)
                        iou_val = (intersection / union) if union > 0 else 1.0
                        fig.text(0.5, 0.02, f"IoU: {iou_val:.3f}", ha="center", va="bottom")
                        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
                        rand_n = random.randint(1, 100000)
                        out_png = os.path.join(vis_dir, f"{img_id}_{rand_n}.png")
                        fig.savefig(out_png)
                        plt.close(fig)
                    except Exception as e:
                        print("Visualization error: ", e, batch_id_list[id_idx]["image_id"], batch_id_list[id_idx]["ann_id"])
                
                
                bbox_iou = 0.0
                if has_bbox:
                    try:     
                        gt_bbox = batch_id_list[id_idx]["bbox"]
                        for pred_bbox in bboxes:
                            if compute_bbox_iou(pred_bbox, gt_bbox) > 0.5:
                                bbox_iou = 1.0
                                break
                    except Exception as e:
                        print("Bbox error: ", e, batch_id_list[id_idx]["image_id"], batch_id_list[id_idx]["ann_id"])
                        # skip this because the image or mask is not correct
                        bbox_iou = 0.0

                
                all_outputs.append({
                    "image_id": batch_id_list[id_idx]["image_id"],
                    "ann_id": batch_id_list[id_idx]["ann_id"],
                    "think": think,
                    "label": label,
                    "intersection": int(intersection),
                    "union": int(union),
                    "bbox_iou": bbox_iou
                })
        print(f"Processed batch {i//args.batch_size + 1}/{(len(messages) + args.batch_size - 1)//args.batch_size}")
        
        # clean GPU memory
        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()

    
    # Modify the output file name, add idx
    output_file = os.path.join(args.output_path, f"output_{args.idx}.json")
    with open(output_file, "w") as f:
        json.dump(all_outputs, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
