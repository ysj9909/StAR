import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from math import isfinite
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
try:
    from transformers import Qwen3VLForConditionalGeneration
except Exception:
    Qwen3VLForConditionalGeneration = None
from qwen_vl_utils import process_vision_info
import torch
import json
import re

from PIL import Image as PILImage
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
from safetensors.torch import load_file
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning_model_path", type=str, default="pretrained_models/StAR-7B/huggingface")
    parser.add_argument("--vl_model_version", type=str, default="qwen2_5", choices=["qwen2_5", "qwen3"])
    parser.add_argument("--qwen3_base_path", type=str, default="pretrained_models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--use_lora", type=str, default="true")
    parser.add_argument("--text", type=str, default="What can I have if I'm thirsty?")
    parser.add_argument("--image_path", type=str, default="./assets/food.webp")
    parser.add_argument("--output_path", type=str, default="./inference_scripts/test_output_multiobject.png")
    return parser.parse_args()


def process_state_dict(state_dict):
    keys_to_modify = [k for k in state_dict.keys() if re.search(r"\.lora_[AB]\.weight$", k)]
    for old_key in keys_to_modify:
        new_key = old_key.replace(".weight", ".default.weight")
        state_dict[new_key] = state_dict.pop(old_key)


# ──── Qwen3-VL coordinate helpers ────

def _infer_qwen3_grid_from_values(values):
    if not values:
        return 999
    vmax = 0.0
    for v in values:
        try:
            fv = float(v)
            if isfinite(fv):
                vmax = max(vmax, abs(fv))
        except Exception:
            continue
    if vmax <= 1.2:
        return 1
    if vmax <= 1000.5:
        return 999
    return None


def _scale_qwen3_coord_to_pixel(val, grid, size):
    if grid is None:
        vv = round(val)
    elif grid == 1:
        vv = round(float(val) * (size - 1))
    else:
        vv = round(float(val) * (size - 1) / 999.0)
    return max(0, min(size - 1, int(vv)))


def _order_box_xyxy(x1, y1, x2, y2):
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _convert_qwen3_predictions_to_pixels(pred_bboxes, pred_points, image,
                                          default_w=840, default_h=840, force_grid=None):
    if image is not None:
        try:
            W, H = image.size
        except Exception:
            W, H = default_w, default_h
    else:
        W, H = default_w, default_h

    vals = []
    for b in pred_bboxes:
        if isinstance(b, (list, tuple)) and len(b) == 4:
            vals += [b[0], b[1], b[2], b[3]]
    for p in pred_points:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            vals += [p[0], p[1]]
    grid = force_grid if force_grid is not None else _infer_qwen3_grid_from_values(vals)

    out_boxes, out_points = [], []
    for b in pred_bboxes:
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            continue
        x1 = _scale_qwen3_coord_to_pixel(b[0], grid, W)
        y1 = _scale_qwen3_coord_to_pixel(b[1], grid, H)
        x2 = _scale_qwen3_coord_to_pixel(b[2], grid, W)
        y2 = _scale_qwen3_coord_to_pixel(b[3], grid, H)
        x1, y1, x2, y2 = _order_box_xyxy(x1, y1, x2, y2)
        out_boxes.append([x1, y1, x2, y2])
    for p in pred_points:
        if not (isinstance(p, (list, tuple)) and len(p) == 2):
            continue
        px = _scale_qwen3_coord_to_pixel(p[0], grid, W)
        py = _scale_qwen3_coord_to_pixel(p[1], grid, H)
        out_points.append([px, py])
    return out_boxes, out_points


# ──── Output parsing ────

def extract_bbox_points_think(output_text, x_factor, y_factor, *, is_qwen3=False, image=None):
    think_pattern = r'<think>\s*(.*?)\s*</think>'
    think_text = ""
    think_match = re.search(think_pattern, output_text, re.DOTALL)
    if think_match:
        think_text = think_match.group(1)

    if is_qwen3 and (not think_text):
        ans_start = re.search(r'<answer>\s*', output_text, re.DOTALL)
        if ans_start:
            think_text = output_text[:ans_start.start()].strip()

    json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text, re.DOTALL)
    if not json_match:
        return [], [], think_text, []

    raw = json_match.group(1)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        fixed = raw
        fixed = re.sub(r'("point_2d"\s*:\s*\[[^\]]*?)\}\s*(\})', r'\1]\2', fixed)
        fixed = re.sub(r'("point_2d"\s*:\s*\[[^\]]*?)\}\s*(])', r'\1]}\2', fixed)
        fixed = re.sub(r'}}(\s*])', r'}\1', fixed)

        def _strip_quotes_in_array(match):
            inner = match.group(2).replace('"', '')
            return match.group(1) + inner + match.group(3)

        fixed = re.sub(r'("bbox_2d"\s*:\s*\[)([^\]]*)(\])', _strip_quotes_in_array, fixed)
        fixed = re.sub(r'("point_2d"\s*:\s*\[)([^\]]*)(\])', _strip_quotes_in_array, fixed)
        fixed = re.sub(r'("bbox_2d"\s*:\s*\[[^\]]*\])"', r'\1', fixed)
        fixed = re.sub(r'("point_2d"\s*:\s*\[[^\]]*\])"', r'\1', fixed)
        try:
            data = json.loads(fixed)
        except json.JSONDecodeError:
            return [], [], think_text, []

    if not data or (isinstance(data, list) and len(data) == 0):
        return [], [], think_text, []

    if is_qwen3:
        raw_bboxes = [item.get("bbox_2d") for item in data if isinstance(item, dict)]
        raw_points = [item.get("point_2d") for item in data if isinstance(item, dict)]
        raw_bboxes = [b for b in raw_bboxes if isinstance(b, (list, tuple)) and len(b) == 4]
        raw_points = [p for p in raw_points if isinstance(p, (list, tuple)) and len(p) == 2]
        pred_bboxes, pred_points = _convert_qwen3_predictions_to_pixels(
            raw_bboxes, raw_points, image, default_w=840, default_h=840,
        )
    else:
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

    label_texts = [item.get("label", "") for item in data if isinstance(item, dict)]
    return pred_bboxes, pred_points, think_text, label_texts


def main():
    args = parse_args()
    use_lora = args.use_lora.lower() in ("true", "1", "yes")
    is_qwen3 = (args.vl_model_version == "qwen3")

    # Load reasoning model
    if is_qwen3:
        if Qwen3VLForConditionalGeneration is None:
            raise ImportError("Qwen3VLForConditionalGeneration not found. Please upgrade transformers.")
        base_path = args.qwen3_base_path if use_lora else args.reasoning_model_path
        reasoning_model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    else:
        base_path = "pretrained_models/Qwen2.5-VL-7B-Instruct" if use_lora else args.reasoning_model_path
        reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

    # Load SAM 2.1
    checkpoint = "./sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    segmentation_model = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    # Load LoRA adapter
    if use_lora:
        print("Fusing LoRA Adapter!")
        from peft import LoraConfig, get_peft_model

        adapter_path = args.reasoning_model_path[:-len("huggingface")] + "lora_adapter"
        lora_config = LoraConfig.from_pretrained(adapter_path)
        peft_model = get_peft_model(reasoning_model, lora_config)

        state_dict = load_file(f"{adapter_path}/adapter_model.safetensors")
        process_state_dict(state_dict)
        peft_model.load_state_dict(state_dict, strict=False)

        reasoning_model = peft_model.to("cuda")

    reasoning_model.eval()

    processor = AutoProcessor.from_pretrained(args.reasoning_model_path, padding_side="left")

    print("User question:", args.text)

    # Prompt template (identical to evaluation_star.py default)
    QUESTION_TEMPLATE = \
        "Please find \"{Question}\" with bbox(es) and point(s). " \
        "Also provide a short label for each object. " \
        "First, understand and summarize what the query —\"{Question}\"— is likely referring to (which object or concept). " \
        "Then apply this to the image and find the matched target object(s). " \
        "Return ALL matching instances; if there are no matches, return an empty list (<answer>[]</answer>). double-check none are missed. " \
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. " \
        "Output the bbox(es) and point(s) inside the interested object(s), along with a short label, in JSON format. " \
        "i.e., <think> thinking process (step-by-step reasoning) here </think> " \
        "<answer>{Answer}</answer>"

    image = PILImage.open(args.image_path)
    image = image.convert("RGB")
    original_width, original_height = image.size
    resize_size = 840
    x_factor, y_factor = original_width / resize_size, original_height / resize_size

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
                    Question=args.text.lower().strip(".\"?!"),
                    Answer="[{\"label\": \"chair\", \"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110]}, "
                           "{\"label\": \"train track\", \"bbox_2d\": [225,296,706,786], \"point_2d\": [302,410]}]"
                )
            }
        ]
    }]
    messages = [message]

    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    if is_qwen3:
        inputs.pop("token_type_ids", None)

    # Inference
    generated_ids = reasoning_model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print(output_text[0])
    bboxes, points, think, labels = extract_bbox_points_think(
        output_text[0], x_factor, y_factor, is_qwen3=is_qwen3, image=image
    )
    print(f"Detected {len(points)} object(s): {labels}")
    print("Thinking process:", think)

    # SAM2 mask generation
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        mask_all = np.zeros((image.height, image.width), dtype=bool)
        segmentation_model.set_image(image)
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

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(image, alpha=0.6)
    mask_overlay = np.zeros_like(image)
    mask_overlay[mask_all] = [255, 0, 0]
    plt.imshow(mask_overlay, alpha=0.4)
    plt.title('Image with Predicted Mask')

    plt.tight_layout()
    plt.savefig(args.output_path)
    plt.close()
    print(f"Result saved to {args.output_path}")


if __name__ == "__main__":
    main()
