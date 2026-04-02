import json
import argparse
from PIL import Image as PILImage, ImageDraw
from pathlib import Path
from math import isfinite, sqrt, ceil
from tqdm import tqdm
import pdb
import os
import re
import random
import numpy as np
import time
from typing import List, Optional, Tuple
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
try:
    # Available in recent transformers; kept optional to avoid impacting Qwen2.5 flows.
    from transformers import Qwen3VLForConditionalGeneration
except Exception:
    Qwen3VLForConditionalGeneration = None
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
from qwen_vl_utils import process_vision_info

import torch
import torch.nn.functional as F
from dataclasses import asdict, dataclass
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

def str2bool(v):
    """
    Robust string-to-bool converter for argparse.
    Accepts: true/false, yes/no, y/n, 1/0 (case-insensitive).
    """
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("true", "t", "yes", "y", "1"):
        return True
    if v in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'.")

@dataclass
class MaskCandidate:
    """
    하나의 reasoning sample에서 나온 개별 object mask 후보.
    """
    mask: np.ndarray        # bool array [H, W]
    score: float
    sample_idx: int         # reasoning sample index (0..N-1)
    obj_idx: int            # 그 sample 내 object index
    bbox: list              # [x1,y1,x2,y2]
    point: list             # [x,y]
    label: str              # short label

def _infer_qwen3_grid_from_values(values):
    """
    Qwen3-VL grounding coords are typically normalized-like (0~1000 range).
    - max <= 1.2   -> [0,1] normalized
    - max <= 1000.5-> treat as 0~1000/0~999 style; use 999 denominator for compatibility
    - else         -> assume already pixels (no conversion)
    """
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
        # Use 999 as denominator (backward-compatible convention)
        return 999
    return None

def _scale_qwen3_coord_to_pixel(val, grid, size):
    """
    Convert Qwen3-VL coord to pixel index [0..size-1].
    grid=None means already pixels.
    grid=1 means [0,1] normalized.
    grid=999 means 0~1000-ish normalized but scaled with denom=999 for compatibility.
    """
    if grid is None:
        vv = round(val)
    elif grid == 1:
        vv = round(float(val) * (size - 1))
    else:
        # denom fixed to 999 (even if model sometimes emits 1000)
        vv = round(float(val) * (size - 1) / 999.0)
    return max(0, min(size - 1, int(vv)))

def _order_box_xyxy(x1, y1, x2, y2):
    left = min(x1, x2); right = max(x1, x2)
    top = min(y1, y2); bottom = max(y1, y2)
    return left, top, right, bottom

def _convert_qwen3_predictions_to_pixels(
    pred_bboxes,
    pred_points,
    image,
    default_w=840,
    default_h=840,
    force_grid=None,
    min_box_size=1,
):
    """
    Convert Qwen3-VL predicted (relative) coords to image pixel coords.
    Returns (boxes[M,4], points[M,2]) as int lists.
    """
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

    boxes_px = []
    points_px = []
    for i in range(len(pred_bboxes)):
        bx = pred_bboxes[i]
        pt = pred_points[i] if i < len(pred_points) else None

        if bx is not None and len(bx) == 4:
            x1 = _scale_qwen3_coord_to_pixel(bx[0], grid, W)
            y1 = _scale_qwen3_coord_to_pixel(bx[1], grid, H)
            x2 = _scale_qwen3_coord_to_pixel(bx[2], grid, W)
            y2 = _scale_qwen3_coord_to_pixel(bx[3], grid, H)
            x1, y1, x2, y2 = _order_box_xyxy(x1, y1, x2, y2)
            # ensure non-degenerate box
            if x2 <= x1:
                x2 = min(W - 1, x1 + min_box_size)
                if x2 <= x1 and W >= 2:
                    x1 = min(x1, W - 2); x2 = W - 1
            if y2 <= y1:
                y2 = min(H - 1, y1 + min_box_size)
                if y2 <= y1 and H >= 2:
                    y1 = min(y1, H - 2); y2 = H - 1
            boxes_px.append([int(x1), int(y1), int(x2), int(y2)])
        else:
            boxes_px.append([0, 0, 0, 0])

        if pt is not None and len(pt) == 2:
            px = _scale_qwen3_coord_to_pixel(pt[0], grid, W)
            py = _scale_qwen3_coord_to_pixel(pt[1], grid, H)
            points_px.append([int(px), int(py)])
        else:
            points_px.append([0, 0])

    return boxes_px, points_px

def _extract_answer_json(output_text: str) -> str:
    """
    Return the raw JSON string inside <answer>...</answer>.
    If missing, return "[]".
    """
    m = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return "[]"

def _safe_mkdir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)

def _sanitize_path_component(name: str) -> str:
    """
    Make a string safe for use as a single path component (folder/file stem).
    """
    if name is None:
        return "na"
    s = str(name).strip()
    if s == "":
        return "na"
    # replace common path separators
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = s.strip("._-")
    return s if s else "na"

def _get_sample_vis_dir(vis_root: str, ann_id, reasoning_type=None) -> str:
    """
    Per-sample visualization directory:
      {vis_root}/{ann_id}_{reasoning_type}
    """
    ann = _sanitize_path_component(ann_id)
    rt = _sanitize_path_component(reasoning_type)
    folder = f"{ann}_{rt}"
    out_dir = os.path.join(vis_root, folder)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def _get_eos_token_ids(processor, reasoning_model=None) -> List[int]:
    """
    Collect eos_token_id(s) from generation_config and tokenizer if available.
    """
    eos_ids: List[int] = []
    try:
        gen_cfg = getattr(reasoning_model, "generation_config", None) if reasoning_model is not None else None
        eos = getattr(gen_cfg, "eos_token_id", None) if gen_cfg is not None else None
        if eos is not None:
            if isinstance(eos, (list, tuple, set)):
                eos_ids.extend([int(x) for x in eos if x is not None])
            else:
                eos_ids.append(int(eos))
    except Exception:
        pass
    try:
        tok = getattr(processor, "tokenizer", None)
        eos = getattr(tok, "eos_token_id", None) if tok is not None else None
        if eos is not None:
            if isinstance(eos, (list, tuple, set)):
                eos_ids.extend([int(x) for x in eos if x is not None])
            else:
                eos_ids.append(int(eos))
    except Exception:
        pass
    eos_ids = list(dict.fromkeys([i for i in eos_ids if i is not None]))
    return eos_ids

def _count_generated_tokens(
    trimmed_ids: torch.Tensor,
    *,
    eos_token_ids: List[int],
    pad_token_id: Optional[int],
) -> int:
    """
    Count generated length in *tokens* for a single trimmed sequence.
    - Caller must pass trimmed ids (prompt removed).
    - Length is measured as index of first EOS (EOS excluded).
    - If EOS not found, strips trailing PAD tokens (if pad_token_id exists).
    """
    if trimmed_ids is None:
        return 0
    try:
        ids = trimmed_ids.tolist()
    except Exception:
        try:
            ids = list(trimmed_ids)
        except Exception:
            return 0
    if not ids:
        return 0

    eos_pos = None
    if eos_token_ids:
        for eid in eos_token_ids:
            try:
                p = ids.index(int(eid))
            except ValueError:
                continue
            if eos_pos is None or p < eos_pos:
                eos_pos = p
    if eos_pos is not None:
        return int(max(0, eos_pos))  # exclude eos itself

    end = len(ids)
    if pad_token_id is not None:
        pid = int(pad_token_id)
        while end > 0 and ids[end - 1] == pid:
            end -= 1
    return int(max(0, end))

def _overlay_mask_on_image(
    base_image: PILImage.Image,
    mask: np.ndarray,
    *,
    color: Tuple[int, int, int],
    alpha: float = 0.4,
    draw_empty_star: bool = False,
    empty_star_color: Tuple[int, int, int] = (0, 0, 0),
) -> PILImage.Image:
    """
    base_image 위에 mask를 반투명 overlay로 올린 RGB 이미지를 반환.
    """
    if base_image is None:
        raise ValueError("base_image is None")
    mask_bool = np.array(mask).astype(bool)
    if mask_bool.ndim != 2:
        raise ValueError("mask must be 2D")
    H, W = mask_bool.shape

    img = base_image.convert("RGBA")
    if img.size != (W, H):
        img = img.resize((W, H), PILImage.BILINEAR)

    overlay = PILImage.new("RGBA", (W, H), (0, 0, 0, 0))
    ov = np.array(overlay, dtype=np.uint8)
    r, g, b = [int(x) for x in color]
    a = int(max(0.0, min(1.0, float(alpha))) * 255.0)
    ov[mask_bool] = [r, g, b, a]
    overlay = PILImage.fromarray(ov, mode="RGBA")

    out = PILImage.alpha_composite(img, overlay).convert("RGB")
    if draw_empty_star and (not mask_bool.any()):
        draw = ImageDraw.Draw(out)
        _draw_star(draw, 10, 10, r=10, color=empty_star_color)
    return out

def _draw_bboxes_points_on_image(
    base_image: PILImage.Image,
    bboxes: List[list],
    points: Optional[List[list]] = None,
    *,
    box_color: Tuple[int, int, int] = (255, 0, 0),
    point_color: Tuple[int, int, int] = (0, 0, 255),
    width: int = 2,
) -> PILImage.Image:
    """
    base_image에 bbox + point(별)을 그려서 반환.
    bboxes[i]와 points[i]를 대응시킴.
    """
    img = base_image.convert("RGB").copy()
    W, H = img.size
    draw = ImageDraw.Draw(img)

    def _clip(v, lo, hi):
        try:
            iv = int(round(float(v)))
        except Exception:
            iv = lo
        return max(lo, min(hi, iv))

    if points is None:
        points = []

    for i, bbox in enumerate(bboxes):
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        x1 = _clip(bbox[0], 0, W - 1)
        y1 = _clip(bbox[1], 0, H - 1)
        x2 = _clip(bbox[2], 0, W - 1)
        y2 = _clip(bbox[3], 0, H - 1)
        if x2 <= x1 or y2 <= y1:
            continue
        # 기존 matplotlib(Rectangle) 방식과 맞추기 위해 x2/y2는 exclusive-like로 취급 -> -1
        draw.rectangle([x1, y1, max(x1, x2 - 1), max(y1, y2 - 1)], outline=box_color, width=int(width))

        if i < len(points):
            pt = points[i]
            if isinstance(pt, (list, tuple)) and len(pt) == 2:
                px = _clip(pt[0], 0, W - 1)
                py = _clip(pt[1], 0, H - 1)
                _draw_star(draw, px, py, r=8, color=point_color)
    return img

def _extract_answer_text(output_text: str) -> str:
    m = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""

def _parse_sc_verify_answer(output_text: str):
    """
    Binary verification parser for SC verify step.
    Returns: "accept" | "reject"
    NOTE: We intentionally default to 'accept' on any ambiguity to be conservative.
    """
    ans = _extract_answer_text(output_text)
    if not ans:
        return "accept"
    low = ans.strip().lower()
    if low in ("accept", "keep"):
        return "accept"
    if low in ("reject", "discard"):
        return "reject"
    # tolerate extra text; be conservative
    if low.startswith("reject") or low.startswith("discard"):
        return "reject"
    return "accept"

def _sc_gen_decode_multi(
    msgs,
    *,
    processor,
    reasoning_model,
    is_qwen3: bool,
    num_samples: int,
    gen_batch_size: int,
    sample_batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    """
    Generate num_samples responses PER message with controlled memory.
    - gen_batch_size: how many prompts per chunk (existing --sc_gen_batch_size)
    - sample_batch_size: how many sampled sequences per prompt per generate call
      (NEW: --sc_verify_sample_batch_size / --sc_missing_sample_batch_size)

    Returns:
      outputs_per_msg: List[List[str]] where len(outputs_per_msg)=len(msgs),
      and each inner list has length=num_samples (unless msgs is empty).
    """
    if msgs is None or len(msgs) == 0:
        return []
    n = int(num_samples)
    if n <= 0:
        n = 1
    gb = max(1, int(gen_batch_size))
    sb = max(1, int(sample_batch_size))
    do_sample = (n > 1)

    outputs_per_msg = [[] for _ in range(len(msgs))]

    for st in range(0, len(msgs), gb):
        chunk = msgs[st:st + gb]
        chunk_text = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in chunk]
        chunk_imgs, chunk_vids = process_vision_info(chunk)  # NOTE: user requested to keep process_vision_info
        chunk_inputs = processor(
            text=chunk_text,
            images=chunk_imgs,
            videos=chunk_vids,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        if is_qwen3:
            chunk_inputs.pop("token_type_ids", None)

        # prompt length in token space (includes left padding)
        prompt_len = int(chunk_inputs.input_ids.shape[1])

        # sample in rounds to control memory:
        # each round generates k sequences per prompt via num_return_sequences=k
        remaining = n
        while remaining > 0:
            k = min(sb, remaining)
            remaining -= k

            if do_sample:
                gen_ids = reasoning_model.generate(
                    **chunk_inputs,
                    use_cache=True,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=True,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    num_return_sequences=int(k),
                )
            else:
                # deterministic single output
                gen_ids = reasoning_model.generate(
                    **chunk_inputs,
                    use_cache=True,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=False,
                )

            # gen_ids shape:
            # - do_sample=False: [B, L]
            # - do_sample=True : [B*k, L]
            if do_sample:
                B = int(chunk_inputs.input_ids.shape[0])
                trimmed_ids_all = []
                for i in range(B * k):
                    trimmed_ids_all.append(gen_ids[i][prompt_len:])
                decoded_all = processor.batch_decode(
                    trimmed_ids_all, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                # group k outputs per prompt
                for b in range(B):
                    outs = decoded_all[b * k:(b + 1) * k]
                    outputs_per_msg[st + b].extend(outs)
            else:
                trimmed = [out_ids[prompt_len:] for out_ids in gen_ids]
                decoded_all = processor.batch_decode(
                    trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                for b in range(len(decoded_all)):
                    outputs_per_msg[st + b].append(decoded_all[b])

            del gen_ids
            torch.cuda.empty_cache()

        del chunk_inputs
        torch.cuda.empty_cache()

    return outputs_per_msg

def _draw_bbox_overlay_image(
    base_image: PILImage.Image,
    boxes: List[list],
    *,
    color: Tuple[int, int, int] = (0, 255, 0),
    width: int = 3,
    draw_index: bool = True,
) -> PILImage.Image:
    """
    Draw green bbox overlays on the *original image coordinate system*.
    - IMPORTANT: boxes must already be converted to the pixel coordinate system
      of base_image (i.e., after Qwen2.5 x/y scaling or Qwen3 grid->pixel conversion).
    - We intentionally do NOT implement mask overlay here.
    """
    if base_image is None:
        raise ValueError("base_image is None")
    img = base_image.convert("RGB").copy()
    W, H = img.size
    draw = ImageDraw.Draw(img)

    def _clip_int(v, lo, hi):
        try:
            iv = int(round(float(v)))
        except Exception:
            iv = lo
        return max(lo, min(hi, iv))

    for idx, box in enumerate(boxes, start=1):
        if not (isinstance(box, (list, tuple)) and len(box) == 4):
            continue
        x1 = _clip_int(box[0], 0, W - 1)
        y1 = _clip_int(box[1], 0, H - 1)
        x2 = _clip_int(box[2], 0, W - 1)
        y2 = _clip_int(box[3], 0, H - 1)
        if x2 <= x1 or y2 <= y1:
            continue
        draw.rectangle([x1, y1, x2, y2], outline=color, width=int(width))
        if draw_index:
            draw.text((x1 + 2, y1 + 2), str(idx), fill=color)
    return img

def _draw_star(draw: ImageDraw.ImageDraw, x: int, y: int, *, r: int = 8, color=(255, 0, 0)):
    """
    Draw a simple 5-point star polygon centered at (x, y).
    """
    import math
    pts = []
    # 10 vertices: outer/inner alternating
    for i in range(10):
        ang = math.radians(-90 + i * 36)  # start upwards
        rr = r if i % 2 == 0 else max(2, int(r * 0.45))
        pts.append((x + rr * math.cos(ang), y + rr * math.sin(ang)))
    draw.polygon(pts, fill=color)

def _draw_bbox_point_overlay_image(
    base_image: PILImage.Image,
    bbox: List[int],
    point: List[int],
    *,
    box_color=(0, 255, 0),
    point_color=(255, 0, 0),
    width: int = 3,
) -> PILImage.Image:
    """
    Draw ONE bbox (green) + ONE point (red star) on base_image.
    bbox/point are in base_image pixel coordinate system.
    """
    img = base_image.convert("RGB").copy()
    W, H = img.size
    draw = ImageDraw.Draw(img)

    def _clip(v, lo, hi):
        try:
            iv = int(round(float(v)))
        except Exception:
            iv = lo
        return max(lo, min(hi, iv))

    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x1 = _clip(bbox[0], 0, W - 1)
        y1 = _clip(bbox[1], 0, H - 1)
        x2 = _clip(bbox[2], 0, W - 1)
        y2 = _clip(bbox[3], 0, H - 1)
        if x2 > x1 and y2 > y1:
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=int(width))

    if isinstance(point, (list, tuple)) and len(point) == 2:
        px = _clip(point[0], 0, W - 1)
        py = _clip(point[1], 0, H - 1)
        _draw_star(draw, px, py, r=8, color=point_color)

    return img

def _bbox_area_ratio(bbox: List[int], W: int, H: int) -> float:
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return 1.0
    x1, y1, x2, y2 = bbox
    bw = max(0, int(x2) - int(x1))
    bh = max(0, int(y2) - int(y1))
    denom = max(1, int(W) * int(H))
    return float(bw * bh) / float(denom)

def _mask_to_bbox_point(mask: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Convert boolean mask to a tight bbox (xyxy) and a point (centroid-ish).
    We use x2/y2 = max + 1 style (same convention used elsewhere in MV bbox derivation).
    """
    ys, xs = np.where(mask.astype(bool))
    if ys.size == 0:
        return [0, 0, 0, 0], [0, 0]
    x1 = int(xs.min()); x2 = int(xs.max()) + 1
    y1 = int(ys.min()); y2 = int(ys.max()) + 1
    px = int(np.mean(xs)); py = int(np.mean(ys))
    return [x1, y1, x2, y2], [px, py]

def _expand_bbox_for_cover_ratio(bbox: List[int], W: int, H: int, cover_ratio: float = 0.8) -> List[int]:
    """
    Expand bbox so that original bbox area occupies ~cover_ratio of the crop area.
    If we expand width/height by scale s, area scales by s^2.
      cover_ratio ~= 1 / s^2  -> s ~= sqrt(1/cover_ratio)
    """
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return [0, 0, W - 1, H - 1]
    cover_ratio = float(cover_ratio)
    cover_ratio = min(0.95, max(0.05, cover_ratio))
    s = sqrt(1.0 / cover_ratio)

    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1, x2, y2 = _order_box_xyxy(x1, y1, x2, y2)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = max(2.0, float(x2 - x1))
    h = max(2.0, float(y2 - y1))
    nw = w * s
    nh = h * s

    nx1 = int(round(cx - 0.5 * nw))
    ny1 = int(round(cy - 0.5 * nh))
    nx2 = int(round(cx + 0.5 * nw))
    ny2 = int(round(cy + 0.5 * nh))
    nx1 = max(0, min(W - 2, nx1))
    ny1 = max(0, min(H - 2, ny1))
    nx2 = max(nx1 + 1, min(W - 1, nx2))
    ny2 = max(ny1 + 1, min(H - 1, ny2))
    return [nx1, ny1, nx2, ny2]

def _make_small_bbox_crop_overlay(
    base_image: PILImage.Image,
    bbox_px: List[int],
    point_px: List[int],
    *,
    cover_ratio: float = 0.8,
) -> PILImage.Image:
    """
    Make a crop image where bbox occupies ~cover_ratio of the crop, then overlay bbox+point.
    """
    W, H = base_image.size
    crop_box = _expand_bbox_for_cover_ratio(bbox_px, W, H, cover_ratio=cover_ratio)
    cx1, cy1, cx2, cy2 = crop_box
    crop = base_image.convert("RGB").crop((cx1, cy1, cx2, cy2))
    # translate bbox/point to crop coordinates
    bb_rel = [bbox_px[0] - cx1, bbox_px[1] - cy1, bbox_px[2] - cx1, bbox_px[3] - cy1]
    pt_rel = [point_px[0] - cx1, point_px[1] - cy1]
    return _draw_bbox_point_overlay_image(crop, bb_rel, pt_rel)

def _coords_px_to_model_coords_str(
    bbox_px: List[int],
    point_px: List[int],
    *,
    is_qwen3: bool,
    image: PILImage.Image,
    x_factor: float,
    y_factor: float,
    qwen3_force_grid: int = -1,
) -> Tuple[str, str]:
    """
    Provide coordinates in the model's expected convention:
      - Qwen2.5-VL path: coords are in resized(=resize_size) pixel space (we invert x_factor/y_factor)
      - Qwen3-VL path: coords are typically in 0~1000-ish grid (we map pixels -> 0..999) unless forced to [0,1]
    """
    if image is not None:
        W, H = image.size
    else:
        W, H = 840, 840

    if not is_qwen3:
        bb = [
            int(round(float(bbox_px[0]) / max(1e-6, x_factor))),
            int(round(float(bbox_px[1]) / max(1e-6, y_factor))),
            int(round(float(bbox_px[2]) / max(1e-6, x_factor))),
            int(round(float(bbox_px[3]) / max(1e-6, y_factor))),
        ]
        pt = [
            int(round(float(point_px[0]) / max(1e-6, x_factor))),
            int(round(float(point_px[1]) / max(1e-6, y_factor))),
        ]
        return str(bb), str(pt)

    # qwen3
    if qwen3_force_grid == 1:
        bb = [
            round(float(bbox_px[0]) / max(1.0, (W - 1)), 4),
            round(float(bbox_px[1]) / max(1.0, (H - 1)), 4),
            round(float(bbox_px[2]) / max(1.0, (W - 1)), 4),
            round(float(bbox_px[3]) / max(1.0, (H - 1)), 4),
        ]
        pt = [
            round(float(point_px[0]) / max(1.0, (W - 1)), 4),
            round(float(point_px[1]) / max(1.0, (H - 1)), 4),
        ]
        return str(bb), str(pt)

    # default to 0..999 style
    denom = 999.0
    bb = [
        int(round(float(bbox_px[0]) * denom / max(1.0, (W - 1)))),
        int(round(float(bbox_px[1]) * denom / max(1.0, (H - 1)))),
        int(round(float(bbox_px[2]) * denom / max(1.0, (W - 1)))),
        int(round(float(bbox_px[3]) * denom / max(1.0, (H - 1)))),
    ]
    pt = [
        int(round(float(point_px[0]) * denom / max(1.0, (W - 1)))),
        int(round(float(point_px[1]) * denom / max(1.0, (H - 1)))),
    ]
    return str(bb), str(pt)

def _convert_model_coords_to_pixels_single(
    bbox_2d,
    point_2d,
    *,
    is_qwen3: bool,
    image: PILImage.Image,
    x_factor: float,
    y_factor: float,
    qwen3_force_grid: int = -1,
) -> Tuple[List[int], List[int]]:
    """
    Convert a single bbox/point from model coordinate to image pixel coordinate.
    """
    if not (isinstance(bbox_2d, (list, tuple)) and len(bbox_2d) == 4):
        return [0, 0, 0, 0], [0, 0]
    if not (isinstance(point_2d, (list, tuple)) and len(point_2d) == 2):
        point_2d = [0, 0]

    if not is_qwen3:
        bb = [
            int(float(bbox_2d[0]) * x_factor + 0.5),
            int(float(bbox_2d[1]) * y_factor + 0.5),
            int(float(bbox_2d[2]) * x_factor + 0.5),
            int(float(bbox_2d[3]) * y_factor + 0.5),
        ]
        pt = [
            int(float(point_2d[0]) * x_factor + 0.5),
            int(float(point_2d[1]) * y_factor + 0.5),
        ]
        bb = list(_order_box_xyxy(bb[0], bb[1], bb[2], bb[3]))
        return [int(v) for v in bb], [int(v) for v in pt]

    # qwen3
    force_grid = None
    if qwen3_force_grid == 0:
        force_grid = None
    elif qwen3_force_grid in (1, 999, 1000):
        force_grid = 1 if qwen3_force_grid == 1 else 999

    boxes_px, points_px = _convert_qwen3_predictions_to_pixels(
        [bbox_2d], [point_2d], image,
        default_w=840, default_h=840,
        force_grid=force_grid,
    )
    return boxes_px[0], points_px[0]

def _maybe_save_sc_verify_inputs(
    *,
    output_path: str,
    enable: bool,
    ann_id: str,
    image_id: str,
    turn: int,
    bbox_idx: int,
    global_img: PILImage.Image,
    crop_img: Optional[PILImage.Image] = None,
    missing_overlay_img: Optional[PILImage.Image] = None,
):
    if not enable:
        return
    root = os.path.join(output_path, "visualization", "sc_verify_inputs", f"ann{ann_id}_img{image_id}")
    _safe_mkdir(root)
    if global_img is not None:
        global_img.save(os.path.join(root, f"t{turn}_bbox{bbox_idx:02d}_global.png"))
    if crop_img is not None:
        crop_img.save(os.path.join(root, f"t{turn}_bbox{bbox_idx:02d}_crop.png"))
    if missing_overlay_img is not None:
        missing_overlay_img.save(os.path.join(root, f"t{turn}_missing_overlay.png"))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning_model_path", type=str, default="Ricky06662/Seg-Zero-7B")
    parser.add_argument("--segmentation_model_path", type=str, default="facebook/sam2-hiera-large")
    parser.add_argument("--use_lora", type=str2bool, default=True)
    parser.add_argument("--visualization", type=str2bool, default=False)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--num_parts", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--test_refcoco", type=str2bool, default=False) 
    parser.add_argument("--test_muse_mmr", type=str2bool, default=False)
    # ---------------------------------------------------
    # Majority voting (parallel test-time scaling)
    # ---------------------------------------------------
    parser.add_argument("--use_majority_voting", type=str2bool, default=False,
                        help="Enable majority-voting reasoning segmentation.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling when using majority voting.")
    parser.add_argument("--num_samples", type=int, default=32,
                        help="Number of reasoning samples per example for majority voting.")
    parser.add_argument("--no_object_vote_threshold", type=float, default=0.5,
                        help="If no-object ratio among valid samples >= this, final prediction is no-object.")
    parser.add_argument("--mask_iou_cluster_threshold", type=float, default=0.85,
                        help="IoU threshold for clustering masks into same-object groups.")
    parser.add_argument("--cluster_vote_threshold", type=float, default=0.2,
                        help="Min vote ratio for a cluster to be considered (fallback to all clusters if empty).")
    parser.add_argument("--pixel_majority_threshold", type=float, default=0.5,
                        help="Pixel-wise consensus threshold within a cluster for consensus aggregation.")
    parser.add_argument("--cluster_agg_mode", type=str, default="best",
                        choices=["consensus", "best"],
                        help="How to aggregate masks within a cluster: consensus (weighted majority) or best single mask.")
    parser.add_argument("--sampling_temperature", type=float, default=1.0,
                        help="Temperature for sampling when use_majority_voting=True.")
    parser.add_argument("--sampling_top_p", type=float, default=0.9,
                        help="Top-p for sampling when use_majority_voting=True.")
    parser.add_argument("--enable_hparam_sweep", type=str2bool, default=False,
                        help="If true (and use_majority_voting is true), reuse the same "
                             "MLLM + SAM outputs to evaluate multiple majority-voting "
                             "hyper-parameter combinations and print gIoU / cIoU.")
    # ---------------------------------------------------
    # Self-correction (sequential test-time scaling)
    # ---------------------------------------------------
    parser.add_argument("--use_self_correction", type=str2bool, default=False,
                        help="Enable sequential self-correction (self-refine) at test time.")
    parser.add_argument("--sc_num_turns", type=int, default=1,
                        help="Number of self-correction turns to run (>=1).")
    parser.add_argument("--sc_include_think", type=str2bool, default=False,
                        help="If true, include previous think/reasoning text in the self-correction prompt.")
    parser.add_argument("--sc_max_new_tokens", type=int, default=1024,
                        help="Max new tokens for each self-correction generation step.")
    parser.add_argument("--sc_small_bbox_area_ratio", type=float, default=0.05,
                        help="Define 'small bbox' as bbox_area/(W*H) <= this. Small bbox uses 2-image verification.")
    parser.add_argument("--sc_crop_bbox_cover_ratio", type=float, default=0.8,
                        help="For small bbox verification crop: make bbox occupy ~this ratio of the crop area.")
    parser.add_argument("--sc_save_verify_inputs", type=str2bool, default=False,
                        help="Save verification prompt input images under output_path/visualization/sc_verify_inputs.")
    parser.add_argument("--sc_gen_batch_size", type=int, default=32,
                        help="Chunk size for SC verification/missing model generation to avoid OOM.")
    parser.add_argument("--sc_verify_num_samples", type=int, default=4,
                        help="Number of sampled responses per bbox verification prompt. "
                             "A bbox is rejected ONLY if ALL samples say reject.")
    parser.add_argument("--sc_missing_num_samples", type=int, default=4,
                        help="Number of sampled responses for missing-object step.")
    parser.add_argument("--sc_verify_sample_batch_size", type=int, default=32,
                        help="num_return_sequences per generate call for verification (controls VRAM).")
    parser.add_argument("--sc_missing_sample_batch_size", type=int, default=32,
                        help="num_return_sequences per generate call for missing step (controls VRAM).")
    parser.add_argument("--sc_sampling_temperature", type=float, default=1.0,
                        help="Sampling temperature used for SC multi-sampling.")
    parser.add_argument("--sc_sampling_top_p", type=float, default=0.9,
                        help="Top-p used for SC multi-sampling.")
    parser.add_argument("--sc_missing_empty_vote_threshold", type=float, default=0.5,
                        help="If empty(missing=[]) vote ratio >= this, finalize missing as empty. "
                             "Default 0.5 matches '>= half => empty'.")
    # ---------------------------------------------------
    # Qwen3-VL support
    # ---------------------------------------------------
    parser.add_argument("--vl_model_version", type=str, default="qwen2_5",
                        choices=["qwen2_5", "qwen3"],
                        help="Select VL backbone. Default keeps Qwen2.5 behavior unchanged.")
    parser.add_argument("--qwen3_base_path", type=str, default="pretrained_models/Qwen3-VL-8B-Instruct",
                        help="Base pretrained path for Qwen3-VL when --use_lora true.")
    parser.add_argument("--qwen3_force_coord_grid", type=int, default=-1,
                        help="Force Qwen3 coord grid: -1=auto, 1=[0,1], 999=0~1000 normalized, 0=pixels(no-conv).")
    return parser.parse_args()

def run_ensemble_with_params(
    mask_candidates: List[MaskCandidate],
    object_counts: List[int],
    no_object_votes: int,
    valid_runs: int,
    gt_mask: np.ndarray,
    no_object_vote_threshold: float,
    mask_iou_cluster_threshold: float,
    cluster_vote_threshold: float,
    pixel_majority_threshold: float,
    cluster_agg_mode: str,
) -> Tuple[int, int]:
    """
    하나의 sample에 대해 주어진 hyper-parameter 세트로
    majority voting 앙상블을 한 번 실행하고
    (intersection, union)만 리턴한다.
    기존 majority voting 구현과 동일한 흐름을 따른다.
    """
    if valid_runs == 0:
        intersection = 0
        union = int(gt_mask.sum())
        return int(intersection), int(union)

    no_obj_ratio = no_object_votes / valid_runs
    if no_obj_ratio >= no_object_vote_threshold:
        final_mask_all = np.zeros_like(gt_mask, dtype=bool)
        intersection, union = compute_iou(final_mask_all, gt_mask)
        return int(intersection), int(union)

    positive_counts = [c for c in object_counts if c > 0]
    if len(positive_counts) == 0:
        K_hat = 0
    else:
        counts_counter = Counter(positive_counts)
        K_hat = counts_counter.most_common(1)[0][0]

    if K_hat <= 0 or len(mask_candidates) == 0:
        final_mask_all = np.zeros_like(gt_mask, dtype=bool)
        intersection, union = compute_iou(final_mask_all, gt_mask)
        return int(intersection), int(union)

    cluster_infos = cluster_mask_candidates(
        mask_candidates,
        iou_thr=mask_iou_cluster_threshold,
        num_samples=valid_runs,
    )
    if len(cluster_infos) == 0:
        final_mask_all = np.zeros_like(gt_mask, dtype=bool)
        intersection, union = compute_iou(final_mask_all, gt_mask)
        return int(intersection), int(union)

    filtered = [
        c for c in cluster_infos
        if c["vote_ratio"] >= cluster_vote_threshold
    ]
    if len(filtered) == 0:
        filtered = cluster_infos

    cluster_infos_sorted = sorted(
        filtered,
        key=lambda c: (c["vote_count"], c["avg_score"]),
        reverse=True,
    )
    num_to_select = min(len(cluster_infos_sorted), K_hat)
    selected = cluster_infos_sorted[:num_to_select]

    final_masks: List[np.ndarray] = []
    for cinfo in selected:
        m_star, _ = aggregate_cluster_mask(
            cinfo,
            mask_candidates,
           mode=cluster_agg_mode,
            pixel_thr=pixel_majority_threshold,
        )
        if m_star is None:
            continue
        final_masks.append(m_star)

    if len(final_masks) == 0:
        final_mask_all = np.zeros_like(gt_mask, dtype=bool)
    else:
        final_mask_all = np.zeros_like(gt_mask, dtype=bool)
        for fm in final_masks:
            final_mask_all = np.logical_or(final_mask_all, fm)

    intersection, union = compute_iou(final_mask_all, gt_mask)
    return int(intersection), int(union)

def extract_bbox_points_think(output_text, x_factor, y_factor, *, is_qwen3=False, image=None, qwen3_force_grid=-1):
    think_pattern = r'<think>\s*(.*?)\s*</think>'
    think_text = ""
    think_match = re.search(think_pattern, output_text, re.DOTALL)
    if think_match:
        think_text = think_match.group(1)
    
    # Qwen3-VL: mostly doesn't emit <think>...</think>.
    # If <think> is missing but <answer> exists, treat everything before <answer> as think_text.
    # (If <think> exists, keep the original behavior.)
    if is_qwen3 and (not think_text):
        ans_start = re.search(r'<answer>\s*', output_text, re.DOTALL)
        if ans_start:
            prefix = output_text[:ans_start.start()]
            # strip potential leading BOS/special artifacts
            think_text = prefix.strip()
        
    json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text, re.DOTALL)
    if not json_match:
        pred_bboxes, pred_points = [], []
        label_texts = []
        return pred_bboxes, pred_points, think_text, label_texts

    raw = json_match.group(1)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # fixed = re.sub(r'}}(\s*])', r'}\1', raw)
        fixed = raw
        
        # Fix malformed point_2d endings like:
        #   "point_2d": [x, y}}]  -> "point_2d": [x, y]}]
        #   "point_2d": [x, y}]   -> "point_2d": [x, y]}]
        # (case 1) [x, y}}...] : replace the FIRST '}' (array-close typo) with ']'
        fixed = re.sub(r'("point_2d"\s*:\s*\[[^\]]*?)\}\s*(\})', r'\1]\2', fixed)
        # (case 2) [x, y}] : insert missing ']' before the object-closing '}'
        fixed = re.sub(r'("point_2d"\s*:\s*\[[^\]]*?)\}\s*(])', r'\1]}\2', fixed)

        # Existing fix: extra '}' before list close
        fixed = re.sub(r'}}(\s*])', r'}\1', fixed)
        
        def _strip_quotes_in_array(match):
            inner = match.group(2).replace('"', '')
            return match.group(1) + inner + match.group(3)
        
        fixed = re.sub(r'("bbox_2d"\s*:\s*\[)([^\]]*)(\])', _strip_quotes_in_array, fixed)
        fixed = re.sub(r'("point_2d"\s*:\s*\[)([^\]]*)(\])', _strip_quotes_in_array, fixed)

        fixed = re.sub(r'("bbox_2d"\s*:\s*\[[^\]]*\])"', r'\1', fixed)
        fixed = re.sub(r'("point_2d"\s*:\s*\[[^\]]*\])"', r'\1', fixed)
        data = json.loads(fixed)

    if raw == "" or (isinstance(data, list) and len(data) == 0) or (isinstance(data, list) and len(data) == 1 and len(data[0]) == 0):
        pred_bboxes, pred_points = [], []
        label_texts = []
        return pred_bboxes, pred_points, think_text, label_texts
    
    if json_match:
        if is_qwen3:
            raw_bboxes = [item.get("bbox_2d", None) for item in data]
            raw_points = [item.get("point_2d", None) for item in data]
            # sanitize: keep only list-like entries
            raw_bboxes = [b for b in raw_bboxes if isinstance(b, (list, tuple)) and len(b) == 4]
            raw_points = [p for p in raw_points if isinstance(p, (list, tuple)) and len(p) == 2]
            force_grid = None
            if qwen3_force_grid == 0:
                force_grid = None
            elif qwen3_force_grid in (1, 999, 1000):
                # NOTE: internal scaling uses 999 denom for compatibility.
                force_grid = 999 if qwen3_force_grid != 1 else 1
            pred_bboxes, pred_points = _convert_qwen3_predictions_to_pixels(
                raw_bboxes, raw_points, image,
                default_w=840, default_h=840,
                force_grid=force_grid,
            )
        else:
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


def _save_sc_mask_delta_viz(
    *,
    base_image: PILImage.Image,
    query_text: str,
    out_path: str,
    orig_mask: np.ndarray,
    sc_mask: np.ndarray,
    orig_iou: float,
    sc_iou: float,
):
    """
    Save a single PNG:
      - Left : original prediction mask overlay
      - Right: self-correction prediction mask overlay
      - Top  : query text
      - Bottom: IoU numbers
    """
    orig_mask = np.array(orig_mask).astype(bool)
    sc_mask = np.array(sc_mask).astype(bool)

    # masks must have same shape for fair visualization
    if orig_mask.shape != sc_mask.shape:
        return
    H, W = orig_mask.shape

    img = base_image.convert("RGB")
    if img.size != (W, H):
        # ensure base image matches mask resolution
        img = img.resize((W, H), PILImage.BILINEAR)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=150)

    def _draw(ax, mask: np.ndarray, title: str):
        ax.imshow(img)
        overlay = np.zeros((H, W, 4), dtype=np.float32)
        # green transparent overlay
        overlay[mask] = [0.0, 1.0, 0.0, 0.35]
        ax.imshow(overlay)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    _draw(axes[0], orig_mask, "Original")
    _draw(axes[1], sc_mask, "Self-correction")

    fig.suptitle(query_text, fontsize=8)
    fig.text(0.25, 0.03, f"Original IoU: {orig_iou:.3f}", ha="center", va="bottom", fontsize=9)
    fig.text(0.75, 0.03, f"Self-corrected IoU: {sc_iou:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout(rect=[0, 0.06, 1, 0.92])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _maybe_save_sc_delta_case(
    *,
    output_path: str,
    base_image: PILImage.Image,
    query_text: str,
    ann_id: str,
    image_id: str,
    turn: int,
    orig_mask: np.ndarray,
    sc_mask: np.ndarray,
    orig_inter: int,
    orig_union: int,
    sc_inter: int,
    sc_union: int,
    delta_thr: float = 0.1,
):
    """
    Save visualization only when |IoU_sc - IoU_orig| >= delta_thr.
    Folder:
      {output_path}/visualization/increase_samples
      {output_path}/visualization/decrease_samples
    """
    orig_iou = (float(orig_inter) / float(orig_union)) if orig_union > 0 else 1.0
    sc_iou = (float(sc_inter) / float(sc_union)) if sc_union > 0 else 1.0
    delta = sc_iou - orig_iou
    if abs(delta) < float(delta_thr):
        return

    vis_root = os.path.join(output_path, "visualization")
    inc_dir = os.path.join(vis_root, "increase_samples")
    dec_dir = os.path.join(vis_root, "decrease_samples")
    out_dir = inc_dir if delta >= 0 else dec_dir
    tag = "inc" if delta >= 0 else "dec"

    fname = f"{tag}_t{int(turn)}_ann{ann_id}_img{image_id}_d{abs(delta):.3f}.png"
    out_path = os.path.join(out_dir, fname)

    _save_sc_mask_delta_viz(
        base_image=base_image,
        query_text=query_text,
        out_path=out_path,
        orig_mask=orig_mask,
        sc_mask=sc_mask,
        orig_iou=orig_iou,
        sc_iou=sc_iou,
    )


def cluster_mask_candidates(mask_candidates: List[MaskCandidate],
                            iou_thr: float,
                            num_samples: int):
    """
    Greedy IoU 기반으로 mask 후보들을 cluster로 묶고,
    각 cluster에 대한 통계(vote_count, vote_ratio, avg_score 등)를 리턴.
    """
    clusters = []
    for idx, cand in enumerate(mask_candidates):
        assigned = False
        for cluster in clusters:
            rep_idx = cluster["member_indices"][0]
            rep = mask_candidates[rep_idx]
            inter, union = compute_iou(cand.mask, rep.mask)
            iou = 0.0 if union == 0 else inter / union
            if iou >= iou_thr:
                cluster["member_indices"].append(idx)
                cluster["sample_ids"].add(cand.sample_idx)
                assigned = True
                break
        if not assigned:
            clusters.append({
                "member_indices": [idx],
                "sample_ids": {cand.sample_idx},
            })

    cluster_infos = []
    for cid, cluster in enumerate(clusters):
        members = [mask_candidates[i] for i in cluster["member_indices"]]
        vote_count = len(cluster["sample_ids"])
        scores = [m.score for m in members]
        areas = [m.mask.sum() for m in members]
        labels = [m.label for m in members if m.label is not None]
        label_counts = Counter(labels)
        dominant_label = label_counts.most_common(1)[0][0] if label_counts else ""
        cluster_infos.append({
            "cluster_id": cid,
            "member_indices": cluster["member_indices"],
            "vote_count": vote_count,
            "vote_ratio": vote_count / max(1, num_samples),
            "avg_score": float(np.mean(scores) if scores else 0.0),
            "avg_area": float(np.mean(areas) if areas else 0.0),
            "dominant_label": dominant_label,
        })
    return cluster_infos


def aggregate_cluster_mask(cluster_info,
                           mask_candidates: List[MaskCandidate],
                           mode: str = "consensus",
                           pixel_thr: float = 0.5) -> Tuple[np.ndarray, str]:
    """
    하나의 클러스터 내 여러 mask 후보들을 하나의 최종 mask로 aggregate.
    mode == "best": 점수/면적이 가장 큰 mask 하나 선택 (NMS 비슷한 전략).
    mode == "consensus": SAM score 를 weight로 한 pixel-wise weighted voting.
    """
    member_indices = cluster_info["member_indices"]
    members = [mask_candidates[i] for i in member_indices]
    if not members:
        return None, ""

    labels = [m.label for m in members if m.label is not None]
    label_counts = Counter(labels)
    dominant_label = label_counts.most_common(1)[0][0] if label_counts else ""

    if mode == "best":
        best = max(members, key=lambda m: (m.score, m.mask.sum()))
        return best.mask, dominant_label

    masks = np.stack([m.mask for m in members], axis=0).astype(np.float32)
    scores = np.array([m.score for m in members], dtype=np.float32)
    if np.all(scores <= 0):
        scores = np.ones_like(scores)
    weights = scores / scores.sum()
    prob = np.tensordot(weights, masks, axes=(0, 0))  # [H, W]
    final_mask = prob >= float(pixel_thr)
    return final_mask, dominant_label

def _sc_select_missing_by_mv(
    sampled_texts: List[str],
    *,
    meta: dict,
    segmentation_model,
    is_qwen3: bool,
    resize_size: int,
    qwen3_force_grid: int,
    empty_vote_threshold: float,
    mv_iou_thr: float,
    mv_cluster_vote_thr: float,
    mv_pixel_thr: float,
    mv_mode: str,
) -> List[dict]:
    """
    Missing-step majority strategy:
    1) Run N sampled missing predictions.
    2) If empty votes >= threshold (>=50% by default), return [].
    3) Else, build MaskCandidate pool from sampled predictions (like MV),
       cluster+aggregate (reuse MV logic) to select final missing masks.

    Returns:
      selected_missing: list of dict
        {"mask": bool[H,W], "score": float, "label": str, "bbox": [x1,y1,x2,y2], "point":[x,y]}
    """
    if sampled_texts is None or len(sampled_texts) == 0:
        return []

    n = int(len(sampled_texts))
    # Pass 1: parse only (cheap) to decide whether to early-return as empty
    parsed = []  # list of (bboxes, points, labels)
    empty_votes = 0
    for out_text in sampled_texts:
        try:
            bxs, pts, _, lbls = extract_bbox_points_think(
                out_text,
                meta["img_width"] / resize_size,
                meta["img_height"] / resize_size,
                is_qwen3=is_qwen3,
                image=meta["image"],
                qwen3_force_grid=qwen3_force_grid,
            )
        except Exception:
            bxs, pts, lbls = [], [], []
        if len(bxs) == 0:
            empty_votes += 1
        parsed.append((bxs, pts, lbls))

    # empty vote threshold (>= 50% by default; for N=8 => 4+ empties => empty)
    thr_cnt = int(ceil(float(empty_vote_threshold) * float(n)))
    if empty_votes >= thr_cnt:
        return []

    # Pass 2: build candidates + MV selection (same spirit as main MV code)
    mask_candidates: List[MaskCandidate] = []
    object_counts: List[int] = []
    no_object_votes = 0
    valid_runs = 0

    for run_idx, (bxs, pts, lbls) in enumerate(parsed):
        valid_runs += 1
        object_counts.append(len(bxs))
        if len(bxs) == 0:
            no_object_votes += 1
            continue
        for obj_idx, (bbox, point) in enumerate(zip(bxs, pts)):
            try:
                masks, scores, _ = segmentation_model.predict(
                    point_coords=[point],
                    point_labels=[1],
                    box=bbox,
                )
                sorted_ind = np.argsort(scores)[::-1]
                masks = masks[sorted_ind]
                scores = scores[sorted_ind]
                m0 = masks[0].astype(bool)
                s0 = float(scores[0])
                lbl = lbls[obj_idx] if obj_idx < len(lbls) else ""
                mask_candidates.append(
                    MaskCandidate(
                        mask=m0,
                        score=s0,
                        sample_idx=run_idx,
                        obj_idx=obj_idx,
                        bbox=bbox,
                        point=point,
                        label=lbl,
                    )
                )
            except Exception:
                # keep going; conservative behavior is handled by vote logic
                continue

    # If somehow degenerate, return empty
    if valid_runs == 0 or len(mask_candidates) == 0:
        return []

    # Determine K_hat (majority object count among positive runs)
    positive_counts = [c for c in object_counts if c > 0]
    if len(positive_counts) == 0:
        return []
    counts_counter = Counter(positive_counts)
    K_hat = int(counts_counter.most_common(1)[0][0])
    if K_hat <= 0:
        return []

    cluster_infos = cluster_mask_candidates(mask_candidates, iou_thr=float(mv_iou_thr), num_samples=int(valid_runs))
    if len(cluster_infos) == 0:
        return []

    filtered = [c for c in cluster_infos if c["vote_ratio"] >= float(mv_cluster_vote_thr)]
    if len(filtered) == 0:
        filtered = cluster_infos

    cluster_infos_sorted = sorted(filtered, key=lambda c: (c["vote_count"], c["avg_score"]), reverse=True)
    selected_infos = cluster_infos_sorted[: min(len(cluster_infos_sorted), K_hat)]

    selected_missing: List[dict] = []
    for cinfo in selected_infos:
        m_star, dom_lbl = aggregate_cluster_mask(
            cinfo, mask_candidates, mode=str(mv_mode), pixel_thr=float(mv_pixel_thr)
        )
        if m_star is None:
            continue
        bbox_px, point_px = _mask_to_bbox_point(m_star)
        selected_missing.append(
            {
                "mask": m_star.astype(bool),
                "score": float(cinfo.get("avg_score", 1.0)),
                "label": str(dom_lbl) if dom_lbl is not None else "",
                "bbox": bbox_px,
                "point": point_px,
            }
        )

    return selected_missing


def main():
    args = parse_args()
    
    is_reasonsegx = ("ReasonSegX" in str(args.test_data_path))
    
    # 시드 고정 (MLLM sampling을 고정해서, 같은 run에서 hyper-param만 바뀌도록)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    #We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    if args.vl_model_version == "qwen3":
        if Qwen3VLForConditionalGeneration is None:
            raise ImportError(
                "Qwen3VLForConditionalGeneration not found in your transformers. "
                "Please upgrade transformers to a version that includes Qwen3-VL."
            )
        base_path = args.qwen3_base_path if args.use_lora else args.reasoning_model_path
        reasoning_model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    else:
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
        # from verl.peft.get_peft_model import get_peft_model
        # from verl.peft.tuners.config import LoraConfig, LoraRuntimeConfig
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
    # if args.vl_model_version == "qwen3" and args.use_lora:
    #     # adapter dir may not contain processor configs reliably; use base
    #     processor = AutoProcessor.from_pretrained(args.qwen3_base_path, padding_side="left")
    # else:
    processor = AutoProcessor.from_pretrained(args.reasoning_model_path, padding_side="left")
    
    eos_token_ids = _get_eos_token_ids(processor, reasoning_model)
    pad_token_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", None)
    
    resize_size = 840
    try:
        dataset = load_from_disk(args.test_data_path)
    except:
        try:
            dataset = load_dataset(args.test_data_path, split='test')
        except:
            dataset = load_dataset(args.test_data_path, split='train')
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
        
    # ---------------------------------------------------
    # Majority-voting hyper-parameter sweep grid 설정
    # (use_majority_voting & enable_hparam_sweep 인 경우에만 사용)
    # ---------------------------------------------------
    sweep_configs = []
    sweep_stats = {}
    if args.use_majority_voting and args.enable_hparam_sweep:
        # 합리적인 기본 탐색 범위
        # - no_object_vote_threshold: no-object 과반 여부
        noobj_list = [0.5, 0.6, 0.7]
        # - mask_iou_cluster_threshold: 같은 객체로 묶을 IoU 기준 (높은 값 위주)
        mask_iou_list = [0.75, 0.8, 0.85]
        # - cluster_vote_threshold: cluster vote ratio 최소값
        cluster_vote_list = [0.1, 0.15, 0.2, 0.25]
        # - pixel_majority_threshold: cluster 내부 pixel-wise consensus threshold
        pixel_t_list = [0.5, 0.6, 0.7]
        # - cluster_agg_mode: consensus vs best
        agg_mode_list = ["consensus", "best"]

        for noobj in noobj_list:
            for miou in mask_iou_list:
                for cvote in cluster_vote_list:
                    for pix in pixel_t_list:
                        for agg in agg_mode_list:
                            # cfg_id = f"noobj{noobj}_miou{miou}_cvote{cvote}_pix{pix}_agg{agg}"
                            cfg_id = f"noobj{noobj:.2f}_miou{miou:.2f}_cvote{cvote:.2f}_pix{pix:.2f}_agg{agg}"
                            cfg = {
                                "id": cfg_id,
                                "no_object_vote_threshold": noobj,
                                "mask_iou_cluster_threshold": miou,
                                "cluster_vote_threshold": cvote,
                                "pixel_majority_threshold": pix,
                                "cluster_agg_mode": agg,
                            }
                            sweep_configs.append(cfg)
                            sweep_stats[cfg_id] = {
                                "sum_intersection": 0.0,
                                "sum_union": 0.0,
                                "count": 0,
                                "sum_iou": 0.0,
                            }
   
    # QUESTION_TEMPLATE = \
    #     "Please find \"{Question}\" with bbox(es) and point(s). " \
    #     "Also provide a short label for each object. " \
    #     "Compare the difference between object(s) and find the most closely matched object(s). " \
    #     "Return ALL matching instances; if there are no matches, return an empty list (<answer>[]</answer>). double-check none are missed. " \
    #     "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. " \
    #     "Output the bbox(es) and point(s) inside the interested object(s), along with a short label, in JSON format. " \
    #     "i.e., <think> thinking process (step-by-step reasoning) here </think> " \
    #     "<answer>{Answer}</answer>"
    
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
    
    if args.test_refcoco:
        QUESTION_TEMPLATE = \
            "Please find \"{Question}\" with bbox(es) and point(s). " \
            "Also provide a short label for each object. " \
            "Compare the difference between object(s) and find the most closely matched object(s). " \
            "Return ALL matching instances; double-check none are missed. " \
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. " \
            "Output the bbox(es) and point(s) inside the interested object(s), along with a short label, in JSON format. " \
            "i.e., <think> thinking process (step-by-step reasoning) here </think> " \
            "<answer>{Answer}</answer>"
    elif args.test_muse_mmr:
        QUESTION_TEMPLATE = \
            "Please find \"{Question}\" with bbox(es) and point(s). " \
            "Also provide a short label for each object. " \
            "First, understand and summarize what the query —\"{Question}\"— is likely referring to (which object or concept). " \
            "Then apply this to the image and find the matched target object(s). " \
            "Return ALL matching instances; double-check none are missed. " \
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. " \
            "Output the bbox(es) and point(s) inside the interested object(s), along with a short label, in JSON format. " \
            "i.e., <think> thinking process (step-by-step reasoning) here </think> " \
            "<answer>{Answer}</answer>"
    
    # SC_INSTRUCTION_TEMPLATE =  \
    #     "Your task is to find the target object(s) that match the query —\"{Question}\"—. " \
    #     "In the image, the prediction(s) from the previous answer are shown as {num_bboxes} green bounding box(es). " \
    #     "There might be errors due to a lack of comprehensive interpretation of the query and image or insufficient fine-grained analysis of the image. " \
    #     "If you find any errors, please correct the error and rewrite the solution.\n" \
    #     "Important rules for self-correction:\n" \
    #     "- First, you should observe and analyze the image along with the query very carefully, and think about what the query —\"{Question}\"— is actually referring to.\n" \
    #     "- Then, you must be sure to identify all {num_bboxes} green bbox(es) in the given image, and verify for each bounding-box region whether it matches the query.\n" \
    #     "- Also, thoroughly examine the image to determine whether there are still missing target object(s) in the image that match or answer the query but are not yet covered by any bounding box.\n" \
    #     "- Through the above process, if you find any errors in the previous answer, correct it and provide the final answer; otherwise, return the previous prediction as your final answer.\n" \
    #     "Return ALL matching instances; if there are no matches, return an empty list (<answer>[]</answer>). Double-check none are missed. " \
    #     "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. Output the bbox(es) and point(s) inside the interested object(s), along with a short label, in JSON format.\n" \
    #     "i.e., <think> thinking process (step-by-step reasoning) here </think> " \
    #     "<answer>{Answer}</answer>"
    
    # ---------------------------------------------------
    # Decomposed self-correction prompts (multi-step)
    # ---------------------------------------------------
    SC_VERIFY_LARGE_TEMPLATE = \
        "You are verifying whether a previously predicted object matches the query —\"{Question}\"—.\n" \
        "In the previous prediction, the model identified the object that matches the query (shown with a green bounding box). " \
        "In this image, the previously predicted bounding box is shown.\n" \
        "Decide ONE action for verification:\n" \
        "1) accept: the object in the bounding box matches the query.\n" \
        "2) reject: discard it ONLY if it is clearly not a match.\n" \
        "Important rules for verifying the region marked by bbox:\n" \
        "- Carefully describe and analyze the image provided to you in the context of the query. To verify if the object in the bounding box matches the query, analyze the visual context both inside and outside the box.\n" \
        "- Accept even if the bounding box covers only part of the query-matched regions—for example, when multiple target objects exist in the image but the box encloses only one of them.\n" \
        "- Be very cautious when concluding reject verdict. Provide a clear reasoning process and explicit evidence for that decision. If you are unsure, do NOT reject.\n" \
        "Output the thinking process in <think> </think> and the final action in <answer> </answer> tags. " \
        "In <answer>, output exactly one of:\n" \
        "- accept\n" \
        "- reject\n" \
        "With these rules in mind, begin verifying whether the region marked by green bbox match the query —\"{Question}\"—!"

    SC_VERIFY_SMALL_TEMPLATE = \
        "You are verifying whether a previously predicted object matches the query —\"{Question}\"—.\n" \
        "Image 1 (full image): the candidate region is shown with ONE green bounding box.\n" \
        "Image 2 (zoomed crop): the same region is zoomed so that the crop covers the box; " \
        "the green box is shown again and the red star marks the predicted point.\n" \
        "Previous coords (for reference): bbox_2d={bbox_str}, point_2d={point_str}\n" \
        "Decide ONE action for verification:\n" \
        "1) accept: the object in the bounding box matches the query.\n" \
        "2) reject: discard it ONLY if it is clearly not a match. List clear reasons.\n" \
        "3) refine: if roughly correct but box/point is not tight/accurate, output refined bbox_2d and point_2d.\n" \
        "Important rules for verifying the region marked by bbox:\n" \
        "- Carefully describe and analyze the image provided to you in the context of the query. To verify if the object in the bounding box matches the query, use Image 2 to identify the object's details and Image 1 to understand the full context. Make your final decision by integrating information from both.\n" \
        "- Be very cautious when concluding reject verdict. Provide a clear reasoning process and explicit evidence for that decision. If you are unsure, do NOT reject.\n" \
        "- If the bounding box does not tightly enclose the target object, or the point (red star) is not on the target, perform a refine action. Use the previous coords as a reference and make only small adjustments—do not change the reference coords significantly. Perform a refine action only when truly necessary, and do so carefully.\n" \
        "Output the thinking process in <think> </think> and the final action in <answer> </answer> tags. " \
        "In <answer>, output exactly one of:\n" \
        "- accept\n" \
        "- reject\n" \
        "- refined bbox_2d and point_2d, for example, {Answer}\n" \
        "With these rules in mind, begin verifying whether the region marked by green bbox match the query —\"{Question}\"—!"

    SC_MISSING_TEMPLATE = \
        "You are checking if any target object(s) for the query —\"{Question}\"— were missed.\n" \
        "The image shows {num_bboxes} green bounding box(es) from the previous prediction.\n" \
        "You should observe and analyze the image along with the query very carefully, and think about what the query —\"{Question}\"— is actually referring to. " \
        "Thoroughly examine the image to determine whether there are still missing target object(s) in the image that match or answer the query but are not yet covered by any bounding box. " \
        "If you think there are missing target object(s), be very cautious and justify it with clear reasoning and evidence. " \
        "If there is even a slight chance that the object does not match the query, output an empty list (<answer>[]</answer>).\n" \
        "Output the thinking process in <think> </think> and the final answer in <answer> </answer> tags.\n" \
        "Output the bbox(es) and point(s) inside the target object(s), along with a short label, in JSON format.\n" \
        "i.e., <think> thinking process (step-by-step reasoning) here </think> <answer>[] or {Answer}</answer>"
            
    messages = []
    id_list = []
    
    total_gen_time_sec = 0.0
    total_gen_samples = 0

    def _cuda_sync_all():
        if torch.cuda.is_available():
            try:
                for d in range(torch.cuda.device_count()):
                    torch.cuda.synchronize(d)
            except Exception:
                # Best-effort sync (do not crash latency reporting)
                pass
    
    def _attach_reasoning_type(out_item: dict, meta: dict):
        # keep existing behavior unchanged unless ReasonSeg-X explicitly enabled
        if not is_reasonsegx:
            return
        rt = meta.get("reasoning_type", None)
        if rt is not None:
            out_item["reasoning_type"] = rt
    
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
                        # Answer="[{\"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110]}, {\"bbox_2d\": [225,296,706,786], \"point_2d\": [302,410]}]"
                        # Answer="[{\"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110], \"label\": \"chair\"}, {\"bbox_2d\": [225,296,706,786], \"point_2d\": [302,410], \"label\": \"train track\"}]"
                        Answer="[{\"label\": \"chair\", \"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110]}, {\"label\": \"train track\", \"bbox_2d\": [225,296,706,786], \"point_2d\": [302,410]}]"
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
        # ReasonSeg-X only: store reasoning_type per sample (if exists)
        if is_reasonsegx:
            rt = item.get("reasoning_type", None)
            if rt is not None:
                id_list[-1]["reasoning_type"] = rt

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
        if args.vl_model_version == "qwen3":
            # HF example explicitly drops token_type_ids for Qwen3-VL. :contentReference[oaicite:4]{index=4}
            inputs.pop("token_type_ids", None)
        
        if not args.use_majority_voting:
            # # Inference: Generation of the output
            # generated_ids = reasoning_model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
            # Inference: Generation of the output (DIRECT)
            # Latency: measure ONLY the MLLM generate() wall time.
            _cuda_sync_all()
            t0 = time.perf_counter()
            generated_ids = reasoning_model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
            _cuda_sync_all()
            t1 = time.perf_counter()
            total_gen_time_sec += float(t1 - t0)
            try:
                total_gen_samples += int(inputs.input_ids.shape[0])
            except Exception:
                total_gen_samples += int(len(batch_messages))
        
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            batch_response_lengths = [
                _count_generated_tokens(out_ids, eos_token_ids=eos_token_ids, pad_token_id=pad_token_id)
                for out_ids in generated_ids_trimmed
            ]
            batch_output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        
        
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for id_idx in range(len(batch_output_text)):
                    meta = batch_id_list[id_idx]
                    is_qwen3 = (args.vl_model_version == "qwen3")
                    resp_len = batch_response_lengths[id_idx] if id_idx < len(batch_response_lengths) else 0
                    _cuda_sync_all()
                    t0 = time.perf_counter()
                    try:
                        bboxes, points, think, label = extract_bbox_points_think(
                                                batch_output_text[id_idx], 
                                                meta["img_width"]/resize_size, 
                                                meta["img_height"]/resize_size,
                                                is_qwen3=is_qwen3,
                                                image=meta["image"],
                                                qwen3_force_grid=args.qwen3_force_coord_grid,
                                            )
                    except Exception as e:
                        # add penalty in this situation
                        print("Reasoning error: ", e, "Text: ", batch_output_text[id_idx], "ID: ", batch_id_list[id_idx]["image_id"])
                        think = ""
                        label = ""
                        intersection = 0
                        union = np.array(meta["mask"]).sum()
                        bbox_iou = 0.0
                        out_item = {
                            "image_id": meta["image_id"],
                            "ann_id": meta["ann_id"],
                            "think": think,
                            "label": label,
                            "response_length": int(resp_len),
                            "intersection": int(intersection),
                            "union": int(union),
                            "bbox_iou": bbox_iou
                        }
                        _attach_reasoning_type(out_item, meta)
                        # If self-correction is enabled but the original prediction failed,
                        # we conservatively set SC metrics equal to the original fallback.
                        if args.use_self_correction and int(args.sc_num_turns) > 0:
                            sc_turns = max(0, int(args.sc_num_turns))
                            for t in range(1, sc_turns + 1):
                                out_item[f"sc_intersection_{t}"] = int(intersection)
                                out_item[f"sc_union_{t}"] = int(union)
                                if t == 1:
                                    out_item["sc_intersection"] = int(intersection)
                                    out_item["sc_union"] = int(union)
                        all_outputs.append(out_item)
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
                        gt_mask = np.array(meta["mask"])
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
                    _cuda_sync_all()
                    t1 = time.perf_counter()
                    total_gen_time_sec += float(t1 - t0)
                    
                    out_item = {
                        "image_id": meta["image_id"],
                        "ann_id": meta["ann_id"],
                        "think": think,
                        "label": label,
                        "response_length": int(resp_len),
                        "intersection": int(intersection),
                        "union": int(union),
                        "bbox_iou": 0.0,
                    }
                    _attach_reasoning_type(out_item, meta)
                    
                    if args.visualization:
                        try:
                            img = meta["image"]
                            ann_id = meta["ann_id"]
                            rt = meta.get("reasoning_type", None)
                            sample_dir = _get_sample_vis_dir(vis_dir, str(ann_id), rt)
                            query_text = meta.get("text", "")
                            
                            # 1) original input
                            try:
                                img.save(os.path.join(sample_dir, "input.png"))
                            except Exception:
                                pass

                            # 2) gt overlay (single)
                            try:
                                gt_overlay_img = _overlay_mask_on_image(
                                    img, gt_mask, color=(255, 0, 0), alpha=0.4, draw_empty_star=True
                                )
                                gt_overlay_img.save(os.path.join(sample_dir, "gt_overlay.png"))
                            except Exception:
                                pass

                            # 3) pred mask-only overlay (single)
                            try:
                                pred_mask_overlay_img = _overlay_mask_on_image(
                                    img, mask_all, color=(0, 255, 0), alpha=0.4, draw_empty_star=False
                                )
                                pred_mask_overlay_img.save(os.path.join(sample_dir, "pred_mask_overlay.png"))
                            except Exception:
                                pred_mask_overlay_img = None

                            # 4) pred bbox/point/mask overlay (single)
                            try:
                                if pred_mask_overlay_img is None:
                                    pred_mask_overlay_img = _overlay_mask_on_image(
                                        img, mask_all, color=(0, 255, 0), alpha=0.4, draw_empty_star=False
                                    )
                                pred_bp_img = _draw_bboxes_points_on_image(
                                    pred_mask_overlay_img, bboxes, points,
                                    box_color=(255, 0, 0), point_color=(0, 0, 255), width=2
                                )
                                pred_bp_img.save(os.path.join(sample_dir, "pred_bbox_point_mask_overlay.png"))
                            except Exception:
                                pass
                            
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
                            label_str = ", ".join([str(x) for x in (label if isinstance(label, list) else [label])])
                            fig.text(0.5, 0.05, f"Label : {label_str}", ha="center", va="bottom", fontsize=6)
                            iou_val = (intersection / union) if union > 0 else 1.0
                            fig.text(0.5, 0.02, f"IoU: {iou_val:.3f}", ha="center", va="bottom")
                            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
                            out_png = os.path.join(sample_dir, "compare.png")
                            fig.savefig(out_png)
                            plt.close(fig)
                        except Exception as e:
                            print("Visualization error: ", e, batch_id_list[id_idx]["image_id"], batch_id_list[id_idx]["ann_id"])
                    
                    
                    bbox_iou = 0.0
                    if has_bbox:
                        try:     
                            gt_bbox = meta["bbox"]
                            for pred_bbox in bboxes:
                                if compute_bbox_iou(pred_bbox, gt_bbox) > 0.5:
                                    bbox_iou = 1.0
                                    break
                        except Exception as e:
                            print("Bbox error: ", e, batch_id_list[id_idx]["image_id"], batch_id_list[id_idx]["ann_id"])
                            # skip this because the image or mask is not correct
                            bbox_iou = 0.0
                    out_item["bbox_iou"] = bbox_iou

                    # --------------------------------------------------
                    # Decomposed Self-correction (multi-step, do_sample=False)
                    # Steps per outer-turn:
                    #   1) verify each bbox region (accept/reject/refine) using 1 or 2 images
                    #   2) check missing objects outside all kept boxes (one forward)
                    #   3) merge kept/refined + missing, then run SAM2 for final mask
                    # --------------------------------------------------
                    if args.use_self_correction and (not args.use_majority_voting) and int(args.sc_num_turns) > 0:
                        sc_turns = max(0, int(args.sc_num_turns))
                        prev_think = think
                        prev_bboxes = list(bboxes)
                        prev_points = list(points)
                        prev_labels = list(label) if isinstance(label, list) else []
                        query_sc = meta["text"].lower().strip(".\"?!")
                        
                        x_factor = meta["img_width"] / resize_size
                        y_factor = meta["img_height"] / resize_size

                        for t in range(1, sc_turns + 1):
                            # --------------------------
                            # Step 1) verify each bbox
                            # --------------------------
                            kept_bboxes = []
                            kept_points = []
                            kept_labels = []

                            # build verification messages (1 bbox per message)
                            msgs_large = []
                            map_large = []
                            msgs_small = []
                            map_small = []

                            for j, (bb_px, pt_px) in enumerate(zip(prev_bboxes, prev_points), start=1):
                                # small vs large by bbox area ratio
                                # small = _bbox_area_ratio(bb_px, meta["img_width"], meta["img_height"]) <= float(args.sc_small_bbox_area_ratio)
                                small = False
                                bbox_str, point_str = _coords_px_to_model_coords_str(
                                    bb_px, pt_px,
                                    is_qwen3=is_qwen3,
                                    image=meta["image"],
                                    x_factor=x_factor,
                                    y_factor=y_factor,
                                    qwen3_force_grid=int(args.qwen3_force_coord_grid),
                                )

                                # if small:
                                #     global_img = _draw_bbox_overlay_image(meta["image"], [bb_px], draw_index=False)
                                # else:
                                #     global_img = _draw_bbox_point_overlay_image(meta["image"], bb_px, pt_px)
                                global_img = _draw_bbox_overlay_image(meta["image"], [bb_px], draw_index=False)
                                    
                                crop_img = None
                                if small:
                                    crop_img = _make_small_bbox_crop_overlay(
                                        meta["image"], bb_px, pt_px, cover_ratio=float(args.sc_crop_bbox_cover_ratio)
                                    )
                                    prompt = SC_VERIFY_SMALL_TEMPLATE.format(
                                        Question=query_sc, bbox_str=bbox_str, point_str=point_str,
                                        Answer="[{\"label\": \"chair\", \"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110]}]"
                                        )
                                    if args.sc_include_think and prev_think:
                                        prompt += "\nPrevious reasoning (reference):\n" + prev_think.strip() + "\n"
                                    msg = [{
                                        "role": "user",
                                        "content": [
                                            {"type": "image", "image": global_img.resize((resize_size, resize_size), PILImage.BILINEAR)},
                                            {"type": "image", "image": crop_img.resize((resize_size, resize_size), PILImage.BILINEAR)},
                                            {"type": "text", "text": prompt},
                                        ],
                                    }]
                                    msgs_small.append(msg)
                                    map_small.append((j, bb_px, pt_px))
                                else:
                                    prompt = SC_VERIFY_LARGE_TEMPLATE.format(
                                        Question=query_sc, bbox_str=bbox_str, point_str=point_str,
                                        # Answer="[{\"label\": \"chair\", \"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110]}]"
                                    )
                                    if args.sc_include_think and prev_think:
                                        prompt += "\nPrevious reasoning (reference):\n" + prev_think.strip() + "\n"
                                    msg = [{
                                        "role": "user",
                                        "content": [
                                            {"type": "image", "image": global_img.resize((resize_size, resize_size), PILImage.BILINEAR)},
                                            {"type": "text", "text": prompt},
                                        ],
                                    }]
                                    msgs_large.append(msg)
                                    map_large.append((j, bb_px, pt_px))

                                # save verify input images (debug)
                                try:
                                    _maybe_save_sc_verify_inputs(
                                        output_path=args.output_path,
                                        enable=bool(args.sc_save_verify_inputs),
                                        ann_id=str(meta.get("ann_id", "")),
                                        image_id=str(meta.get("image_id", "")),
                                        turn=int(t),
                                        bbox_idx=int(j),
                                        global_img=global_img,
                                        crop_img=crop_img,
                                    )
                                except Exception as e:
                                    print("[SC-VERIFY-SAVE] error:", e)

                            # run verify generation (batched) with multi-sampling
                            outs_large = _sc_gen_decode_multi(
                                msgs_large,
                                processor=processor,
                                reasoning_model=reasoning_model,
                                is_qwen3=is_qwen3,
                                num_samples=int(args.sc_verify_num_samples),
                                gen_batch_size=int(args.sc_gen_batch_size),
                                sample_batch_size=int(args.sc_verify_sample_batch_size),
                                max_new_tokens=int(args.sc_max_new_tokens),
                                temperature=float(args.sc_sampling_temperature),
                                top_p=float(args.sc_sampling_top_p),
                            )
                            outs_small = _sc_gen_decode_multi(
                                msgs_small,
                                processor=processor,
                               reasoning_model=reasoning_model,
                                is_qwen3=is_qwen3,
                                num_samples=int(args.sc_verify_num_samples),
                                gen_batch_size=int(args.sc_gen_batch_size),
                                sample_batch_size=int(args.sc_verify_sample_batch_size),
                                max_new_tokens=int(args.sc_max_new_tokens),
                                temperature=float(args.sc_sampling_temperature),
                                top_p=float(args.sc_sampling_top_p),
                            )

                            # apply decisions with UNANIMOUS reject rule:
                            # discard ONLY if ALL sampled responses say reject.
                            def _apply_verify_votes(outputs_per_msg, mapping):
                                for outs, (j, bb_px, pt_px) in zip(outputs_per_msg, mapping):
                                    votes = []
                                    for ot in (outs or []):
                                        votes.append(_parse_sc_verify_answer(ot))
                                    if len(votes) == 0:
                                        votes = ["accept"]
                                    if all(v == "reject" for v in votes):
                                        continue
                                    kept_bboxes.append(bb_px)
                                    kept_points.append(pt_px)
                                    kept_labels.append(prev_labels[j-1] if (j-1) < len(prev_labels) else "")

                            _apply_verify_votes(outs_large, map_large)
                            _apply_verify_votes(outs_small, map_small)

                            # --------------------------
                            # Step 2) missing objects outside prev boxes
                            # --------------------------
                            turn_prev_bboxes = list(prev_bboxes)
                            try:
                                missing_overlay = _draw_bbox_overlay_image(meta["image"], turn_prev_bboxes, draw_index=True)
                            except Exception:
                                missing_overlay = meta["image"].convert("RGB")

                            try:
                                _maybe_save_sc_verify_inputs(
                                    output_path=args.output_path,
                                    enable=bool(args.sc_save_verify_inputs),
                                    ann_id=str(meta.get("ann_id", "")),
                                    image_id=str(meta.get("image_id", "")),
                                    turn=int(t),
                                    bbox_idx=0,
                                    global_img=meta["image"].convert("RGB"),
                                    missing_overlay_img=missing_overlay,
                                )
                            except Exception:
                                pass

                            missing_prompt = SC_MISSING_TEMPLATE.format(
                                Question=query_sc, num_bboxes=len(turn_prev_bboxes),
                                Answer="[{\"label\": \"chair\", \"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110]}]"
                                )
                            if args.sc_include_think and prev_think:
                                missing_prompt += "\nPrevious reasoning (reference):\n" + prev_think.strip() + "\n"

                            missing_msg = [[{
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": missing_overlay.resize((resize_size, resize_size), PILImage.BILINEAR)},
                                    {"type": "text", "text": missing_prompt},
                                ],
                            }]]
                            
                            miss_sampled = _sc_gen_decode_multi(
                                missing_msg,
                                processor=processor,
                                reasoning_model=reasoning_model,
                               is_qwen3=is_qwen3,
                                num_samples=int(args.sc_missing_num_samples),
                                gen_batch_size=1,
                                sample_batch_size=int(args.sc_missing_sample_batch_size),
                               max_new_tokens=int(args.sc_max_new_tokens),
                                temperature=float(args.sc_sampling_temperature),
                                top_p=float(args.sc_sampling_top_p),
                            )
                            sampled_missing_texts = miss_sampled[0] if (miss_sampled and len(miss_sampled) > 0) else []

                            # MV-style missing selection (only if empty is NOT >= 50%)
                            missing_selected = _sc_select_missing_by_mv(
                                sampled_missing_texts,
                                meta=meta,
                                segmentation_model=segmentation_model,
                                is_qwen3=is_qwen3,
                                resize_size=resize_size,
                                qwen3_force_grid=int(args.qwen3_force_coord_grid),
                                empty_vote_threshold=float(args.sc_missing_empty_vote_threshold),
                                mv_iou_thr=float(args.mask_iou_cluster_threshold),
                                mv_cluster_vote_thr=float(args.cluster_vote_threshold),
                                mv_pixel_thr=float(args.pixel_majority_threshold),
                                mv_mode=str(args.cluster_agg_mode),
                            )
                            miss_bboxes = [d["bbox"] for d in missing_selected]
                            miss_points = [d["point"] for d in missing_selected]
                            miss_labels = [d["label"] for d in missing_selected]

                            # --------------------------
                            # Step 3) merge + SAM2
                            # --------------------------
                            final_bboxes = list(kept_bboxes) + list(miss_bboxes)
                            final_points = list(kept_points) + list(miss_points)
                            final_labels = list(kept_labels) + list(miss_labels)

                            sc_mask_all = np.zeros((meta["img_height"], meta["img_width"]), dtype=bool)
                            try:
                                for bbox_px, point_px in zip(kept_bboxes, kept_points):
                                    masks, scores, _ = segmentation_model.predict(
                                        point_coords=[point_px],
                                        point_labels=[1],
                                        box=bbox_px
                                    )
                                    sorted_ind = np.argsort(scores)[::-1]
                                    masks = masks[sorted_ind]
                                    sc_mask_all = np.logical_or(sc_mask_all, masks[0].astype(bool))
                                # missing masks -> use MV-selected masks directly (no re-SAM)
                                for d in missing_selected:
                                    m = d.get("mask", None)
                                    if m is None:
                                        continue
                                    sc_mask_all = np.logical_or(sc_mask_all, np.array(m).astype(bool))
                            except Exception:
                                sc_mask_all = np.zeros_like(gt_mask, dtype=bool)

                            sc_inter, sc_union = compute_iou(sc_mask_all, gt_mask)
                            out_item[f"sc_intersection_{t}"] = int(sc_inter)
                            out_item[f"sc_union_{t}"] = int(sc_union)
                            if t == 1:
                                out_item["sc_intersection"] = int(sc_inter)
                                out_item["sc_union"] = int(sc_union)

                            # delta-case save (orig vs SC)
                            try:
                                _maybe_save_sc_delta_case(
                                    output_path=args.output_path,
                                    base_image=meta["image"],
                                    query_text=meta.get("text", ""),
                                    ann_id=str(meta.get("ann_id", "")),
                                    image_id=str(meta.get("image_id", "")),
                                    turn=int(t),
                                    orig_mask=mask_all,
                                    sc_mask=sc_mask_all,
                                    orig_inter=int(intersection),
                                    orig_union=int(union),
                                    sc_inter=int(sc_inter),
                                    sc_union=int(sc_union),
                                    delta_thr=0.1,
                                )
                            except Exception as e:
                                print("[SC-DELTA-SAVE] error:", e)

                            # next outer turn conditioning
                            prev_bboxes = final_bboxes
                            prev_points = final_points
                            prev_labels = final_labels
                            prev_think = prev_think

                    all_outputs.append(out_item)

            print(f"Processed batch {i//args.batch_size + 1}/{(len(messages) + args.batch_size - 1)//args.batch_size}")
            
            # clean GPU memory
            del inputs, generated_ids, generated_ids_trimmed
            torch.cuda.empty_cache()
        
        else:
            # ----------------------------------------
            # Majority voting reasoning segmentation
            # ----------------------------------------
            num_samples = args.num_samples

            with torch.no_grad():
                # Latency: measure ONLY the MLLM generate() wall time for MV initial sampling.
                try:
                    bsz_for_latency = int(inputs.input_ids.shape[0])
                except Exception:
                    bsz_for_latency = int(len(batch_messages))

                _cuda_sync_all()
                t0 = time.perf_counter()
                generated_ids = reasoning_model.generate(
                    **inputs,
                    use_cache=True,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=args.sampling_temperature,
                    top_p=args.sampling_top_p,
                    num_return_sequences=num_samples,
                )
                
                total_gen_samples += int(bsz_for_latency)

            batch_size = inputs.input_ids.shape[0]
            input_lens = [len(ids) for ids in inputs.input_ids]

            # generated_ids: [batch_size * num_samples, seq_len]
            trimmed_ids_all = []
            trimmed_lens_all = []
            for b_idx in range(batch_size):
                base_offset = b_idx * num_samples
                prompt_len = input_lens[b_idx]
                for s_idx in range(num_samples):
                    out_ids = generated_ids[base_offset + s_idx]
                    trimmed = out_ids[prompt_len:]
                    trimmed_ids_all.append(trimmed)
                    trimmed_lens_all.append(
                        _count_generated_tokens(trimmed, eos_token_ids=eos_token_ids, pad_token_id=pad_token_id)
                    )

            decoded_all = processor.batch_decode(
                trimmed_ids_all,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            # reshape to [batch_size, num_samples]
            batch_output_text_multi: List[List[str]] = []
            batch_output_len_multi: List[List[int]] = []
            idx_dec = 0
            for b_idx in range(batch_size):
                texts_for_one = []
                lens_for_one = []
                for _ in range(num_samples):
                    texts_for_one.append(decoded_all[idx_dec])
                    lens_for_one.append(int(trimmed_lens_all[idx_dec]) if idx_dec < len(trimmed_lens_all) else 0)
                    idx_dec += 1
                batch_output_text_multi.append(texts_for_one)
                batch_output_len_multi.append(lens_for_one)

            _cuda_sync_all()
            t1 = time.perf_counter()
            total_gen_time_sec += float(t1 - t0)
            
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for id_idx in range(batch_size):
                    # ------------------------------------------------------------
                    # - "Self-correction + Majority Voting"
                    # - Collect ALL responses: original + sequential self-corrections
                    # - Run majority voting ONLY ONCE over the pooled mask candidates
                    # ------------------------------------------------------------
                    if args.use_self_correction and int(args.sc_num_turns) > 0:
                        meta = batch_id_list[id_idx]
                        texts_for_one = batch_output_text_multi[id_idx]
                        gt_mask = np.array(meta["mask"])
                        is_qwen3 = (args.vl_model_version == "qwen3")

                        try:
                            segmentation_model.set_image(meta["image"])
                        except Exception as e:
                            print("Set image error: ", e, meta["image_id"], meta["ann_id"])
                            continue

                        # Collect candidates from ALL outputs (turn0 + SC turns)
                        mask_candidates: List[MaskCandidate] = []
                        object_counts: List[int] = []
                        no_object_votes = 0
                        valid_runs = 0

                        # For SC prompting per chain (we update them sequentially)
                        # chain_states = [{"text": t, "bboxes": [], "points": [], "think": ""} for t in texts_for_one]
                        chain_states = [{"text": t, "bboxes": [], "points": [], "labels": [], "think": ""} for t in texts_for_one]
                        num_samples = int(args.num_samples)
                        sc_turns = max(0, int(args.sc_num_turns))
                        requested_total_runs = num_samples * (sc_turns + 1)
                        query_sc = meta["text"].lower().strip(".\"?!")

                        def _accumulate_from_text(out_text: str, run_sample_idx: int):
                            nonlocal valid_runs, no_object_votes, object_counts, mask_candidates
                            try:
                                bboxes, points, think_txt, labels = extract_bbox_points_think(
                                    out_text,
                                    meta["img_width"]/resize_size,
                                    meta["img_height"]/resize_size,
                                    is_qwen3=is_qwen3,
                                    image=meta["image"],
                                    qwen3_force_grid=args.qwen3_force_coord_grid,
                                )
                            except Exception as e:
                                print("Reasoning error: ", e, "Text: ", out_text, "ID: ", meta["image_id"])
                                return [], [], "", []

                            if len(bboxes) == 0:
                                valid_runs += 1
                                no_object_votes += 1
                                object_counts.append(0)
                                return bboxes, points, think_txt, labels

                            valid_runs += 1
                            object_counts.append(len(bboxes))

                            try:
                                for obj_idx, (bbox, point) in enumerate(zip(bboxes, points)):
                                    masks, scores, _ = segmentation_model.predict(
                                        point_coords=[point],
                                        point_labels=[1],
                                        box=bbox
                                    )
                                    sorted_ind = np.argsort(scores)[::-1]
                                    masks = masks[sorted_ind]
                                    scores = scores[sorted_ind]
                                    mask = masks[0].astype(bool)
                                    score = float(scores[0])
                                    lbl = labels[obj_idx] if obj_idx < len(labels) else ""
                                    mask_candidates.append(
                                        MaskCandidate(
                                            mask=mask,
                                            score=score,
                                            sample_idx=run_sample_idx,
                                            obj_idx=obj_idx,
                                            bbox=bbox,
                                            point=point,
                                            label=lbl,
                                        )
                                    )
                            except Exception as e:
                                print("Segmentation error: ", e, meta["image_id"], meta["ann_id"])
                            return bboxes, points, think_txt, labels

                        # ------------------------------
                        # turn 0: original sampled outputs
                        # ------------------------------
                        for s, out_text in enumerate(texts_for_one):
                            bboxes, points, think_txt, labels = _accumulate_from_text(out_text, run_sample_idx=s)
                            chain_states[s]["bboxes"] = bboxes
                            chain_states[s]["points"] = points
                            chain_states[s]["labels"] = labels
                            chain_states[s]["think"] = think_txt
                        
                        def _accumulate_from_boxes(bxs, pts, lbls, run_sample_idx: int):
                            nonlocal valid_runs, no_object_votes, object_counts, mask_candidates
                            if len(bxs) == 0:
                                valid_runs += 1
                                no_object_votes += 1
                                object_counts.append(0)
                                return
                            valid_runs += 1
                            object_counts.append(len(bxs))
                            try:
                                for obj_idx, (bbox, point) in enumerate(zip(bxs, pts)):
                                    masks, scores, _ = segmentation_model.predict(
                                        point_coords=[point],
                                        point_labels=[1],
                                        box=bbox
                                    )
                                    sorted_ind = np.argsort(scores)[::-1]
                                    masks = masks[sorted_ind]
                                    scores = scores[sorted_ind]
                                    mask = masks[0].astype(bool)
                                    score = float(scores[0])
                                    lbl = lbls[obj_idx] if obj_idx < len(lbls) else ""
                                    mask_candidates.append(
                                        MaskCandidate(
                                            mask=mask,
                                            score=score,
                                            sample_idx=run_sample_idx,
                                            obj_idx=obj_idx,
                                            bbox=bbox,
                                            point=point,
                                            label=lbl,
                                        )
                                    )
                            except Exception as e:
                                print("Segmentation error: ", e, meta["image_id"], meta["ann_id"])

                        # Helper: run the SAME MV finalize logic (copied from existing behavior)
                        def _finalize_mv(mask_candidates, object_counts, no_object_votes, valid_runs):
                            if valid_runs == 0:
                                final_mask_all = np.zeros_like(gt_mask, dtype=bool)
                                intersection, union = compute_iou(final_mask_all, gt_mask)
                                bbox_iou = 0.0
                                final_labels: List[str] = []
                                final_num_masks = 0
                                return final_mask_all, intersection, union, bbox_iou, final_labels, final_num_masks

                            no_obj_ratio = no_object_votes / valid_runs
                            if no_obj_ratio >= args.no_object_vote_threshold:
                                final_mask_all = np.zeros_like(gt_mask, dtype=bool)
                                intersection, union = compute_iou(final_mask_all, gt_mask)
                                bbox_iou = 0.0
                                final_labels = []
                                final_num_masks = 0
                                return final_mask_all, intersection, union, bbox_iou, final_labels, final_num_masks

                            positive_counts = [c for c in object_counts if c > 0]
                            if len(positive_counts) == 0:
                                K_hat = 0
                            else:
                                counts_counter = Counter(positive_counts)
                                K_hat = counts_counter.most_common(1)[0][0]

                            if K_hat <= 0 or len(mask_candidates) == 0:
                                final_mask_all = np.zeros_like(gt_mask, dtype=bool)
                                intersection, union = compute_iou(final_mask_all, gt_mask)
                                bbox_iou = 0.0
                                final_labels = []
                                final_num_masks = 0
                                return final_mask_all, intersection, union, bbox_iou, final_labels, final_num_masks

                            cluster_infos = cluster_mask_candidates(
                                mask_candidates,
                                iou_thr=args.mask_iou_cluster_threshold,
                                num_samples=valid_runs,
                            )
                            if len(cluster_infos) == 0:
                                final_mask_all = np.zeros_like(gt_mask, dtype=bool)
                                intersection, union = compute_iou(final_mask_all, gt_mask)
                                bbox_iou = 0.0
                                final_labels = []
                                final_num_masks = 0
                                return final_mask_all, intersection, union, bbox_iou, final_labels, final_num_masks

                            filtered = [
                                c for c in cluster_infos
                                if c["vote_ratio"] >= args.cluster_vote_threshold
                            ]
                            if len(filtered) == 0:
                                filtered = cluster_infos

                            cluster_infos_sorted = sorted(
                                filtered,
                                key=lambda c: (c["vote_count"], c["avg_score"]),
                                reverse=True,
                            )
                            num_to_select = min(len(cluster_infos_sorted), K_hat)
                            selected = cluster_infos_sorted[:num_to_select]

                            final_masks: List[np.ndarray] = []
                            final_labels = []
                            for cinfo in selected:
                                m_star, lbl = aggregate_cluster_mask(
                                    cinfo,
                                    mask_candidates,
                                    mode=args.cluster_agg_mode,
                                    pixel_thr=args.pixel_majority_threshold,
                                )
                                if m_star is None:
                                    continue
                                final_masks.append(m_star)
                                final_labels.append(lbl)

                            if len(final_masks) == 0:
                                final_mask_all = np.zeros_like(gt_mask, dtype=bool)
                            else:
                                final_mask_all = np.zeros_like(gt_mask, dtype=bool)
                                for fm in final_masks:
                                    final_mask_all = np.logical_or(final_mask_all, fm)

                            intersection, union = compute_iou(final_mask_all, gt_mask)

                            bbox_iou = 0.0
                            if has_bbox and len(final_masks) > 0:
                                try:
                                    gt_bbox = meta["bbox"]
                                    for fm in final_masks:
                                        ys, xs = np.where(fm)
                                        if ys.size == 0:
                                            continue
                                        x1, x2 = xs.min(), xs.max() + 1
                                        y1, y2 = ys.min(), ys.max() + 1
                                        pred_bbox = [int(x1), int(y1), int(x2), int(y2)]
                                        if compute_bbox_iou(pred_bbox, gt_bbox) > 0.5:
                                            bbox_iou = 1.0
                                            break
                                except Exception as e:
                                    print("Bbox error: ", e, meta["image_id"], meta["ann_id"])
                                    bbox_iou = 0.0

                            final_num_masks = len(final_masks)
                            return final_mask_all, intersection, union, bbox_iou, final_labels, final_num_masks

                        for t in range(1, sc_turns + 1):
                            # ------------------------------------------------------------
                            # Decomposed SC round (verify each bbox -> missing -> merge)
                            # ------------------------------------------------------------
                            x_factor = meta["img_width"] / resize_size
                            y_factor = meta["img_height"] / resize_size

                            # 1) verify each bbox across all chains (flatten -> batch)
                            msgs_large = []
                            map_large = []   # (chain_id, bbox_idx, bb_px, pt_px, lbl)
                            msgs_small = []
                            map_small = []

                            for s in range(num_samples):
                                prev_bxs = chain_states[s]["bboxes"]
                                prev_pts = chain_states[s]["points"]
                                prev_lbls = chain_states[s]["labels"] if isinstance(chain_states[s].get("labels", []), list) else []
                                for j, (bb_px, pt_px) in enumerate(zip(prev_bxs, prev_pts), start=1):
                                    # small = _bbox_area_ratio(bb_px, meta["img_width"], meta["img_height"]) <= float(args.sc_small_bbox_area_ratio)
                                    small = False
                                    bbox_str, point_str = _coords_px_to_model_coords_str(
                                        bb_px, pt_px,
                                        is_qwen3=is_qwen3,
                                        image=meta["image"],
                                        x_factor=x_factor,
                                        y_factor=y_factor,
                                        qwen3_force_grid=int(args.qwen3_force_coord_grid),
                                    )
                                    # if small:
                                    #     global_img = _draw_bbox_overlay_image(meta["image"], [bb_px], draw_index=False)
                                    # else:
                                    #     global_img = _draw_bbox_point_overlay_image(meta["image"], bb_px, pt_px)
                                    global_img = _draw_bbox_overlay_image(meta["image"], [bb_px], draw_index=False)
                                    # global_img = _draw_bbox_point_overlay_image(meta["image"], bb_px, pt_px)
                                    crop_img = None
                                    if small:
                                        crop_img = _make_small_bbox_crop_overlay(
                                            meta["image"], bb_px, pt_px, cover_ratio=float(args.sc_crop_bbox_cover_ratio)
                                        )
                                        prompt = SC_VERIFY_SMALL_TEMPLATE.format(
                                            Question=query_sc, bbox_str=bbox_str, point_str=point_str,
                                            Answer="[{\"label\": \"chair\", \"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110]}]"
                                            )
                                        if args.sc_include_think and chain_states[s]["think"]:
                                            prompt += "\nPrevious reasoning (reference):\n" + chain_states[s]["think"].strip() + "\n"
                                        msg = [{
                                            "role": "user",
                                            "content": [
                                                {"type": "image", "image": global_img.resize((resize_size, resize_size), PILImage.BILINEAR)},
                                                {"type": "image", "image": crop_img.resize((resize_size, resize_size), PILImage.BILINEAR)},
                                                {"type": "text", "text": prompt},
                                            ],
                                        }]
                                        msgs_small.append(msg)
                                        map_small.append((s, j, bb_px, pt_px, prev_lbls[j-1] if (j-1) < len(prev_lbls) else ""))
                                    else:
                                        prompt = SC_VERIFY_LARGE_TEMPLATE.format(
                                            Question=query_sc, bbox_str=bbox_str, point_str=point_str,
                                            # Answer="[{\"label\": \"chair\", \"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110]}]"
                                            )
                                        if args.sc_include_think and chain_states[s]["think"]:
                                            prompt += "\nPrevious reasoning (reference):\n" + chain_states[s]["think"].strip() + "\n"
                                        msg = [{
                                            "role": "user",
                                            "content": [
                                                {"type": "image", "image": global_img.resize((resize_size, resize_size), PILImage.BILINEAR)},
                                                {"type": "text", "text": prompt},
                                            ],
                                        }]
                                        msgs_large.append(msg)
                                        map_large.append((s, j, bb_px, pt_px, prev_lbls[j-1] if (j-1) < len(prev_lbls) else ""))

                            outs_large = _sc_gen_decode_multi(
                                msgs_large,
                                processor=processor,
                                reasoning_model=reasoning_model,
                                is_qwen3=is_qwen3,
                                num_samples=int(args.sc_verify_num_samples),
                                gen_batch_size=int(args.sc_gen_batch_size),
                                sample_batch_size=int(args.sc_verify_sample_batch_size),
                                max_new_tokens=int(args.sc_max_new_tokens),
                                temperature=float(args.sc_sampling_temperature),
                                top_p=float(args.sc_sampling_top_p),
                            )
                            outs_small = _sc_gen_decode_multi(
                                msgs_small,
                                processor=processor,
                                reasoning_model=reasoning_model,
                                is_qwen3=is_qwen3,
                                num_samples=int(args.sc_verify_num_samples),
                                gen_batch_size=int(args.sc_gen_batch_size),
                                sample_batch_size=int(args.sc_verify_sample_batch_size),
                                max_new_tokens=int(args.sc_max_new_tokens),
                                temperature=float(args.sc_sampling_temperature),
                                top_p=float(args.sc_sampling_top_p),
                            )

                            kept_bxs = [[] for _ in range(num_samples)]
                            kept_pts = [[] for _ in range(num_samples)]
                            kept_lbls = [[] for _ in range(num_samples)]

                            # UNANIMOUS reject rule (per-box, per-chain)
                            def _apply_votes(outputs_per_msg, mapping):
                                for outs, (s, j, bb_px, pt_px, lbl) in zip(outputs_per_msg, mapping):
                                    votes = []
                                    for ot in (outs or []):
                                        votes.append(_parse_sc_verify_answer(ot))
                                    if len(votes) == 0:
                                        votes = ["accept"]
                                    if all(v == "reject" for v in votes):
                                        continue
                                    kept_bxs[s].append(bb_px)
                                    kept_pts[s].append(pt_px)
                                    kept_lbls[s].append(lbl)

                            _apply_votes(outs_large, map_large)
                            _apply_votes(outs_small, map_small)

                            # 2) missing step across all chains (batch)
                            miss_msgs = []
                            for s in range(num_samples):
                                prev_bboxes = chain_states[s]["bboxes"]
                                try:
                                    overlay_all = _draw_bbox_overlay_image(meta["image"], prev_bboxes, draw_index=True)
                                except Exception:
                                    overlay_all = meta["image"].convert("RGB")
                                
                                prompt = SC_MISSING_TEMPLATE.format(
                                    Question=query_sc, num_bboxes=len(prev_bboxes),
                                    Answer="[{\"label\": \"chair\", \"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110]}]"
                                    )
                                if args.sc_include_think and chain_states[s]["think"]:
                                    prompt += "\nPrevious reasoning (reference):\n" + chain_states[s]["think"].strip() + "\n"
                                miss_msgs.append([{
                                    "role": "user",
                                    "content": [
                                        {"type": "image", "image": overlay_all.resize((resize_size, resize_size), PILImage.BILINEAR)},
                                        {"type": "text", "text": prompt},
                                    ],
                                }])

                            miss_outs_multi = _sc_gen_decode_multi(
                                miss_msgs,
                                processor=processor,
                                reasoning_model=reasoning_model,
                                is_qwen3=is_qwen3,
                                num_samples=int(args.sc_missing_num_samples),
                                gen_batch_size=int(args.sc_gen_batch_size),
                                sample_batch_size=int(args.sc_missing_sample_batch_size),
                                max_new_tokens=int(args.sc_max_new_tokens),
                                temperature=float(args.sc_sampling_temperature),
                                top_p=float(args.sc_sampling_top_p),
                            )

                            # helper: accumulate one SC run into the MV pool, but use MV-selected missing masks directly
                            def _accumulate_sc_run(kept_bx, kept_pt, kept_lb, miss_sel, run_sample_idx: int):
                                nonlocal valid_runs, no_object_votes, object_counts, mask_candidates
                                total_k = int(len(kept_bx)) + int(len(miss_sel))
                                valid_runs += 1
                                object_counts.append(total_k)
                                if total_k == 0:
                                    no_object_votes += 1
                                    return
                                obj_idx = 0
                                # kept -> run SAM2 once
                                for bbox, point, lbl in zip(kept_bx, kept_pt, kept_lb):
                                    try:
                                        masks, scores, _ = segmentation_model.predict(
                                            point_coords=[point],
                                            point_labels=[1],
                                            box=bbox,
                                        )
                                        sorted_ind = np.argsort(scores)[::-1]
                                        masks = masks[sorted_ind]
                                        scores = scores[sorted_ind]
                                        m0 = masks[0].astype(bool)
                                        s0 = float(scores[0])
                                        mask_candidates.append(
                                            MaskCandidate(
                                                mask=m0,
                                                score=s0,
                                                sample_idx=run_sample_idx,
                                                obj_idx=obj_idx,
                                                bbox=bbox,
                                                point=point,
                                                label=lbl,
                                            )
                                        )
                                        obj_idx += 1
                                    except Exception:
                                        continue
                                # missing -> use MV-selected mask directly (no re-SAM)
                                for d in miss_sel:
                                    try:
                                        m = np.array(d.get("mask", None)).astype(bool)
                                        if m is None:
                                            continue
                                        bbox = d.get("bbox", [0, 0, 0, 0])
                                        point = d.get("point", [0, 0])
                                        lbl = d.get("label", "")
                                        sc = float(d.get("score", 1.0))
                                        mask_candidates.append(
                                            MaskCandidate(
                                                mask=m,
                                                score=sc,
                                                sample_idx=run_sample_idx,
                                                obj_idx=obj_idx,
                                                bbox=bbox,
                                                point=point,
                                                label=lbl,
                                            )
                                        )
                                        obj_idx += 1
                                    except Exception:
                                        continue

                            # 3) merge + accumulate masks as new "run" for MV pool
                            for s in range(num_samples):
                                run_sample_idx = s + t * num_samples

                                sampled_missing_texts = miss_outs_multi[s] if (s < len(miss_outs_multi)) else []
                                miss_selected = _sc_select_missing_by_mv(
                                    sampled_missing_texts,
                                    meta=meta,
                                    segmentation_model=segmentation_model,
                                    is_qwen3=is_qwen3,
                                    resize_size=resize_size,
                                    qwen3_force_grid=int(args.qwen3_force_coord_grid),
                                    empty_vote_threshold=float(args.sc_missing_empty_vote_threshold),
                                    mv_iou_thr=float(args.mask_iou_cluster_threshold),
                                    mv_cluster_vote_thr=float(args.cluster_vote_threshold),
                                    mv_pixel_thr=float(args.pixel_majority_threshold),
                                    mv_mode=str(args.cluster_agg_mode),
                                )

                                mb = [d["bbox"] for d in miss_selected]
                                mp = [d["point"] for d in miss_selected]
                                mlbl = [d["label"] for d in miss_selected]

                                final_bxs = kept_bxs[s] + mb
                                final_pts = kept_pts[s] + mp
                                final_lbls = kept_lbls[s] + mlbl

                                chain_states[s]["bboxes"] = final_bxs
                                chain_states[s]["points"] = final_pts
                                chain_states[s]["labels"] = final_lbls
                                # keep think as-is (optional)
                                chain_states[s]["think"] = chain_states[s]["think"]

                                _accumulate_sc_run(kept_bxs[s], kept_pts[s], kept_lbls[s], miss_selected, run_sample_idx=run_sample_idx)

                        # --------------------------------------------------
                        # ONE-SHOT MV over pooled candidates (original + all SC turns)
                        # --------------------------------------------------
                        final_mask_all, intersection, union, bbox_iou, final_labels, final_num_masks = _finalize_mv(
                            mask_candidates, object_counts, no_object_votes, valid_runs
                        )

                        out_item = {
                            "image_id": meta["image_id"],
                            "ann_id": meta["ann_id"],
                            "think": "",
                            "label": final_labels if valid_runs > 0 else [],
                            "intersection": int(intersection),
                            "union": int(union),
                            "bbox_iou": bbox_iou,
                            # keep original fields
                            "num_samples": int(args.num_samples),
                            # additional bookkeeping (optional, doesn't affect existing metrics scripts)
                            "sc_num_turns": int(sc_turns),
                            "num_total_requested_runs": int(requested_total_runs),
                            "num_valid_samples": int(valid_runs),
                            "no_object_votes": int(no_object_votes),
                            "final_num_masks": int(final_num_masks),
                        }
                        _attach_reasoning_type(out_item, meta)

                        all_outputs.append(out_item)
                        continue
                    
                    _cuda_sync_all()
                    t0 = time.perf_counter()
                    
                    meta = batch_id_list[id_idx]
                    texts_for_one = batch_output_text_multi[id_idx]
                    lens_for_one = batch_output_len_multi[id_idx] if id_idx < len(batch_output_len_multi) else []
                    resp_len_mean = float(sum(lens_for_one) / max(1, len(lens_for_one))) if lens_for_one else 0.0
                    gt_mask = np.array(meta["mask"])

                    try:
                        segmentation_model.set_image(meta["image"])
                    except Exception as e:
                        print("Set image error: ", e, meta["image_id"], meta["ann_id"])
                        continue

                    mask_candidates: List[MaskCandidate] = []
                    object_counts: List[int] = []
                    no_object_votes = 0
                    valid_runs = 0
                    final_masks_for_viz: List[np.ndarray] = []

                    # 각 reasoning sample에 대해 bbox/point 파싱 및 SAM2 mask 생성
                    for sample_idx, out_text in enumerate(texts_for_one):
                        try:
                            bboxes, points, _, labels = extract_bbox_points_think(
                                out_text,
                                meta["img_width"]/resize_size,
                                meta["img_height"]/resize_size,
                                is_qwen3=(args.vl_model_version == "qwen3"),
                                image=meta["image"],
                                qwen3_force_grid=args.qwen3_force_coord_grid,
                            )
                        except Exception as e:
                            print("Reasoning error: ", e, "Text: ", out_text, "ID: ", meta["image_id"])
                            # invalid run: abstain
                            continue

                        if len(bboxes) == 0:
                            valid_runs += 1
                            no_object_votes += 1
                            object_counts.append(0)
                            continue

                        valid_runs += 1
                        object_counts.append(len(bboxes))

                        try:
                            for obj_idx, (bbox, point) in enumerate(zip(bboxes, points)):
                                masks, scores, _ = segmentation_model.predict(
                                    point_coords=[point],
                                    point_labels=[1],
                                    box=bbox
                                )
                                sorted_ind = np.argsort(scores)[::-1]
                                masks = masks[sorted_ind]
                                scores = scores[sorted_ind]
                                mask = masks[0].astype(bool)
                                score = float(scores[0])
                                label = labels[obj_idx] if obj_idx < len(labels) else ""
                                mask_candidates.append(
                                    MaskCandidate(
                                        mask=mask,
                                        score=score,
                                        sample_idx=sample_idx,
                                        obj_idx=obj_idx,
                                        bbox=bbox,
                                        point=point,
                                        label=label,
                                    )
                                )
                        except Exception as e:
                            print("Segmentation error: ", e, meta["image_id"], meta["ann_id"])
                            continue

                    # voting/ensemble 단계
                    if valid_runs == 0:
                        intersection = 0
                        union = gt_mask.sum()
                        bbox_iou = 0.0
                        final_labels: List[str] = []
                        final_mask_all = np.zeros_like(gt_mask, dtype=bool)
                        final_num_masks = 0
                    else:
                        no_obj_ratio = no_object_votes / valid_runs
                        if no_obj_ratio >= args.no_object_vote_threshold:
                            # 최종 no-object
                            final_mask_all = np.zeros_like(gt_mask, dtype=bool)
                            intersection, union = compute_iou(final_mask_all, gt_mask)
                            bbox_iou = 0.0
                            final_labels = []
                            final_num_masks = 0
                        else:
                            # object 개수 majority (k>0인 경우만 사용)
                            positive_counts = [c for c in object_counts if c > 0]
                            if len(positive_counts) == 0:
                                K_hat = 0
                            else:
                                counts_counter = Counter(positive_counts)
                                K_hat = counts_counter.most_common(1)[0][0]

                            if K_hat <= 0 or len(mask_candidates) == 0:
                                final_mask_all = np.zeros_like(gt_mask, dtype=bool)
                                intersection, union = compute_iou(final_mask_all, gt_mask)
                                bbox_iou = 0.0
                                final_labels = []
                                final_num_masks = 0
                            else:
                                cluster_infos = cluster_mask_candidates(
                                    mask_candidates,
                                    iou_thr=args.mask_iou_cluster_threshold,
                                    num_samples=valid_runs,
                                )
                                if len(cluster_infos) == 0:
                                    final_mask_all = np.zeros_like(gt_mask, dtype=bool)
                                    intersection, union = compute_iou(final_mask_all, gt_mask)
                                    bbox_iou = 0.0
                                    final_labels = []
                                    final_num_masks = 0
                                else:
                                    # vote_ratio 기반 필터링 (비면 전체 사용)
                                    filtered = [
                                        c for c in cluster_infos
                                        if c["vote_ratio"] >= args.cluster_vote_threshold
                                    ]
                                    if len(filtered) == 0:
                                        filtered = cluster_infos

                                    cluster_infos_sorted = sorted(
                                        filtered,
                                        key=lambda c: (c["vote_count"], c["avg_score"]),
                                        reverse=True,
                                    )
                                    num_to_select = min(len(cluster_infos_sorted), K_hat)
                                    selected = cluster_infos_sorted[:num_to_select]

                                    final_masks: List[np.ndarray] = []
                                    final_labels = []
                                    for cinfo in selected:
                                        m_star, lbl = aggregate_cluster_mask(
                                            cinfo,
                                            mask_candidates,
                                            mode=args.cluster_agg_mode,
                                            pixel_thr=args.pixel_majority_threshold,
                                        )
                                        if m_star is None:
                                            continue
                                        final_masks.append(m_star)
                                        final_labels.append(lbl)

                                    if len(final_masks) == 0:
                                        final_mask_all = np.zeros_like(gt_mask, dtype=bool)
                                    else:
                                        final_mask_all = np.zeros_like(gt_mask, dtype=bool)
                                        for fm in final_masks:
                                            final_mask_all = np.logical_or(final_mask_all, fm)
                                    final_masks_for_viz = list(final_masks)
                                    
                                    intersection, union = compute_iou(final_mask_all, gt_mask)

                                    bbox_iou = 0.0
                                    if has_bbox and len(final_masks) > 0:
                                        try:
                                            gt_bbox = meta["bbox"]
                                            for fm in final_masks:
                                                ys, xs = np.where(fm)
                                                if ys.size == 0:
                                                    continue
                                                x1, x2 = xs.min(), xs.max() + 1
                                                y1, y2 = ys.min(), ys.max() + 1
                                                pred_bbox = [int(x1), int(y1), int(x2), int(y2)]
                                                if compute_bbox_iou(pred_bbox, gt_bbox) > 0.5:
                                                    bbox_iou = 1.0
                                                    break
                                        except Exception as e:
                                            print("Bbox error: ", e, meta["image_id"], meta["ann_id"])
                                            bbox_iou = 0.0

                                    final_num_masks = len(final_masks)

                    _cuda_sync_all()
                    t1 = time.perf_counter()
                    total_gen_time_sec += float(t1 - t0)
                    
                    # --------------------------------------------------
                    # (추가) Hyper-parameter sweep:
                    # 같은 MLLM + SAM 결과(mask_candidates 등)를 재사용해서
                    # 여러 hp 조합에 대해 intersection/union을 누적
                    # --------------------------------------------------
                    if args.enable_hparam_sweep and args.use_majority_voting and len(sweep_configs) > 0:
                        for cfg in sweep_configs:
                            inter_cfg, union_cfg = run_ensemble_with_params(
                                mask_candidates=mask_candidates,
                                object_counts=object_counts,
                                no_object_votes=no_object_votes,
                                valid_runs=valid_runs,
                                gt_mask=gt_mask,
                                no_object_vote_threshold=cfg["no_object_vote_threshold"],
                                mask_iou_cluster_threshold=cfg["mask_iou_cluster_threshold"],
                                cluster_vote_threshold=cfg["cluster_vote_threshold"],
                                pixel_majority_threshold=cfg["pixel_majority_threshold"],
                                cluster_agg_mode=cfg["cluster_agg_mode"],
                            )
                            st = sweep_stats[cfg["id"]]
                            st["sum_intersection"] += inter_cfg
                            st["sum_union"] += union_cfg
                            st["count"] += 1
                            if union_cfg > 0:
                                iou_val_cfg = inter_cfg / union_cfg
                            else:
                                iou_val_cfg = 1.0
                            # st["ious"].append(iou_val_cfg)
                            st["sum_iou"] += float(iou_val_cfg)
                    
                    if args.visualization:
                        try:
                            img = meta["image"]
                            ann_id = meta["ann_id"]
                            rt = meta.get("reasoning_type", None)
                            sample_dir = _get_sample_vis_dir(vis_dir, str(ann_id), rt)
                            query_text = meta.get("text", "")
                            
                            # 1) original input
                            try:
                                img.save(os.path.join(sample_dir, "input.png"))
                            except Exception:
                                pass

                            # 2) gt overlay
                            try:
                                gt_overlay_img = _overlay_mask_on_image(
                                    img, gt_mask, color=(255, 0, 0), alpha=0.4, draw_empty_star=True
                                )
                                gt_overlay_img.save(os.path.join(sample_dir, "gt_overlay.png"))
                            except Exception:
                                pass

                            # 3) pred mask-only overlay
                            try:
                                pred_mask_overlay_img = _overlay_mask_on_image(
                                    img, final_mask_all, color=(0, 255, 0), alpha=0.4, draw_empty_star=False
                                )
                                pred_mask_overlay_img.save(os.path.join(sample_dir, "pred_mask_overlay.png"))
                            except Exception:
                                pred_mask_overlay_img = None

                            # 4) pred bbox/point/mask overlay (MV에서는 final_masks에서 bbox/point 유도)
                            try:
                                viz_boxes = []
                                viz_points = []
                                if final_masks_for_viz and len(final_masks_for_viz) > 0:
                                    for m in final_masks_for_viz:
                                        bb, pt = _mask_to_bbox_point(np.array(m).astype(bool))
                                        if isinstance(bb, (list, tuple)) and len(bb) == 4:
                                            viz_boxes.append(bb)
                                            viz_points.append(pt)
                                elif np.array(final_mask_all).astype(bool).any():
                                    bb, pt = _mask_to_bbox_point(np.array(final_mask_all).astype(bool))
                                    viz_boxes = [bb]
                                    viz_points = [pt]

                                if pred_mask_overlay_img is None:
                                    pred_mask_overlay_img = _overlay_mask_on_image(
                                        img, final_mask_all, color=(0, 255, 0), alpha=0.4, draw_empty_star=False
                                    )
                                pred_bp_img = _draw_bboxes_points_on_image(
                                    pred_mask_overlay_img, viz_boxes, viz_points,
                                    box_color=(255, 0, 0), point_color=(0, 0, 255), width=2
                                )
                                pred_bp_img.save(os.path.join(sample_dir, "pred_bbox_point_mask_overlay.png"))
                            except Exception:
                                pass
                            
                            fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=150)

                            # 좌: GT overlay
                            axes[0].imshow(img)
                            H_gt, W_gt = gt_mask.shape
                            gt_overlay = np.zeros((H_gt, W_gt, 4), dtype=np.float32)
                            gt_overlay[gt_mask] = [1.0, 0.0, 0.0, 0.4]
                            axes[0].imshow(gt_overlay)
                            axes[0].set_title("GT Overlay")
                            axes[0].axis("off")
                            if not gt_mask.any():
                                star_x = 10
                                star_y = 10
                                axes[0].plot(
                                    star_x, star_y,
                                    marker="*", markersize=15,
                                    markeredgecolor="black", markerfacecolor="black"
                                )

                            # 우: ensemble mask overlay
                            axes[1].imshow(img)
                            H_pr, W_pr = final_mask_all.shape
                            pred_overlay = np.zeros((H_pr, W_pr, 4), dtype=np.float32)
                            pred_overlay[final_mask_all] = [0.0, 1.0, 0.0, 0.4]
                            axes[1].imshow(pred_overlay)
                            axes[1].set_title("Ensemble Pred Overlay")
                            axes[1].axis("off")

                            fig.suptitle(query_text, fontsize=7)
                            iou_val = (intersection / union) if union > 0 else 1.0
                            fig.text(0.5, 0.02, f"IoU: {iou_val:.3f}", ha="center", va="bottom")
                            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
                            out_png = os.path.join(sample_dir, "compare.png")
                            fig.savefig(out_png)
                            plt.close(fig)
                        except Exception as e:
                            print("Visualization error: ", e, meta["image_id"], meta["ann_id"])

                    out_item = {
                        "image_id": meta["image_id"],
                        "ann_id": meta["ann_id"],
                        "think": "",
                        "label": final_labels if valid_runs > 0 else [],
                        "response_length": float(resp_len_mean),
                        "intersection": int(intersection),
                        "union": int(union),
                        "bbox_iou": bbox_iou,
                        "num_samples": args.num_samples,
                        "num_valid_samples": int(valid_runs),
                        "no_object_votes": int(no_object_votes),
                        "final_num_masks": int(final_num_masks),
                    }
                    _attach_reasoning_type(out_item, meta)
                    all_outputs.append(out_item)

            print(f"[Majority] Processed batch {i//args.batch_size + 1}/{(len(messages) + args.batch_size - 1)//args.batch_size}")

            # clean GPU memory
            del inputs, generated_ids
            torch.cuda.empty_cache()

    if total_gen_samples > 0:
        avg_latency_sec = float(total_gen_time_sec) / float(total_gen_samples)
        print(f"\n[Latency] Mask generation per-sample avg: {avg_latency_sec:.1f}s")
    else:
        print("\n[Latency] Mask generation per-sample avg: n/a (no samples measured)")
    
    if args.use_majority_voting and args.enable_hparam_sweep and len(sweep_configs) > 0:
        
        sweep_json_path = os.path.join(args.output_path, f"sweep_stats_{args.idx}.json")
        payload = {
            "idx": int(args.idx),
            "num_parts": int(args.num_parts),
            "reasoning_model_path": args.reasoning_model_path,
            "test_data_path": args.test_data_path,
            "num_samples": int(args.num_samples),
            "sampling_temperature": float(args.sampling_temperature),
            "sampling_top_p": float(args.sampling_top_p),
            "configs": sweep_configs,
            "stats": sweep_stats,
        }
        with open(sweep_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[SWEEP] Shard sweep stats written to: {sweep_json_path}")
    
    # Modify the output file name, add idx
    output_file = os.path.join(args.output_path, f"output_{args.idx}.json")
    with open(output_file, "w") as f:
        json.dump(all_outputs, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
