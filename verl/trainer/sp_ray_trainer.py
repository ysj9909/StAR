import io
import tempfile
import os
import warnings
import re
import json
import math
import random
import uuid
import shutil
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ray
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL import Image as _PILImage
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.rl_dataset import collate_fn, process_image, RLHFDataset
from verl.utils.proposer_rl_dataset import ProposerRLHFDataset, ProposerSample
import verl.utils.torch_functional as verl_F
from verl.models.transformers.qwen2_5_vl import get_rope_index
from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value, Sequence

from verl.trainer.ray_trainer import (
    RayPPOTrainer,
    _timer,
    apply_kl_penalty,
    compute_advantage,
    reduce_metrics,
    compute_data_metrics,
    compute_timing_metrics,
)

# -------------------------
# (A) Self-play 상수/타입
# -------------------------

DEFAULT_RS_TYPES = [
    "function_purpose",
    "commonsense",
    "comparative_relational",                   
    # "compositional"
]

NEG_RES_QUERIES = [
    "bottom right white couch",
    "first giraffe on left",
    "the little kid holding a racket",
    "the dog under the table",
    "the person in red shirt",
]


class SPRayPPOTrainer(RayPPOTrainer):
    """
    - gen: proposer (image→caption+query+points/bboxes)
    - pred: solver (query→segments)
    """

    # ====== 초기화 ======
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        sp = getattr(self.config.worker.actor, "selfplay", None)
        self.rs_types: List[str] = DEFAULT_RS_TYPES
        self.update_iteration: int = getattr(sp, "update_iteration", 1)
        self.num_pos_refs: int = getattr(sp, "num_pos_refs", 10)
        self.neg_res_examples: List[str] = NEG_RES_QUERIES
        self.n_learnability_samples: int = max(2, getattr(sp, "n_learnability_samples", 8))  # num_samples for measuring mean acc.
        # {'uniform_total','step','half_new','max_new','frontier'}
        self.pred_data_mix_strategy: str = getattr(sp, "pred_data_mix_strategy", "max_new")  
        self.save_proposer_outputs: bool = bool(getattr(sp, "save_proposer_outputs", True))
        self.proposer_monitor_dir: str = os.path.join(
            self.config.trainer.save_checkpoint_path, "monitor", "proposer"
        )
        os.makedirs(self.proposer_monitor_dir, exist_ok=True)
        
        # ---- Selection / validation config (defaults safe) ----
        self.auto_mask_cfg: Dict[str, Any] = dict(getattr(sp, "auto_mask_cfg", {}) or {})
        # self.candidate_mask_source: str = str(getattr(sp, "candidate_mask_source", "amg"))  # mask source: 'amg' | 'dataset'
        # dataset mask bank (fingerprint -> List[np.bool])
        self._dataset_mask_bank: Dict[bytes, List[np.ndarray]] = {}
        self.mask_dup_iou_thr: float = float(getattr(sp, "mask_dup_iou_thr", 0.9))
        self.nested_contain_thr: float = float(getattr(sp, "nested_contain_thr", 0.95))
        self.selection_max_k: int = int(getattr(sp, "selection_max_k", 80))
        self.require_both_selection: bool = bool(getattr(sp, "require_both_selection", True))
        self.use_learnability_adv_weight: bool = bool(getattr(sp, "use_learnability_adv_weight", True))
        self.proposer_update_after_step: int = int(getattr(sp, "proposer_update_after_step", 0))
        self.proposer_update_every: int = int(getattr(sp, "proposer_update_every", 5))
        self.train_proposer: bool = bool(getattr(sp, "train_proposer", True))
        
        # ---- task pool ----
        self.max_task_pool: int = getattr(sp, "max_task_pool", 200)
        self.fixed_pos_refs_by_type: Dict[str, List[str]] = getattr(sp, "fixed_pos_refs_by_type", {}) or {}
        self.task_pool: List[Dict[str, Any]] = []   # [{rtype, query, image(PIL), gt_json, gt_masks, learnability, step}, ...]
        
        self._recent_additions: Dict[str, int] = defaultdict(int)   # current step counter
        self._prev_recent_additions: Dict[str, int] = defaultdict(int)

        self._first_pred_iter_by_type: Dict[str, int] = {}
        
        # ===== Prompts  =====
        # 1) Query-Referenced Description (QRD)
        self.qrd_prompt_template = "<image>\n" \
            "For the region described as {Question} in the image, provide a single detailed sentence describing an object or part of a object by including its location, appearance (color, shape), and distinctive characteristics including relevant actions, state, or function. " \
            "Focus on features that would help uniquely identify this specific region from others in the image."
        
        # 2) Proposal Region Description (PRD) on BBox overlaid image and mask cropped image
        self.prd_prompt_template = "<image>\n<image>\n" \
            "You are presented with two complementary views of the same region: " \
            "1) A cropped masked view showing detailed visual properties; " \
            "2) A full view with a bounding box showing location and context. " \
            "Generate a single detailed sentence following these guidelines:\n" \
            "* FOR COMPLETE OBJECTS:\n" \
            "- Combine visual details and spatial context naturally.\n" \
            "- Describe visual properties (color, shape, texture, size);\n" \
            "- Location in the scene;\n" \
            "- Relationships with surroundings;\n" \
            "- State or action if relevant.\n" \
            "* FOR PARTIAL REGIONS:\n" \
            "- Describe the part while providing clear context;\n" \
            "- Part identification and its visual properties;\n" \
            "- Its position within the larger object/scene;\n" \
            "- Relevant contextual details.\n" \
            "* Important Rules: Start directly with the subject: 'A [description]...' or 'The [description]...'; " \
            "Describe only what is visible in the non-black regions for visual properties and the image with green bounding box is for location and relation analysis; " \
            "Never mention masks, boxes, or annotations; " \
            "Use confident language for clear identifications; " \
            "Use tentative language when inferring; " \
            "Create natural, flowing descriptions that combine all information seamlessly; " \
            "Focus on creating cohesive descriptions that feel natural and informative without drawing attention to the source of the information."
        
        # 3-a) Text-Image comparison (QRD + Query + BBox image) → yes/no
        self.tic_prompt_template = "<image>\n<image>\n" \
            "You are evaluating if the following reference text describes the non-black region of the cropped mask image: {QRD}. " \
            "The target is {Query} for context if the reference text is inaccurate." \
            "You have two images for context: " \
            "1) A cropped mask image showing a region in non-black color; " \
            "2) An image with a green bounding box surrounding the region showing the full scene and spatial relationships. " \
            "Evaluate if the reference text describes the non-black region of the cropped mask image by checking:\n" \
            "- Spatial location match (the location is relative location, not absolute location);\n" \
            "- Visual characteristics match (color, shape, size);\n" \
            "- Object/subject identity match;\n" \
            "- State/action consistency (if applicable).\n" \
            "Return 'yes' or 'no' ONLY: 'yes' if most aspects substantially match; 'no' if some significant aspect differs."
            
        # 3-b) Text-Text comparison (PRD vs QRD+Query) → yes/no
        self.ttc_prompt_template = "You are evaluating if the following candidate text describes the input expression target region: {Query}.\n" \
            "Reference information provided for context if the input expression text is not clear: {QRD}\n" \
            "Here is the candidate text to evaluate: {Proposal_desc}\n" \
            "Evaluate if the candidate text refer to the target by checking:\n" \
            "- Spatial location match;\n" \
            "- Visual characteristics match (color, shape, size);\n" \
            "- Object/subject identity match;\n" \
            "- State/action consistency (if applicable).\n" \
            "Return 'yes' or 'no' ONLY: 'yes' if most aspects substantially match; 'no' if some significant aspect differs."
        
        # ---- 이미지 소스(HF dataset) 접근자 ----
        # RayPPOTrainer._create_dataloader()에서 생성된 self.train_dataset 를 재활용
        # (self.train_dataset.dataset 은 HF Dataset이며 'image' 열을 가짐)
        assert hasattr(self, "train_dataset"), "RayPPOTrainer.__init__ must be called first."
    
    # ----- (helper) task_pool에서 learnability≈0.5 프론티어 쿼리 샘플 -----
    def _sample_queries_from_task_pool(self, rtype: str, k: int, exclude: Optional[set] = None) -> List[str]:
        if k <= 0:
            return []
        exclude = exclude or set()
        cand = [task for task in self.task_pool
                if task.get("rtype") == rtype and isinstance(task.get("learnability", None), (int, float))]
        if not cand:
            return []
        cand_sorted = sorted(
            cand,
            key=lambda task: (-float(task.get("learnability", 0.0)), -int(task.get("step", 0)))
        )
        seen = set(exclude)
        picks: List[str] = []
        for t in cand_sorted:
            q = str(t.get("query", "")).strip()
            if not q or q in seen:
                continue
            seen.add(q)
            picks.append(q)
            if len(picks) >= k:
                break
        return picks

    # ----- (helper) proposer용 positive refs 구축: 고정 K//2 + 프론티어 K//2 (+ 부족분도 task_pool로 보충) -----
    def _get_pos_refs_for_gen(self, rtype: str) -> List[str]:
        K = max(0, int(self.num_pos_refs))
        if K == 0:
            return []
        k_fixed = K // 2
        k_dyn   = K - k_fixed
        fixed_all = self.fixed_pos_refs_by_type.get(rtype, []) or []
        fixed_half = fixed_all[:k_fixed] if len(fixed_all) >= k_fixed else fixed_all[:]
        dyn_half = self._sample_queries_from_task_pool(rtype, k_dyn, exclude=set(fixed_half))
        out: List[str] = []
        seen = set()
        for q in (fixed_half + dyn_half):
            q = str(q).strip()
            if not q or q in seen:
                continue
            seen.add(q)
            out.append(q)
        # 부족분은 task_pool 프론티어에서 추가 보충
        remain = max(0, K - len(out))
        if remain > 0:
            more = self._sample_queries_from_task_pool(rtype, remain, exclude=seen)
            for q in more:
                if len(out) >= K:
                    break
                out.append(q)
        return out[:K]
       
    # ====== 유틸: solver용 HF DatasetDict -> RLHFDataset 로더 생성 ======
    def _rows_to_solver_dataloader(self,
                                   rows: List[Dict[str, Any]],
                                   shuffle=None,
                                   batch_size=None,
                                   drop_last=True) -> DataLoader:
        """
        rows: [{"problem": str, "image": PIL.Image, "solution": str, "solution_mask": List[H x W bool], "rtype": str, "query": str, "task_id": int}, ...]
        """ 
        tmp_dir = os.path.join(self.config.trainer.save_checkpoint_path, "tmp_pred_hfds")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_dir = os.path.join(tmp_dir, f"step_{self.global_steps}_{uuid.uuid4().hex}")

        # nested variable shapes allowed via nested Sequence
        features = Features({
            "problem": Value("string"),
            "image": HFImage(),
            "solution": Value("string"),
            "solution_mask": Sequence(Sequence(Sequence(Value("bool")))),  # [num_masks][H][W] (ragged OK)
            "rtype": Value("string"),
            "query": Value("string"),
            "task_id": Value("string"),
            "learnability": Value("float32"),
        })
        # Convert masks to nested lists
        rows_proc = []
        for r in rows:
            masks = r.get("solution_mask", None)
            masks_list = None if masks is None else [m.astype(bool).tolist() for m in masks]
            rows_proc.append({
                "problem": r["problem"],
                "image": r["image"],
                "solution": r["solution"],
                "solution_mask": masks_list,
                "rtype": r.get("rtype", ""),
                "query": r.get("query", r["problem"]),
                "task_id": str(r.get("task_id", "")),
                "learnability": float(r["learnability"]) if r.get("learnability", None) is not None else 0.0,
            })

        ds = Dataset.from_list(rows_proc, features=features)
        ds_dict = DatasetDict({"train": ds})
        ds_dict.save_to_disk(tmp_dir)

        solver_ds = RLHFDataset(
            data_path=tmp_dir,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key="prompt",
            max_prompt_length=self.config.data.max_prompt_length,
            truncation="right",
            system_prompt=self.config.data.system_prompt,
            max_pixels=self.config.data.max_pixels,
            min_pixels=self.config.data.min_pixels,
            remove_lisa=False,
        )
        
        if shuffle is None:
            use_shuffle = self.config.data.shuffle
        else:
            use_shuffle = shuffle   
        if use_shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.seed)
            sampler = RandomSampler(solver_ds, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(solver_ds)
        
        if batch_size is None:
            batch_size = self.config.data.rollout_batch_size
        
        return DataLoader(
            dataset=solver_ds,
            batch_size=batch_size,
            drop_last=drop_last,
            collate_fn=collate_fn,
            sampler=sampler,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )
    
    def _parse_proposer_output(self, text: str) -> Optional[Dict[str, Any]]:
        """
        expected input format:
          <caption> ... </caption>
          <think> ... </think>
          <answer>
            [ {"query": "..."} ,
              {"bbox_2d":[...], "point_2d":[...], "label":"..."}, ... ]
          </answer>
        output:
          {"caption": str, "query": str, "targets": [{"point":[x,y],"box":[x1,y1,x2,y2]}...]}
        """
        try:
            caption_match = re.search(r"<caption>\s*(.*?)\s*</caption>", text, re.DOTALL)
            think_match = re.search(r"<think>\s*(.*?)\s*</think>", text, re.DOTALL)
            answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
            if not answer_match:
                return None
            if caption_match:
                caption = caption_match.group(1).strip()
            else:
                caption = ""
            body = answer_match.group(1).strip()

            data = json.loads(body)
            if not isinstance(data, list) or len(data) < 2 or not isinstance(data[0], dict) or "query" not in data[0]:
                return None
            query = str(data[0]["query"]).strip()
            think_txt = think_match.group(1).strip() if think_match else ""

            targets = []
            for item in data[1:]:
                if not isinstance(item, dict):
                    continue
                bbox = item.get("bbox_2d")
                point = item.get("point_2d")
                label = item.get("label", "")
                if (isinstance(bbox, list) and len(bbox) == 4 and
                    isinstance(point, list) and len(point) == 2 and
                    isinstance(label, str) and label.strip()):
                    targets.append({"box"  : bbox,
                                    "point": point,
                                    "label": label.strip()})
            if not targets:
                return None

            return {"caption": caption, "think": think_txt, "query": query, "targets": targets, "raw_list": data, "answer_text": body}
        except Exception:
            return None

    def _proposer_format_reward(self, text: str) -> float:
        """
        format_reward = 0.5 (태그 패턴 일치) + 0.5 (answer 리스트가 요구 형식 충족) in [0,1]
        """
        score = 0.0
        # 태그 구조
        pattern = r"<caption>.*?</caption>\s*<think>.*?</think>\s*<answer>.*?</answer>"
        if re.search(pattern, text, re.DOTALL):
            score += 0.5
        # answer 리스트 검사
        try:
            answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
            if not answer_match:
                return score
            js = json.loads(answer_match.group(1))
            ok = (isinstance(js, list) and len(js) >= 2 and isinstance(js[0], dict) and "query" in js[0])
            if ok:
                for it in js[1:]:
                    if not (isinstance(it, dict) and
                            isinstance(it.get("bbox_2d"), list) and len(it["bbox_2d"]) == 4 and
                            isinstance(it.get("point_2d"), list) and len(it["point_2d"]) == 2 and
                            isinstance(it.get("label", ""), str) and it["label"].strip() != ""):
                        ok = False
                        break
            if ok:
                score += 0.5
        except Exception:
            pass
        return float(score)
    
    def _generate_masks_with_sam(self, image: Image.Image, targets: List[Dict[str, Any]]) -> Optional[List[np.ndarray]]:
        if not hasattr(self, "sam_wg"):
            return None
        if not targets:
            return None
        try:
            points = [t["point"] for t in targets]
            points = np.array(points)
            boxes = [t["box"] for t in targets]
            boxes = np.array(boxes)
            masks = self.sam_wg.predict_masks_from_points_boxes(image, points, boxes)
            # List[np.bool(H,W)]
            if not masks:
                return None
            if all(np.array(mask).sum() == 0 for mask in masks):
                return None
            return [np.array(mask) for mask in masks]
        except Exception:
            return None
    
    def _generate_masks_from_boxes_sam(self, image: Image.Image, boxes: List[List[float]]) -> Optional[List[np.ndarray]]:
        """Box-only → SAM2 mask. Uses worker's box-only path when points=None."""
        if not hasattr(self, "sam_wg"):
            return None
        if not boxes:
            return None
        
        boxes_np = np.array(boxes)
        masks = self.sam_wg.predict_masks_from_points_boxes(image, points=None, boxes=boxes_np)
        if not masks or all(np.array(m).sum() == 0 for m in masks):
            return None
        return [np.array(m) for m in masks]
        
    # ==============================
    # Low-level utilities for masks
    # ==============================
    def _mask_iou_bool(self, m1: np.ndarray, m2: np.ndarray) -> float:
        if m1.dtype != bool: m1 = m1.astype(bool)
        if m2.dtype != bool: m2 = m2.astype(bool)
        inter = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()
        return float(inter) / float(union + 1e-6)

    def _merge_and_dedup_tagged(self, predicted_masks: List[np.ndarray], auto_masks: List[np.ndarray],
                                iou_thr: float, max_k: int) -> List[Dict[str, Any]]:
        """Return list of {'mask':..., 'src': 'ml'|'amg', 'ml_idx': Optional[int]} with guided priority."""
        items: List[Dict[str, Any]] = []
        for idx, m in enumerate(predicted_masks or []):
            if isinstance(m, np.ndarray) and m.ndim == 2:
                items.append({"mask": m.astype(bool), "src": "ml", "ml_idx": idx})
        for m in (auto_masks or []):
            if not (isinstance(m, np.ndarray) and m.ndim == 2): 
                continue
            m = m.astype(bool)
            if all(self._mask_iou_bool(x["mask"], m) < iou_thr for x in items):
                items.append({"mask": m, "src": "amg", "ml_idx": None})
        if len(items) > max_k:
            areas = [int(x["mask"].sum()) for x in items]
            order = np.argsort([-a for a in areas]).tolist()
            items = [items[i] for i in order[:max_k]]
        return items

    def _dedup_masks(self, masks: List[np.ndarray], iou_thr: float) -> List[np.ndarray]:
        """Greedy de-duplication within a single mask list."""
        out: List[np.ndarray] = []
        for m in masks or []:
            if not (isinstance(m, np.ndarray) and m.ndim == 2):
                continue
            m = m.astype(bool)
            if all(self._mask_iou_bool(m, x) < iou_thr for x in out):
                out.append(m)
        return out
    
    def _suppress_nested_masks(self, items: List[Dict[str, Any]], contain_thr: float) -> Tuple[List[Dict[str, Any]], int]:
        """
        큰 마스크를 우선 보존하면서, 다른 마스크에 contain_thr 이상 포함되는 작은 마스크를 제거.
        items: [{'mask': np.ndarray(bool,H,W), 'src': 'ml'|'amg', 'ml_idx': Optional[int]}, ...]
        return: (kept_items, num_removed)
        """
        if not items:
            return [], 0
        areas = np.array([int(x["mask"].sum()) for x in items], dtype=np.int64)
        order = np.argsort(-areas)  # 큰 것부터
        keep = np.ones(len(items), dtype=bool)
        for ii in order:
            if not keep[ii]:
                continue
            mi = items[ii]["mask"]
            ai = float(areas[ii]) + 1e-6
            for jj in order[::-1]:  # 작은 것부터 보게됨
                if jj == ii or not keep[jj]:
                    continue
                mj = items[jj]["mask"]
                aj = float(areas[jj]) + 1e-6
                if aj > ai:  # jj가 더 크면 skip (큰 것은 나중에 억제 대상 아님)
                    continue
                inter = float(np.logical_and(mj, mi).sum())
                frac = inter / aj  # 작은(후보) 마스크의 픽셀 중 큰 마스크에 포함되는 비율
                if frac >= contain_thr:
                    keep[jj] = False
        kept = [items[k] for k in range(len(items)) if keep[k]]
        removed = int((~keep).sum())
        return kept, removed
    
    # ---------- Monitoring: visualization helpers ----------
    def _star_polygon(self, cx: float, cy: float, r_outer: float = 8.0, r_inner: float = 4.0, n: int = 5):
        """Return a list of (x,y) for a star centered at (cx,cy)."""
        import math as _m
        pts = []
        for k in range(2 * n):
            ang = _m.pi / n * k - _m.pi / 2
            r = r_outer if k % 2 == 0 else r_inner
            pts.append((cx + r * _m.cos(ang), cy + r * _m.sin(ang)))
        return pts

    def _save_proposer_visual(
        self,
        image: Image.Image,
        query: str,
        targets: List[Dict[str, Any]],
        masks: Optional[List[np.ndarray]],
        out_path: str,
        accepted: bool,
    ) -> None:
        """Draw query title, masks (red alpha), boxes (green), and star point (green)."""
        try:
            img = image.convert("RGBA")
            W, H = img.size
            base = img.copy()

            # Overlay masks in semi-transparent red
            if masks:
                # accumulate overlay
                overlay = Image.new("RGBA", (W, H), (255, 0, 0, 0))
                ov = np.array(overlay)
                for m in masks:
                    # ensure shape matches
                    if m.shape[0] == H and m.shape[1] == W:
                        ov[m] = [255, 0, 0, 80]
                overlay = Image.fromarray(ov, mode="RGBA")
                base = Image.alpha_composite(base, overlay)

            draw = ImageDraw.Draw(base)
            # Boxes & points in green
            for t in targets or []:
                b = t.get("box", None)
                p = t.get("point", None)
                if isinstance(b, list) and len(b) == 4:
                    x1, y1, x2, y2 = [int(round(v)) for v in b]
                    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0, 255), width=3)
                if isinstance(p, list) and len(p) == 2:
                    sx, sy = float(p[0]), float(p[1])
                    star = self._star_polygon(sx, sy, 8.0, 4.0, 5)
                    draw.polygon(star, outline=(0, 255, 0, 255), fill=(0, 255, 0, 180))

            # Title bar with query
            title_h = 32
            canvas = Image.new("RGBA", (W, H + title_h), (0, 0, 0, 0))
            canvas.paste(base, (0, title_h))
            draw2 = ImageDraw.Draw(canvas)
            draw2.rectangle([0, 0, W, title_h], fill=(0, 0, 0, 180))
            title = (query or "(invalid)") + ("  [ACCEPTED]" if accepted else "  [REJECTED]")
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
            draw2.text((6, 6), title, fill=(255, 255, 255, 255), font=font)

            canvas.convert("RGB").save(out_path)
        except Exception:
            pass
        # ---------- Monitoring: visualization helpers ----------
    
    def _overlay_bbox_on_image(self, image: Image.Image, bbox: List[int]) -> Image.Image:
        """Green bbox only."""
        img = image.convert("RGB").copy()
        draw = ImageDraw.Draw(img)
        try:
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        except Exception:
            x1, y1, x2, y2 = 0, 0, 0, 0
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        return img
    
    def _mask_cropped_image(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        """Generate an image that preserves the original size
           but fills pixels outside the mask with black."""
        arr = np.array(image.convert("RGB"))
        m = np.array(mask, dtype=bool)
        if m.shape[0] != arr.shape[0] or m.shape[1] != arr.shape[1]:
            # safety: 크기 불일치 시 최근접 보간으로 맞춤
            m_img = Image.fromarray(m.astype(np.uint8) * 255).resize((arr.shape[1], arr.shape[0]), Image.NEAREST)
            m = np.array(m_img) > 0
        arr[~m] = 0
        return Image.fromarray(arr)

    def _mask_to_bbox(self, mask: np.ndarray) -> List[int]:
        ys, xs = np.where(mask.astype(bool))
        if len(xs) == 0 or len(ys) == 0:
            return [0, 0, 0, 0]
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        return [x1, y1, x2, y2]

    def _mask_centroid(self, mask: np.ndarray) -> List[float]:
        ys, xs = np.where(mask.astype(bool))
        if len(xs) == 0 or len(ys) == 0:
            return [0.0, 0.0]
        return [float(xs.mean()), float(ys.mean())]
    
    # ---------- dataset mask helpers ----------
    def _img_fingerprint(self, image: Image.Image, size: int = 32) -> bytes:
        """리사이즈에 비교적 강인한 간단한 그레이스케일 지문 생성."""
        g = image.convert("L").resize((size, size), Image.BILINEAR)
        return bytes(np.asarray(g, dtype=np.uint8).ravel().tolist())

    def _resize_mask_to_image(self, mask: np.ndarray, image: Image.Image) -> np.ndarray:
        H, W = image.size[1], image.size[0]
        m = np.array(mask, dtype=bool)
        if m.shape[0] == H and m.shape[1] == W:
            return m
        m_img = Image.fromarray(m.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)
        return (np.array(m_img) > 0)

    # ======================
    # MLLM generation helpers (batch)
    # ======================
    def _build_features_for_prompt(self, images: Optional[List[Image.Image]], user_prompt: str) -> Dict[str, Any]:
        """Single-sample feature dict compatible with collate_fn -> DataProto."""
        system_prompt = self.config.data.system_prompt
        messages = [
           {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        row: Dict[str, Any] = {}

        if images is not None and len(images) > 0:
            proc_imgs = [process_image(image, self.config.data.max_pixels, self.config.data.min_pixels, use_resize=False)
                         for image in images]
            row["images"] = proc_imgs
            # replace <image> tokens
            prompt_work = prompt
            raw_prompt = prompt_work.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
            image_inputs = self.processor.image_processor(row["images"], return_tensors="pt")
            image_grid_thw = image_inputs["image_grid_thw"]
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                idx = 0
                while "<image>" in prompt_work:
                    prompt_work = prompt_work.replace(
                        "<image>",
                        "<|vision_start|>"
                        + "<|placeholder|>" * (image_grid_thw[idx].prod() // merge_length)
                        + "<|vision_end|>",
                        1,
                    )
                    idx += 1
                prompt_work = prompt_work.replace("<|placeholder|>", self.processor.image_token)
            prompt = prompt_work
            row.update(image_inputs)
            row["image_grid_thw"] = image_grid_thw
        else:
            raw_prompt = prompt

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_length=self.config.data.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation="right",
        )
        if images is not None and len(images) > 0:
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=row["image_grid_thw"],
                attention_mask=attention_mask,
            )
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)

        row["input_ids"] = input_ids
        row["attention_mask"] = attention_mask
        row["position_ids"] = position_ids
        row["raw_prompt_ids"] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        return row

    def _batch_generate_texts(self, images: List[Optional[Image.Image]], prompts: List[str],
                                do_sample: bool = True) -> List[str]:
        """Run one-shot generation for many (image,prompt) pairs; returns decoded response texts."""
        # feats = [self._build_features_for_prompt(img, pmpt) for img, pmpt in zip(images, prompts)]
        feats = []
        for obj, pmpt in zip(images, prompts):
            if obj is None:
                feats.append(self._build_features_for_prompt([], pmpt))
            elif isinstance(obj, list):
                feats.append(self._build_features_for_prompt(obj, pmpt))
            else:
                feats.append(self._build_features_for_prompt([obj], pmpt))
        
        batch_dict = collate_fn(feats)
        dp_in = DataProto.from_single_dict(batch_dict)
        dp_in.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": bool(do_sample),
            "validate": False,
        }
        dp_pad, pad_sz = pad_dataproto_to_divisor(dp_in, self.actor_rollout_wg.world_size)
        dp_out_pad = self.actor_rollout_wg.generate_sequences(dp_pad)
        dp_out = unpad_dataproto(dp_out_pad, pad_size=pad_sz)

        # decode: use input attention_mask to find valid response length
        attn = dp_in.batch["attention_mask"]
        responses = dp_out.batch["responses"]
        has_prompt = "prompts" in dp_in.batch
        texts: List[str] = []
        for j in range(len(responses)):
            resp_ids = responses[j]
            rlen = resp_ids.shape[-1]
            if has_prompt:
                prompt_len = int(dp_in.batch["prompts"][j].shape[-1])
                valid_len = int(attn[j, prompt_len:].sum().item())
            else:
                valid_len = int(attn[j, -rlen:].sum().item())
            txt = self.tokenizer.decode(resp_ids[:valid_len], skip_special_tokens=True)
            texts.append(txt.strip())
        return texts

    def _parse_yes_no(self, text: str) -> bool:
        """Return True if first decisive token is 'yes' (case-insensitive), else False."""
        s = (text or "").strip().lower()
        # find first occurrence of 'yes' or 'no'
        idx_y = s.find("yes")
        idx_n = s.find("no")
        if idx_y == -1 and idx_n == -1:
            return False
        if idx_y == -1:
            return False
        if idx_n == -1:
            return True
        return idx_y < idx_n
  
    def _create_train_rs_gen_dataloader(self, rtype: str, data_len: int):
        """
        image sample → (rtype, K pos refs, N neg refs) with ProposerRLHFDataset
        """
        hf_ds = getattr(self.train_dataset, "dataset", None)
        assert hf_ds is not None, "train_dataset.dataset(HF Dataset) is needed."

        # image random sampling
        idxs = random.sample(range(len(hf_ds)), k=min(data_len, len(hf_ds)))
        
        # Positive references: 고정 K//2 + task_pool 프론티어 K//2 (+부족분 task_pool 보충)
        pos_refs = self._get_pos_refs_for_gen(rtype)
        neg_refs = list(self.neg_res_examples)  # 전 타입 공통

        samples: List[ProposerSample] = []
        for i in idxs:
            ds_bboxes = None
            obj = hf_ds[i].get("objects", None)
            if obj is not None:
                ds_bboxes = obj['bbox']
            samples.append(ProposerSample(
                image=hf_ds[i]["image"],
                rtype=rtype,
                pos_refs=pos_refs,
                neg_refs=neg_refs,
                dataset_bboxes=ds_bboxes,
            ))

        rs_gen_train_dataset = ProposerRLHFDataset(
            samples=samples,
            tokenizer=self.tokenizer,
            processor=self.processor,
            system_prompt=self.config.data.system_prompt,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation="right",
            max_pixels=self.config.data.max_pixels,
            min_pixels=self.config.data.min_pixels,
            type_definitions=None,
        )
        
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.seed)
            sampler = RandomSampler(rs_gen_train_dataset, generator=train_dataloader_generator)
        else:    
            sampler = SequentialSampler(rs_gen_train_dataset)
        
        return iter(DataLoader(
            dataset=rs_gen_train_dataset,
            batch_size=self.config.data.rollout_batch_size,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        ))
    
    def _create_train_rs_pred_dataloader(self, rtype: str, data_len: int):
        pool = [task for task in self.task_pool if task["rtype"] == rtype]
        if len(pool) < data_len:
            return None

        def select_indices(strategy: str) -> List[int]:
            n = min(data_len, len(pool))
            if strategy == "uniform_total":
                return random.sample(range(len(pool)), n)
            elif strategy == "step":
                steps = np.array([p["step"] for p in pool], dtype=np.float32)
                weights = steps - steps.min() + 1.0
                probs = weights / weights.sum()
                return list(np.random.choice(len(pool), size=n, replace=len(pool) < n, p=probs))
            elif strategy == "half_new":
                half = n // 2
                recent_sorted = sorted(range(len(pool)), key=lambda i: pool[i]["step"], reverse=True)
                recent_idx = recent_sorted[:half]
                rest = [i for i in range(len(pool)) if i not in set(recent_idx)]
                remain = n - len(recent_idx)
                others = random.sample(rest, remain) if remain > 0 and len(rest) >= remain else rest[:remain]
                return recent_idx + others
            elif strategy == "max_new":
                total_recent = int(self._prev_recent_additions.get(rtype, 0))
                new_problems = list(range(len(pool) - total_recent, len(pool)))
                remain = n - len(new_problems)
                if remain > 0:
                    rest = [i for i in range(len(pool)) if i not in new_problems]
                    possible_remain = min(remain, len(rest))
                    new_problems += random.sample(rest, possible_remain)
                    remain -= possible_remain
                if remain > 0:
                    rest = [i for i in range(len(pool)) if i not in new_problems]
                    new_problems += random.sample(rest, remain)
                return new_problems
            elif strategy == "frontier":
                # 프론티어: learnability 큰 순 + step 큰 순
                def key_fn(i):
                    t = pool[i]
                    l = float(t.get("learnability", 0.0))
                    return (-l, -int(t.get("step", 0)))
                order = sorted(range(len(pool)), key=key_fn)
                return order[:n]
            else:
                return random.sample(range(len(pool)), n)

        idxs = select_indices(self.pred_data_mix_strategy)
        # RLHFDataset 기반으로 생성
        rows: List[Dict[str, Any]] = []
        for i in idxs:
            t = pool[i]
            rows.append({
                "problem": t["query"],
                "image": t["image"],
                "solution": t["gt_json"],
                "solution_mask": t["gt_masks"],
                "rtype": t["rtype"],
                "query": t["query"],
                "task_id": uuid.uuid4().hex,
                "learnability": t.get("learnability", 0.0),
            })
        return iter(self._rows_to_solver_dataloader(rows))
    
    def _sanitize_non_tensor_for_concat(self, dp: DataProto):
        keep = {"pixel_values", "image_grid_thw", "uid"}
        for k in list(dp.non_tensor_batch.keys()):
            if k not in keep:
                try:
                    del dp.non_tensor_batch[k]
                except Exception:
                    pass
        return dp
    
    # ====== (3) GEN 배치 처리(보상: learnability) / PRED 배치 처리(보상: 정확도) ======
    def _compute_batch(self, batch: DataProto, metrics: dict, timing_raw: dict,
                       problem_type: str) -> Tuple[DataProto, dict]:
        
        is_gen = problem_type.startswith("gen_")
        rtype = problem_type.split("gen_")[-1] if is_gen else problem_type.split("pred_")[-1]

        # ---------- 1) 텍스트 생성 ----------
        # 이미지 멀티모달은 이미 batch에 포함되어 있음
        if is_gen:
            gen_batch = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"],
                              non_tensor_batch_keys=["pixel_values", "image_grid_thw", "raw_prompt_ids", "images", "dataset_bboxes"],
        )
        else:
            gen_batch = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"],
                              non_tensor_batch_keys=["pixel_values", "image_grid_thw", "raw_prompt_ids", "images"])

        with _timer(f"gen/{problem_type}", timing_raw):
            gen_out = self.actor_rollout_wg.generate_sequences(gen_batch)

        # uid/rollout 정렬
        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
        batch = batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
        batch = batch.union(gen_out)
        
        # compute global_valid tokens
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
        
        # recompute old log prob
        with _timer(f"old_log_prob/{problem_type}", timing_raw):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            batch = batch.union(old_log_prob)

        if is_gen:
            # (a) proposer 출력 파싱 + SAM2로 마스크 생성 + ref buffer/ task pool 업데이트
            # (b) solver를 N회 샘플링해서 평균 정답률 추정 → learnability 보상
            with _timer(f"gen/post/{problem_type}", timing_raw):
                
                already_print = 0
                num_examine = 1
                
                valid_response_len = []
                decoded_prompt_texts = []
                decoded_response_texts = []
                B = len(batch)
                for i in range(B):
                    data_item = batch[i]
                    
                    prompt_ids = data_item.batch["prompts"]
                    prompt_length = prompt_ids.shape[-1]
                    valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                    valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                    
                    response_ids = data_item.batch["responses"]
                    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                    valid_response_ids = response_ids[:valid_response_length]

                    # decode
                    prompt_str   = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                    response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                    
                    valid_response_len.append(valid_response_length)
                    decoded_prompt_texts.append(prompt_str)
                    decoded_response_texts.append(response_str)


                new_tasks = []   # local pool
                learnability_r = np.zeros(B, dtype=np.float32)
                format_r = np.zeros(B, dtype=np.float32)

                accepted_indices = []
                for i in range(B):
                    txt = decoded_response_texts[i]
                    format_r[i] = self._proposer_format_reward(txt)
                    parsed = self._parse_proposer_output(txt)
                    if not parsed:
                        learnability_r[i] = 0.0
                        continue
                    
                    image = gen_batch[i].non_tensor_batch["images"][0] 
                    
                    # Guided masks (point+box → SAM2)
                    predicted_masks = self._generate_masks_with_sam(image, parsed["targets"])
                    if not predicted_masks:
                        learnability_r[i] = 0.0
                        continue
                    
                    # Auto masks (SAM2 AMG) + Dataset instance masks + remove duplicate
                    auto_masks = []
                    if hasattr(self, "sam_wg"):
                        auto_masks = self.sam_wg.generate_auto_masks(image, self.auto_mask_cfg)
                        
                    ds_bbox_list = gen_batch[i].non_tensor_batch.get("dataset_bboxes", None)
                    ds_masks = []
                    if ds_bbox_list:
                        ds_masks = self._generate_masks_from_boxes_sam(image, ds_bbox_list)
                    
                    can_masks = self._dedup_masks((ds_masks or []) + (auto_masks or []), iou_thr=self.mask_dup_iou_thr)
                    
                    cand_items = self._merge_and_dedup_tagged(
                        predicted_masks, can_masks, iou_thr=self.mask_dup_iou_thr, max_k=self.selection_max_k
                    )

                    # ----- (Step 1) Query-Referenced Description (QRD) -----
                    qrd_prompt = self.qrd_prompt_template.format(Question=parsed["query"])
                    qrd_text = self._batch_generate_texts([image], [qrd_prompt], do_sample=True)[0]

                    # ----- (Step 2) Proposal Region Descriptions (PRD) on bbox-only overlay -----
                    prd_images: List[List[Image.Image]] = []
                    prd_prompts: List[str] = []
                    bboxes_for_item: List[List[int]] = []
                    for ci in cand_items:
                        m = ci["mask"]
                        bb = self._mask_to_bbox(m)
                        bboxes_for_item.append(bb)
                        bbox_img = self._overlay_bbox_on_image(image, bb)
                        crop_img = self._mask_cropped_image(image, m)
                        prd_images.append([crop_img, bbox_img])
                        prd_prompts.append(self.prd_prompt_template)
                    prd_texts: List[str] = []
                    if prd_images:
                        prd_texts = self._batch_generate_texts(prd_images, prd_prompts, do_sample=True)
                    else:
                        prd_texts = []

                    # ----- (Step 3-a) Text-Image decision (QRD + Query + bbox image) → yes/no -----
                    # tic_images: List[Image.Image] = []
                    tic_prompts = [self.tic_prompt_template.format(QRD=qrd_text, Query=parsed["query"])] * len(prd_images)
                    tic_texts: List[str] = []
                    if prd_images:
                        tic_texts = self._batch_generate_texts(prd_images, tic_prompts, do_sample=False)
                    tic_yes = [self._parse_yes_no(t) for t in tic_texts] if tic_texts else []
                    assert len(tic_yes) == len(cand_items), \
                        f"Text-Image Comparison length mismatch: cand={len(cand_items)} vs tic_yes={len(tic_yes)}"

                    # ----- (Step 3-b) Text-Text decision (PRD vs QRD+Query) → yes/no -----
                    ttc_prompts: List[str] = []
                    for prd in prd_texts:
                        ttc_prompts.append(
                            self.ttc_prompt_template.format(Query=parsed["query"], QRD=qrd_text, Proposal_desc=prd)
                        )
                    ttc_texts: List[str] = []
                    if ttc_prompts:
                        # text-only: images=None
                        ttc_texts = self._batch_generate_texts([None] * len(ttc_prompts), ttc_prompts, do_sample=False)
                    ttc_yes = [self._parse_yes_no(text) for text in ttc_texts] if ttc_texts else []
                    assert len(ttc_yes) == len(cand_items), \
                        f"Text-Text Comparison length mismatch: cand={len(cand_items)} vs ttc_yes={len(ttc_yes)}"

                    # ----- Final Selection (AND) -----
                    final_indices: List[int] = []
                    for idx_c in range(len(cand_items)):
                        tic_match = tic_yes[idx_c] 
                        ttc_match = ttc_yes[idx_c] 
                        if self.require_both_selection:
                            if tic_match and ttc_match:
                                final_indices.append(idx_c)
                        else:
                            if tic_match or ttc_match:
                                final_indices.append(idx_c)

                    if not final_indices:
                        learnability_r[i] = 0.0
                        continue
                    
                    # ----- Nested suppression -----
                    selected_items = [cand_items[k] for k in final_indices]
                    selected_items, suppressed_cnt = self._suppress_nested_masks(
                        selected_items, contain_thr=self.nested_contain_thr
                    )
                    metrics[f"{problem_type}/nested_suppressed"] = int(suppressed_cnt)
                    
                    if not selected_items:
                        learnability_r[i] = 0.0
                        continue

                    # ----- new_task 구성: src=='ml'이면 MLLM bbox/point, src=='amg'이면 mask 기반 bbox/centroid -----
                    final_masks: List[np.ndarray] = []
                    gt_list: List[Dict[str, Any]] = []
                    for item in selected_items:
                        mask = item["mask"]
                        final_masks.append(mask)
                        if item["src"] == "ml":
                            ml_idx = int(item.get("ml_idx", -1))
                            if 0 <= ml_idx < len(parsed["targets"]):
                                target = parsed["targets"][ml_idx]
                                box = [int(round(v)) for v in target.get("box", self._mask_to_bbox(mask))]
                                pt = [int(round(p)) for p in target.get("point", self._mask_centroid(mask))]
                                label = target.get("label", "")
                            else:
                                box = self._mask_to_bbox(mask)
                                pt = self._mask_centroid(mask)
                                label = ""
                        else:  # From Autometic Mask Generation
                            box = self._mask_to_bbox(mask)
                            pt = self._mask_centroid(mask)
                            label = ""
                        gt_list.append({"bbox_2d": box, "point_2d": pt, "label": label})

                    gt_json = json.dumps(gt_list)
                    new_tasks.append({
                        "rtype": rtype, "query": parsed["query"], "image": image,
                        "gt_json": gt_json, "gt_masks": final_masks,  "learnability": None,
                        "step": self.global_steps
                    })
                    
                    accepted_indices.append(i)

                # task pool 업데이트 (최대 길이 유지)
                # self._recent_additions[rtype] = len(new_tasks)
                # if new_tasks:
                #     self.task_pool.extend(new_tasks)
                #     if len(self.task_pool) > self.max_task_pool:
                #         self.task_pool = self.task_pool[-self.max_task_pool:]

                metrics[f"{problem_type}/accepted"] = len(new_tasks)
                metrics[f"{problem_type}/proposed"] = B
                metrics[f"{problem_type}/format_reward/mean"] = float(format_r.mean()) if B > 0 else 0.0

                # (b) learnability: solver를 N회 샘플링, 정답률 평균
                if accepted_indices:
                    # 동일 이미지/질문으로 N회 구성 → RLHFDataset 경유로 생성/평가
                    rep_rows: List[Dict[str, Any]] = []
                    for i, idx in enumerate(accepted_indices):
                        task = new_tasks[i]
                        for _ in range(self.n_learnability_samples):
                            rep_rows.append({
                                "problem": task["query"],
                                "image": task["image"],
                                "solution": task["gt_json"],
                                "solution_mask": task["gt_masks"],
                                "rtype": task["rtype"],
                                "query": task["query"],
                                "task_id": str(id(task)),
                            })
                    rep_loader = self._rows_to_solver_dataloader(
                        rep_rows,
                        shuffle=False,
                        batch_size=len(rep_rows),
                        drop_last=False,
                    )
                    rep_batch_dict = next(iter(rep_loader))
                    rep_dp = DataProto.from_single_dict(rep_batch_dict)
                    rep_gen = rep_dp.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["pixel_values", "image_grid_thw", "raw_prompt_ids", "images"],
                    )
                    rep_gen.meta_info = {
                        'eos_token_id': self.tokenizer.eos_token_id,
                        'pad_token_id': self.tokenizer.pad_token_id,
                        'recompute_log_prob': False,
                        'do_sample': True,
                        'validate': False,
                    }
                    rep_pad, pad_sz = pad_dataproto_to_divisor(rep_gen, self.actor_rollout_wg.world_size)
                    rep_out_pad = self.actor_rollout_wg.generate_sequences(rep_pad)
                    rep_out = unpad_dataproto(rep_out_pad, pad_size=pad_sz)

                    # per-sample evaluation
                    acc_per_task: Dict[str, List[int]] = defaultdict(list)
                    responses = rep_out.batch["responses"]
                    attention_mask = rep_gen.batch["attention_mask"]
                    has_prompts = "prompts" in rep_dp.batch
                    for j in range(len(rep_dp)):
                        response_ids = responses[j]
                        response_len = response_ids.shape[-1]
                        if has_prompts:
                            prompt_len = int(rep_dp.batch["prompts"][j].shape[-1])
                            valid_resp_len = int(attention_mask[j, prompt_len:].sum().item())
                        else:
                            valid_resp_len = int(attention_mask[j, -response_len:].sum().item())
                        response_txt = self.tokenizer.decode(response_ids[:valid_resp_len], skip_special_tokens=True)
                        gt = rep_dp.non_tensor_batch["solution"][j]
                        gt_masks = rep_dp.non_tensor_batch["solution_mask"][j]
                        img = rep_dp.non_tensor_batch["image"][j]
                        _, details = self.sam_wg.compute_reward(response_txt, gt, gt_masks, img)
                        # mask_score >= 1.0 ↔ IoU > 0.3
                        is_correct = 1 if details.get("mask_iou", 0.0) >= 1.0 else 0
                        task_id = rep_dp.non_tensor_batch["task_id"][j]
                        acc_per_task[task_id].append(is_correct)

                    # 평균 정답률 → learnability, ref buffer 통계도 업데이트(근사)
                    for i, idx in enumerate(accepted_indices):
                        task = new_tasks[i]
                        task_id = str(id(task))
                        acc_list = acc_per_task.get(task_id, [])
                        mean_acc = float(sum(acc_list) / max(1, self.n_learnability_samples))
                        learnability = 1.0 - 2.0 * abs(mean_acc - 0.5)
                        learnability_r[idx] = max(0.0, min(1.0, learnability))
                        # task_pool에도 저장 (frontier 샘플링에 사용)
                        task["learnability"] = float(learnability_r[idx])
                        task["mean_acc"] = float(mean_acc)
                        

                    metrics[f"{problem_type}/learnability_mean"] = float(learnability_r.mean())
                
                # === (1) learnability > 0 인 task만 task_pool에 반영 ===
                signal_ok_tasks = [t for t in new_tasks if float(t.get("learnability", 0.0)) > 0.0]
                self._recent_additions[rtype] = len(signal_ok_tasks)
                if signal_ok_tasks:
                    self.task_pool.extend(signal_ok_tasks)
                    if len(self.task_pool) > self.max_task_pool:
                        self.task_pool = self.task_pool[-self.max_task_pool:]
                # 모니터링: selection 통과 수 vs. learnability 통과 수
                metrics[f"{problem_type}/accepted_signal_ok"] = len(signal_ok_tasks)

                # 보상 텐서 구성(마지막 토큰에만 부여)
                # shape: (B, resp_max_len)
                reward_tensor = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
                for i in range(B):
                    L = valid_response_len[i]
                    if L > 0:
                        final_r = float(format_r[i] + 2 * learnability_r[i])  # proposer total reward
                        reward_tensor[i, L - 1] = final_r
                    
                    if already_print < num_examine:
                        already_print += 1
                        print("[prompt]", decoded_prompt_texts[i])
                        print("[response]", decoded_response_texts[i])
                        print("[score]", final_r)
                        
                batch.batch["token_level_scores"] = reward_tensor
                
                # === Proposer 모니터링 로그 저장 (per-iteration) ===
                if self.save_proposer_outputs:
                    log_path = os.path.join(self.proposer_monitor_dir, f"step_{self.global_steps:06d}.jsonl")
                    with open(log_path, "a", encoding="utf-8") as f_log:
                        for i in range(B):
                            rec = {
                                "step": int(self.global_steps),
                                "rtype": rtype,
                                "format_reward": float(format_r[i]),
                                "accepted": bool(i in accepted_indices),
                                "learnability": float(learnability_r[i]) if i in accepted_indices else None,
                            }
                            # 태그별 텍스트 추출
                            txt = decoded_response_texts[i]
                            cap_m = re.search(r"<caption>\s*(.*?)\s*</caption>", txt, re.DOTALL)
                            think_m = re.search(r"<think>\s*(.*?)\s*</think>", txt, re.DOTALL)
                            ans_m = re.search(r"<answer>\s*(.*?)\s*</answer>", txt, re.DOTALL)
                            rec["caption"] = cap_m.group(1).strip() if cap_m else ""
                            rec["think"] = think_m.group(1).strip() if think_m else ""
                            rec["answer_text"] = ans_m.group(1).strip() if ans_m else ""
                            # query(파싱 성공 시)
                            parsed = self._parse_proposer_output(txt)
                            rec["query"] = parsed["query"] if parsed else ""
                            f_log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    
                    step_dir = os.path.join(self.proposer_monitor_dir, f"step_{self.global_steps:06d}")
                    os.makedirs(step_dir, exist_ok=True)
                    for i in accepted_indices:
                        lr_val = float(learnability_r[i])
                        
                        # img_i = gen_batch[i].non_tensor_batch.get("images", None)
                        img_i = gen_batch[i].non_tensor_batch["images"][0]

                        # parsing
                        parsed_i = self._parse_proposer_output(decoded_response_texts[i])
                        query_i = parsed_i.get("query", "") if parsed_i else ""
                        targets_i = parsed_i.get("targets", []) if parsed_i else []
                        
                        try:
                            t_idx = accepted_indices.index(i)
                            masks_i = new_tasks[t_idx]["gt_masks"]
                        except Exception:
                            masks_i = None
                        n_masks = len(masks_i) if masks_i is not None else 0
                        out_name = f"img_{i:03d}_{rtype}_lr{lr_val}_n_masks_{n_masks}.png"
                        out_path = os.path.join(step_dir, out_name)
                        self._save_proposer_visual(
                            image=img_i,
                            query=query_i,
                            targets=targets_i,
                            masks=masks_i,
                            out_path=out_path,
                            accepted=True,
                        )        

        else:
            # PRED: 당신의 reward_fn(z_star)을 호출
            with _timer(f"reward/{problem_type}", timing_raw):
                reward_output = self.reward_fn(batch)
                if isinstance(reward_output, tuple):
                    reward_tensor, reward_stats = reward_output
                    metrics.update({f"{problem_type}/rewards/{k}": v for k, v in reward_stats.items()})
                else:
                    reward_tensor = reward_output
                batch.batch["token_level_scores"] = reward_tensor

        # ref policy
        if self.use_reference_policy:
            with _timer(f"ref/{problem_type}", timing_raw):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        # values
        if self.use_critic:
            with _timer(f"values/{problem_type}", timing_raw):
                values = self.critic_wg.compute_values(batch)
                batch = batch.union(values)

        # KL / rewards → advantages
        if self.config.worker.actor.use_kl_loss:
            batch, kl_metrics = apply_kl_penalty(
                batch, kl_ctrl=self.kl_ctrl, kl_penalty=self.config.algorithm.kl_penalty
            )
            metrics.update({f"{problem_type}/{k}": v for k, v in kl_metrics.items()})
        else:
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

        with _timer(f"adv/{problem_type}", timing_raw):
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.worker.rollout.n,
            )
        
        if (not is_gen) and self.use_learnability_adv_weight and ("advantages" in batch.batch):
            learnability_r = batch.non_tensor_batch.get("learnability", None)
            print("-"*10, learnability_r)
            adv = batch.batch["advantages"]
            device = adv.device
            
            w_list = []
            for l in learnability_r:
                l = 0.0 if l is None else float(l)
                w = l
                w_list.append(w)
            w = torch.tensor(w_list, dtype=adv.dtype, device=device).unsqueeze(1)  # (B,1)
            # 전체 토큰에 동일 가중치(프롬프트 구간은 어차피 0인 경우가 대부분)
            batch.batch["advantages"] = adv * w

        batch = self._sanitize_non_tensor_for_concat(batch)
        
        return batch, metrics

    # ====== (4) 학습 루프 ======
    def fit(self):
        """
          for step:
            for each type in self.rs_types:
              (gen dataloader, pred dataloader 생성)
              update_iteration 만큼 반복:
                gen → pred 결과들을 concat → critic/actor update
        """
        # 기본 초기화/검증/체크포인트 로드는 부모에서 하던 방식 유지
        logger = self._init_logger()
        
        self.global_steps = 0
        total_training_steps = self.config.trainer.total_iters
        
        if not self.ref_in_actor:
            self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.val_before_train and self.global_steps == 0:
            val_metrics = self._validate()
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.val_only:
                return

        # training loop
        while self.global_steps < total_training_steps:
            self.global_steps += 1
            metrics = {}
            timing_raw = {}

            data_len = self.config.data.rollout_batch_size * self.update_iteration
            
            self._prev_recent_additions = defaultdict(int, self._recent_additions)
            self._recent_additions = defaultdict(int)

            # per-type dataloader preparation
            gen_loaders = {}
            pred_loaders = {}
            for rtype in self.rs_types:
                # gen은 항상 시도(accept되지 않은 샘플은 보상 0)
                gen_loaders[rtype] = self._create_train_rs_gen_dataloader(rtype, data_len)
                pred_loaders[rtype] = self._create_train_rs_pred_dataloader(rtype, data_len)
            
            for rtype in self.rs_types:
                if rtype not in self._first_pred_iter_by_type and pred_loaders[rtype] is not None:
                    self._first_pred_iter_by_type[rtype] = self.global_steps
                    print(f"[Self-Play] Solver for type='{rtype}' begins at step {self.global_steps}")

            # update_iteration
            with _timer("step", timing_raw):
                batches_per_role = []

                for _ in range(self.update_iteration):
                    # 각 type 마다 gen→pred 처리
                    for rtype in self.rs_types:
                        # GEN
                        batch_dict = next(gen_loaders[rtype])
                        batch = DataProto.from_single_dict(batch_dict)
                        batch, metrics = self._compute_batch(batch, metrics, timing_raw,
                                                                problem_type=f"gen_{rtype}")
                        allow_prop_update = False
                        if self.train_proposer:
                            if self.global_steps <= self.proposer_update_after_step:
                                allow_prop_update = True
                            elif (self.global_steps % self.proposer_update_every) == 0:
                                allow_prop_update = True
                        if allow_prop_update:
                            batches_per_role.append(batch)

                        # PRED
                        if pred_loaders[rtype] is not None:
                            batch_dict = next(pred_loaders[rtype])
                            batch = DataProto.from_single_dict(batch_dict)
                            batch, metrics = self._compute_batch(batch, metrics, timing_raw,
                                                                    problem_type=f"pred_{rtype}")
                            batches_per_role.append(batch)
                        
                if len(batches_per_role) == 0:
                    # 아직 task pool이 비거나 데이터가 부족한 경우
                    logger.log(data={"warn/empty_batches": 1}, step=self.global_steps)
                    continue
                
                # concatenate batches
                big_batch = DataProto.concat(batches_per_role)

                # critic
                if self.use_critic:
                    with _timer("update_critic", timing_raw):
                        c_out = self.critic_wg.update_critic(big_batch)
                    metrics.update(reduce_metrics(c_out.meta_info["metrics"]))

                # actor
                if self.config.trainer.critic_warmup <= self.global_steps:
                    with _timer("update_actor", timing_raw):
                        a_out = self.actor_rollout_wg.update_actor(big_batch)
                    metrics.update(reduce_metrics(a_out.meta_info["metrics"]))

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with _timer("testing", timing_raw):
                        val_metrics = self._validate()
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                    with _timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # 데이터/타이밍/length 메트릭 로깅
            metrics.update(compute_timing_metrics(big_batch, timing_raw))

            for rtype in self.rs_types:
                tp = [t for t in self.task_pool
                      if t.get("rtype") == rtype and isinstance(t.get("mean_acc", None), (int, float))]
                if tp:
                    metrics[f"selfplay/ref_mean_acc/{rtype}"] = float(np.mean([t["mean_acc"] for t in tp]))
                    lr_vals = [t.get("learnability") for t in tp if isinstance(t.get("learnability"), (int, float))]
                    if lr_vals:
                        metrics[f"selfplay/ref_mean_learnability/{rtype}"] = float(np.mean(lr_vals))
            logger.log(data=metrics, step=self.global_steps)

        # 종료 전 최종 검증/저장
        if self.val_reward_fn is not None:
            val_metrics = self._validate()
            logger.log(data=val_metrics, step=self.global_steps)
        self._save_checkpoint()

    # ====== 부모의 로거 초기화 포맷 유지 ======
    def _init_logger(self):
        from verl.utils.tracking import Tracking
        return Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=self.config.to_dict(),
        )