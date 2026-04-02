"""
Build a Hugging Face Dataset from polygon-based Q/A dataset.

Each image record contains:
- "id", "height", "width", "coco_url", "questions", and "answer"/"answers".
- "answer"/"answers" is List[List[Dict]] where each inner list corresponds to a question.
  Each dict may contain "id", "segmentation" (list of polygons), "bbox", etc.

We expand to Q–A pair items:
- image: datasets.Image()  (bytes from coco_url)
- text: str (question)
- mask: 2D bool list  (union of all polygons for that question)
- image_id: int
- ann_id: str (underscore-joined ann ids for that question)
- img_height: int
- img_width: int

Usage:
    python build_hf_dataset_from_answers.py data.json --out ./hf_eval_dataset
    # Then:
    # >>> from datasets import load_from_disk
    # >>> ds = load_from_disk("./hf_eval_dataset")
    # >>> img = ds[0]["image"].convert("RGB")
    # >>> mask = ds[0]["mask"]  # 2D bool list
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from io import BytesIO

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as cocomask
from datasets import Dataset, Features, Sequence, Value, Image as HFImage


# ------------------------- JSON / helpers -------------------------

def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("data", "items", "records"):
            v = data.get(k)
            if isinstance(v, list):
                return v
        return [data]
    raise ValueError("Top-level JSON must be a list or dict.")


def _normalize_polygons(segmentation: Any) -> List[List[float]]:
    """
    Normalize segmentation into a list of polygon lists.
    Accepts either [x1,y1,...] or [[...], [...], ...].
    Returns [] if invalid.
    """
    if not isinstance(segmentation, list) or len(segmentation) == 0:
        return []
    # Single flat polygon
    if all(isinstance(v, (int, float)) for v in segmentation):
        return [segmentation] if len(segmentation) >= 6 else []
    # List of polygons
    polys: List[List[float]] = []
    for poly in segmentation:
        if isinstance(poly, list) and len(poly) >= 6 and all(isinstance(v, (int, float)) for v in poly):
            polys.append(poly)
    return polys


def polygons_to_rles_for_group(answers_group: List[Dict[str, Any]], height: int, width: int):
    """
    Collect RLEs for all polygons across all dicts in one question's answers_group.
    Returns a list of RLEs (possibly empty).
    """
    all_rles = []
    for ann in answers_group:
        seg = ann.get("segmentation", None)
        polys = _normalize_polygons(seg)
        if not polys:
            continue
        rles = cocomask.frPyObjects(polys, height, width)
        if isinstance(rles, dict):  # single polygon case can return dict
            rles = [rles]
        all_rles.extend(rles)
    return all_rles


# ------------------------- Image fetch (with cache) -------------------------

class ImageFetcher:
    def __init__(self, timeout: float = 10.0):
        self.sess = requests.Session()
        self.sess.headers.update({"User-Agent": "hf-qa-builder/1.0"})
        self.timeout = timeout
        self.cache: Dict[str, bytes] = {}

    def fetch_bytes(self, url: str) -> Optional[bytes]:
        if not url:
            return None
        if url in self.cache:
            return self.cache[url]
        try:
            r = self.sess.get(url, timeout=self.timeout)
            r.raise_for_status()
            self.cache[url] = r.content
            return r.content
        except Exception:
            return None


# ------------------------- Build HF dataset -------------------------

def build_dataset(records: List[Dict[str, Any]]) -> Dataset:
    fetcher = ImageFetcher()

    buf_image: List[Dict[str, Optional[bytes]]] = []    # {"bytes": b"..."}
    buf_text: List[str] = []
    buf_mask: List[List[List[bool]]] = []               # 2D bool list
    buf_image_id: List[int] = []
    buf_ann_id: List[str] = []
    buf_h: List[int] = []
    buf_w: List[int] = []

    n_items = 0

    for rec in tqdm(records, desc="Expanding Q-A pairs"):
        # Base fields
        image_id = int(rec.get("id", -1))
        H_json = int(rec.get("height", 0))
        W_json = int(rec.get("width", 0))
        coco_url = rec.get("coco_url", "")

        questions = rec.get("questions", []) or []
        # Support both "answer" and "answers"
        answers_all = rec.get("answers", None)
        if answers_all is None:
            answers_all = rec.get("answer", [])
        if not isinstance(answers_all, list) or not isinstance(questions, list):
            continue

        # Fetch image bytes once per image
        img_bytes = fetcher.fetch_bytes(coco_url)
        if img_bytes is None:
            continue

        # Actual downloaded image size (may rarely differ from JSON)
        try:
            pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
            W_img, H_img = pil_img.size
        except Exception:
            continue

        num_pairs = min(len(questions), len(answers_all))
        for qi in range(num_pairs):
            q_text = questions[qi]
            answers_group = answers_all[qi]
            if not isinstance(answers_group, list) or len(answers_group) == 0:
                continue

            # Collect RLEs across all dicts → union → decode
            try:
                rles_all = polygons_to_rles_for_group(answers_group, H_json, W_json)
            except Exception:
                rles_all = []
            if not rles_all:
                continue

            try:
                merged = cocomask.merge(rles_all)   # union
                m = cocomask.decode(merged)         # (H_json, W_json) uint8 {0,1}
                mask = m.astype(np.uint8)
            except Exception:
                # Fallback: OR individual decodes
                mask = np.zeros((H_json, W_json), dtype=np.uint8)
                for r in rles_all:
                    try:
                        dm = cocomask.decode(r)
                        if dm.ndim == 3:
                            dm = dm[:, :, 0]
                        mask |= (dm.astype(np.uint8) > 0).astype(np.uint8)
                    except Exception:
                        pass

            # Ensure mask shape == image shape (as requested)
            target_h, target_w = H_img, W_img
            if mask.shape != (target_h, target_w):
                # Resize mask to image shape using nearest neighbor
                mask_pil = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
                mask_resized = mask_pil.resize((target_w, target_h), resample=Image.NEAREST)
                mask = (np.array(mask_resized) > 0).astype(np.uint8)

            # Convert to 2D bool list
            mask_bool_2d = (mask > 0).tolist()

            # Build ann_id string by joining all ann ids with underscore
            ann_ids = []
            for ann in answers_group:
                aid = ann.get("id", None)
                if isinstance(aid, int):
                    ann_ids.append(str(aid))
                elif isinstance(aid, str) and aid.strip():
                    ann_ids.append(aid.strip())
            ann_id_joined = "_".join(ann_ids) if ann_ids else ""

            # Append buffers
            buf_image.append({"bytes": img_bytes})
            buf_text.append(q_text)
            buf_mask.append(mask_bool_2d)
            buf_image_id.append(image_id)
            buf_ann_id.append(ann_id_joined)
            buf_h.append(H_json)   # per instruction: from JSON
            buf_w.append(W_json)

            n_items += 1

    features = Features({
        "image": HFImage(),                                # bytes → PIL Image
        "text": Value("string"),
        "mask": Sequence(Sequence(Value("bool"))),         # 2D bool list (H x W of image)
        "image_id": Value("int64"),
        "ann_id": Value("string"),                         # "13446_13447" style
        "img_height": Value("int64"),
        "img_width": Value("int64"),
    })

    ds = Dataset.from_dict(
        {
            "image": buf_image,
            "text": buf_text,
            "mask": buf_mask,
            "image_id": buf_image_id,
            "ann_id": buf_ann_id,
            "img_height": buf_h,
            "img_width": buf_w,
        },
        features=features,
    )
    print(f"[INFO] Total items written: {len(ds)}")
    return ds


def main():
    ap = argparse.ArgumentParser(description="Convert polygon Q/A dataset to Hugging Face Dataset (Arrow).")
    ap.add_argument("json_file", type=str, help="Path to input JSON")
    ap.add_argument("--out", type=str, default="./hf_eval_dataset", help="save_to_disk directory")
    args = ap.parse_args()

    records = load_json(args.json_file)
    ds = build_dataset(records)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))
    print(f"Saved dataset to: {out_dir}")


if __name__ == "__main__":
    main()