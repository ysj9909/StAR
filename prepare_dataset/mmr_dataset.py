"""
JSON 포맷(이미지당 여러 Q/A)을 Q-A pair 단위로 풀어 Hugging Face Dataset(Arrow)로 저장.

Usage:
    python build_hf_dataset_from_json.py data.json --out ./hf_qa_dataset
    # 저장 후:
    # >>> from datasets import load_from_disk
    # >>> ds = load_from_disk("./hf_qa_dataset")
    # >>> for item in ds:
    # ...     image = item["image"].convert("RGB")
    # ...     mask  = item["mask"]  # 2D bool list
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from io import BytesIO
from collections import defaultdict

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as cocomask
from datasets import Dataset, Features, Sequence, Value, Image as HFImage


# ------------------------- JSON / IO -------------------------

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


# ------------------------- Image fetch (remote, cached) -------------------------

class ImageFetcher:
    def __init__(self, timeout: float = 10.0):
        self.sess = requests.Session()
        self.sess.headers.update({"User-Agent": "hf-dataset-builder/1.0"})
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


# ------------------------- RLE → merged mask -------------------------

def decode_merge_rles(rles: List[Dict[str, Any]]) -> Optional[np.ndarray]:
    """
    여러 compressed RLE(dict: {"size":[H,W], "counts": str|bytes}) → (H, W) uint8(0/1)
    OR-merge. 유효 RLE가 없으면 None.
    """
    if not rles:
        return None
    normed = []
    for r in rles:
        size = r.get("size")
        counts = r.get("counts")
        if size is None or counts is None:
            continue
        if isinstance(counts, str):
            counts = counts.encode("utf-8")
        normed.append({"size": size, "counts": counts})
    if not normed:
        return None

    m = cocomask.decode(normed)  # (H,W,N) 또는 (H,W)
    if m.ndim == 2:
        merged = m.astype(bool)
    else:
        merged = (m.sum(axis=2) > 0)
    return merged.astype(np.uint8)


# ------------------------- Build HF dataset -------------------------

def build_dataset(
    records: List[Dict[str, Any]],
) -> Dataset:
    """
    JSON 레코드(이미지 단위)를 Q-A pair 단위의 아이템으로 변환.
    """
    fetcher = ImageFetcher()

    # 누적 버퍼
    buf_image: List[Dict[str, Optional[bytes]]] = []  # {"bytes": b"..."} 형태
    buf_text: List[str] = []
    buf_mask: List[List[List[bool]]] = []            # 2D bool list
    buf_image_id: List[int] = []
    buf_ann_id: List[List[int]] = []                 # 여러 ann id 가능 → list[int]
    buf_h: List[int] = []
    buf_w: List[int] = []
    buf_file_name: List[str] = []

    total_items = 0

    for item in tqdm(records, desc="Expanding Q-A pairs"):
        # 공통 필드
        file_name: str = item.get("file_name", "")
        coco_url: str = item.get("coco_url", "")
        image_id: int = int(item.get("image_id", -1))
        H: int = int(item.get("height", 0))
        W: int = int(item.get("width", 0))

        questions = item.get("questions", [])
        answers = item.get("answers", [])
        if not isinstance(questions, list) or not isinstance(answers, list):
            continue

        num_pairs = min(len(questions), len(answers))
        if num_pairs == 0:
            continue

        # 이미지 원격 다운로드 (한 번만)
        img_bytes = fetcher.fetch_bytes(coco_url)
        if img_bytes is None:
            # COCO URL 다운로드 실패 시 스킵
            continue

        # Q/A 각각을 아이템으로
        for qi in range(num_pairs):
            q_text = questions[qi]
            ans_group = answers[qi]  # 여러 ann(dict)의 리스트
            if not isinstance(ans_group, list) or len(ans_group) == 0:
                continue

            # RLE 수집
            rles = []
            ann_ids: List[int] = []
            for ann in ans_group:
                seg = ann.get("segmentation")
                if isinstance(seg, dict) and "counts" in seg and "size" in seg:
                    rles.append(seg)
                ann_id_val = ann.get("id", None)
                if isinstance(ann_id_val, int):
                    ann_ids.append(ann_id_val)

            if not rles:
                # 이 Q는 마스크가 없으면 스킵(요구사항: mask 열에 2D bool)
                continue

            # 마스크 디코드 & OR-merge
            merged = decode_merge_rles(rles)  # uint8 (0/1)
            if merged is None:
                continue

            # 크기 안전 체크 (드물게 size 불일치 대비)
            if merged.shape != (H, W):
                # nearest로 강제 맞춤
                from cv2 import resize, INTER_NEAREST
                merged = resize(merged, (W, H), interpolation=INTER_NEAREST)

            # 2D bool list로 변환 (Arrow 저장 호환)
            mask_bool_2d: List[List[bool]] = merged.astype(bool).tolist()

            # 누적
            buf_image.append({"bytes": img_bytes})
            buf_text.append(q_text)
            buf_mask.append(mask_bool_2d)
            buf_image_id.append(image_id)
            buf_ann_id.append(ann_ids)  # 없으면 빈 리스트
            buf_h.append(H)
            buf_w.append(W)
            buf_file_name.append(file_name)

            total_items += 1

    # HF Datasets 생성
    features = Features({
        "image": HFImage(),                                # PIL로 바로 로드 가능
        "text": Value("string"),
        "mask": Sequence(Sequence(Value("bool"))),         # 2D bool 리스트
        "image_id": Value("int64"),
        "ann_id": Sequence(Value("int64")),                # 여러 ann id 가능
        "img_height": Value("int64"),
        "img_width": Value("int64"),
        "file_name": Value("string"),
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
            "file_name": buf_file_name,
        },
        features=features,
    )
    # image 컬럼은 features로 이미 Image() 지정되어 있어, item["image"]는 PIL Image로 제공됨
    return ds, total_items


def main():
    ap = argparse.ArgumentParser(description="Build Hugging Face Dataset (Arrow) from JSON Q/A pairs.")
    ap.add_argument("json_file", type=str, help="입력 JSON 경로")
    ap.add_argument("--out", type=str, default="./hf_qa_dataset", help="save_to_disk 경로")
    args = ap.parse_args()

    records = load_json(args.json_file)
    ds, n_items = build_dataset(records)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))

    print(f"\nSaved dataset to: {out_dir}")
    print(f"Total items written: {n_items}")
    print(ds)


if __name__ == "__main__":
    main()