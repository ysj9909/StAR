import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from math import isfinite

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from scipy.optimize import linear_sum_assignment

from collections import defaultdict
from tensordict import TensorDict

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, Execute, register
from verl.utils.reward_score.z_star import SAM_MODEL_CFG, SAM_CHECKPOINT


class SAMPredictorWorker(Worker):
    def __init__(self, config, role: str = "sam", *_, **__):
        super().__init__()
        
        # cache for auto mask generator
        self._auto_gen = None
        self._auto_cfg_sig = None
    
    # rank-0 GPU 한 대만 실행
    # @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.RANK_ZERO, blocking=True)
    @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.ALL, blocking=True)
    def init_model(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.predictor = SAM2ImagePredictor(build_sam2(SAM_MODEL_CFG, SAM_CHECKPOINT))
        self.predictor.model.to(device).eval()
        for p in self.predictor.model.parameters():
            p.requires_grad_(False)
    
    def _compute_reward_single(
        self,
        predict_str: str,
        ground_truth: str,
        ground_truth_masks,
        image,
        is_qwen3: bool = False,
        is_stage2: bool = False,
    ):
        """단일 샘플 STAR reward 계산 로직."""
        format_reward = _vision_reasoner_format_reward(predict_str)
        non_repeat_reward = _vision_reasoner_non_repeat_reward(predict_str)

        if not is_stage2:  # 1st stage training
            (
                accuracy_reward,
                bbox_iou_score,
                bbox_l1_score,
                point_score,
                mask_score,
                count_match_score,
            ) = star_accuracy_reward(
                predict_str,
                ground_truth,
                ground_truth_masks,
                image,
                self.predictor,
                is_qwen3=is_qwen3,
            )

            details = {
                "format": format_reward,
                "accuracy": accuracy_reward,
                "bbox_iou": bbox_iou_score,
                "bbox_l1": bbox_l1_score,
                "point_l1": point_score,
                "mask_iou": mask_score,
                "count_match": count_match_score,
                "non_repeat": non_repeat_reward,
            }
        else:  # 2nd stage training
            accuracy_reward, mask_score, pred_masks = star_accuracy_reward(
                predict_str,
                ground_truth,
                ground_truth_masks,
                image,
                self.predictor,
                is_qwen3=is_qwen3,
                is_stage2=is_stage2,
            )

            details = {
                "format": format_reward,
                "accuracy": accuracy_reward,
                # "pred_masks": pred_masks,
                "mask_iou": mask_score,
                "non_repeat": non_repeat_reward,
            }

        score = float(format_reward + accuracy_reward + non_repeat_reward)
        return score, details

    # 기존 단일 샘플용 API (호환성 유지)
    @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.RANK_ZERO, blocking=True)
    def compute_reward(
        self,
        predict_str: str,
        ground_truth: str,
        ground_truth_masks,
        image,
        is_qwen3: bool = False,
        is_stage2: bool = False,
    ) -> float:
        score, details = self._compute_reward_single(
            predict_str,
            ground_truth,
            ground_truth_masks,
            image,
            is_qwen3=is_qwen3,
            is_stage2=is_stage2,
        )
        return score, details

    # DataProto를 world_size로 split해서 각 GPU가 일부 샘플을 처리
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, execute_mode=Execute.ALL, blocking=False)
    def compute_reward_batch(self, data: DataProto) -> DataProto:
        """
        DataProto 입력:
          - non_tensor_batch:
              'predict_str'   : np.ndarray[object] (str)
              'ground_truth'  : np.ndarray[object] (str or None)
              'solution_mask' : GT mask
              'image'         : 원본 이미지
          - meta_info:
              'is_qwen3'  : bool
              'is_stage2' : bool

        반환:
          - DataProto:
              batch['scores']       : (local_batch,) float32 tensor
              non_tensor_batch[...] : metric 리스트 / ndarray
        """
        num = len(data)
        device = torch.device("cpu")
        scores = torch.empty(num, dtype=torch.float32, device=device)
        detail_lists: Dict[str, list] = defaultdict(list)

        is_qwen3 = bool(data.meta_info.get("is_qwen3", False))
        is_stage2 = bool(data.meta_info.get("is_stage2", False))

        predict_arr = data.non_tensor_batch["predict_str"]
        gt_arr = data.non_tensor_batch["ground_truth"]
        mask_arr = data.non_tensor_batch["solution_mask"]
        # img_arr = data.non_tensor_batch["image"]
        img_arr = data.batch['images']

        for i in range(num):
            predict_str = predict_arr[i]
            ground_truth = gt_arr[i]
            gt_masks = mask_arr[i]
            image = img_arr[i]

            score, details = self._compute_reward_single(
                predict_str,
                ground_truth,
                gt_masks,
                image,
                is_qwen3=is_qwen3,
                is_stage2=is_stage2,
            )
            scores[i] = score
            for k, v in details.items():
                detail_lists[k].append(v)

        non_tensors: Dict[str, Any] = {}
        for k, v in detail_lists.items():
            if len(v) == 0:
                continue
            if isinstance(v[0], (np.ndarray, list, dict)):
               non_tensors[k] = v
            else:
                non_tensors[k] = np.asarray(v, dtype=np.float32)

        batch = TensorDict(
           source={"scores": scores},
            batch_size=(num,),
        )

        return DataProto(batch=batch, non_tensor_batch=non_tensors, meta_info={})
            
    @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.RANK_ZERO, blocking=True)
    def predict_masks_from_points_boxes(self, image, points: List[List[float]], boxes: List[List[float]]):
        """
        Args:
            image : PIL.Image
            points: [[x,y], ...] or None
            boxes : [[x1,y1,x2,y2], ...]
        Returns:
            List[np.ndarray(H,W,bool)] in the same order
        """
        pil_img = image.convert("RGB")
        np_img = np.array(pil_img, dtype=np.uint8)
        self.predictor.set_image(np_img)

        masks = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if points is None:
                iter_points = [None] * len(boxes)
            else:
                iter_points = points
            for point, bbox in zip(iter_points, boxes):
                # try:
                pred_masks, scores, _ = self.predictor.predict(
                    point_coords=[point.tolist()] if point is not None else point,
                    point_labels=[1] if point is not None else point,
                    box=bbox.tolist(),
                )
                mask = pred_masks[np.argmax(scores)].astype(bool)
                # except Exception:
                #     mask = np.zeros((np_img.shape[0], np_img.shape[1]), dtype=bool)
                masks.append(mask)
        return masks
    
    @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.RANK_ZERO, blocking=True)
    def generate_auto_masks(self, image, cfg: Optional[dict] = None):
        """
        SAM2AutomaticMaskGenerator based autometic mask generation.
        Args:
            image: PIL.Image
            cfg: dict (points_per_side, points_per_batch, pred_iou_thresh, stability_score_thresh,
                       stability_score_offset, crop_n_layers, box_nms_thresh,
                       crop_n_points_downscale_factor, min_mask_region_area, use_m2m, ...)
        Returns:
            List[np.ndarray(H,W,bool)]
        """
        if not hasattr(self, "predictor"):
            raise RuntimeError("SAM predictor not initialized. Call init_model() first.")

        pil_img = image.convert("RGB")
        np_img = np.array(pil_img, dtype=np.uint8)

        # default setting
        default_cfg = dict(
            points_per_side=64,
            points_per_batch=128,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.9,
            stability_score_offset=0.7,
            crop_n_layers=1,
            box_nms_thresh=0.7,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=25.0,
            use_m2m=True,
        )
        user_cfg = dict(default_cfg)
        if isinstance(cfg, dict):
            user_cfg.update({k: v for k, v in cfg.items() if v is not None})

        # 생성기 캐싱 (cfg 시그니처가 같으면 재사용)
        cfg_sig = tuple(sorted(user_cfg.items()))
        if self._auto_gen is None or self._auto_cfg_sig != cfg_sig:
            self._auto_gen = SAM2AutomaticMaskGenerator(
                model=self.predictor.model,
                **user_cfg,
            )
            self._auto_cfg_sig = cfg_sig

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks = self._auto_gen.generate(np_img)

        H, W = np_img.shape[:2]
        min_pixels = int(np.ceil(H * W * 0.005))
        
        out = []
        for m in masks:
            seg = m.get("segmentation", None)
            seg = seg.astype(bool)
            if seg.sum() >= min_pixels:
                out.append(seg)
        return out

   
def _vision_reasoner_format_reward(predict_str: str) -> float:
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    match = re.fullmatch(pattern, predict_str, re.DOTALL)
    thinking_format_reward = 1.0 if match else 0.0

    def segmentation_format(predict_str: str) -> float:
        segmentation_format_reward = 0.0
        try:
            json_match = re.search(r"<answer>\s*(.*?)\s*</answer>", predict_str, re.DOTALL)
            if not json_match:
                return segmentation_format_reward
            data = json.loads(json_match.group(1))

            data_cnt = len(data)

            for item in data:
                cur_reward = 0.0
                
                if "bbox_2d" in item:
                    bbox_2d = item["bbox_2d"]
                    if isinstance(bbox_2d, list) and len(bbox_2d) == 4:
                        cur_reward += (2.0 / 3.0)

                if "point_2d" in item:
                    point_2d = item["point_2d"]
                    if isinstance(point_2d, list) and len(point_2d) == 2:
                        cur_reward += (2.0 / 3.0)
                
                if "label" in item:
                    label = item["label"]
                    if isinstance(label, str) and label.strip() != "":
                        cur_reward += (2.0 / 3.0)

                segmentation_format_reward += cur_reward / data_cnt
        except Exception:
            pass
        return segmentation_format_reward

    segmentation_format_reward = segmentation_format(predict_str)

    return thinking_format_reward + segmentation_format_reward


def star_accuracy_reward(
    predict_str: str,
    ground_truth: str,
    ground_truth_masks: Optional[List[Any]] = None,
    image: Optional[Any] = None,
    predictor = None,
    is_qwen3: bool = False,
    is_stage2: bool = False,
) -> float:
    max_accuracy_reward = 0.0
    bbox_iou_score = 0.0
    bbox_l1_score = 0.0
    point_score = 0.0
    mask_score = 0.0
    count_match_score = 0.0
    MAX_OBJECTS = 120
    
    pred_masks = None  
    
    # ---------------- Stage 2: mask-only reward path ----------------
    if is_stage2:
        try:
            json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                pred_bboxes = [item['bbox_2d'] for item in data]
                pred_points = [item['point_2d'] for item in data]

                if len(pred_bboxes) > MAX_OBJECTS:
                    pred_bboxes = pred_bboxes[:MAX_OBJECTS]
                    pred_points = pred_points[:MAX_OBJECTS]

                if ground_truth_masks is not None and len(ground_truth_masks) > MAX_OBJECTS:
                    ground_truth_masks = ground_truth_masks[:MAX_OBJECTS]

                if is_qwen3:
                    pred_bboxes, pred_points = _convert_qwen3_predictions_to_pixels(
                        pred_bboxes, pred_points, image, default_w=840, default_h=840, force_grid=None
                    )
                else:
                    pred_bboxes = np.array(pred_bboxes)
                    pred_points = np.array(pred_points)

                if ground_truth_masks is not None and image is not None and len(pred_bboxes) > 0:
                    # pil_img = image.convert("RGB")
                    # np_img = np.array(pil_img, dtype=np.uint8)
                    np_img = np.array(image)
                    predictor.set_image(np_img)

                    pred_masks = []
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        for bbox, point in zip(pred_bboxes, pred_points):
                            try:
                                masks, scores, _ = predictor.predict(
                                    point_coords=[point.tolist()],
                                    point_labels=[1],
                                    box=bbox.tolist(),
                                )
                                mask = masks[np.argmax(scores)].astype(bool)
                            except Exception:
                                mask = np.zeros((np_img.shape[0], np_img.shape[1]), dtype=bool)
                            pred_masks.append(mask)
                    
                    pred_masks = np.array(pred_masks)
                    gt_masks = np.array(ground_truth_masks).astype(bool)
                    mask_iou = batch_mask_iou(pred_masks, gt_masks)

                    mask_reward = np.zeros_like(mask_iou)
                    mask_reward[(mask_iou > 0.3) & (mask_iou <= 0.5)] = 1.0
                    mask_reward[(mask_iou > 0.5) & (mask_iou <= 0.7)] = 2.0
                    mask_reward[(mask_iou > 0.7) & (mask_iou <= 0.8)] = 3.0
                    mask_reward[(mask_iou > 0.8) & (mask_iou <= 0.9)] = 4.0
                    mask_reward[mask_iou > 0.9] = 5.0

                    # Stage2: total_matrix is mask_reward ONLY; no bbox/point/count-match used.
                    total_matrix = mask_reward
                    # Use a compatible cost so that higher reward => lower cost (mask_reward ∈ [0,5]).
                    cost_matrix = 5.0 - total_matrix

                    row_indices, col_indices = linear_sum_assignment(cost_matrix)
                    total_reward = total_matrix[row_indices, col_indices].sum()

                    max_length = max(len(pred_masks), len(gt_masks))
                    if max_length > 0:
                        mask_score = total_reward / max_length
                        max_accuracy_reward = 2 * mask_score
        except Exception:
            pass
        return max_accuracy_reward, mask_score, pred_masks

    # ---------------- Stage1 training path ----------------
    
    try:
        gt_data = json.loads(ground_truth)
        gt_bboxes = [item['bbox_2d'] for item in gt_data]
        gt_points = [item['point_2d'] for item in gt_data]

        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            pred_bboxes = [item['bbox_2d'] for item in data]
            pred_points = [item['point_2d'] for item in data]

            if len(pred_bboxes) > MAX_OBJECTS:
                pred_bboxes = pred_bboxes[:MAX_OBJECTS]
                pred_points = pred_points[:MAX_OBJECTS]

            if len(gt_bboxes) > MAX_OBJECTS:
                gt_bboxes = gt_bboxes[:MAX_OBJECTS]
                gt_points = gt_points[:MAX_OBJECTS]
                if ground_truth_masks is not None:
                    ground_truth_masks = ground_truth_masks[:MAX_OBJECTS]
            
            if is_qwen3:
                pred_bboxes, pred_points = _convert_qwen3_predictions_to_pixels(
                    pred_bboxes, pred_points, image, default_w=840, default_h=840, force_grid=None
                )
            else:
                pred_bboxes = np.array(pred_bboxes)
                pred_points = np.array(pred_points)
            
            gt_bboxes = np.array(gt_bboxes)
            gt_points = np.array(gt_points)

            iou_matrix = batch_iou(pred_bboxes, gt_bboxes)
            l1_matrix = batch_l1_distance(pred_bboxes, gt_bboxes)
            points_dist_matrix = batch_points_distance(pred_points, gt_points)
            points_in_box = batch_points_in_box(pred_points, pred_bboxes)
            bbox_iou_reward = (iou_matrix > 0.5).astype(float)
            bbox_l1_reward = (l1_matrix < 10).astype(float)
            point_reward = ((points_dist_matrix < 30) & points_in_box[:, np.newaxis]).astype(float)
            
            if ground_truth_masks is not None and image is not None and len(pred_bboxes) > 0:
                # pil_img = image.convert("RGB")
                # np_img = np.array(pil_img, dtype=np.uint8)
                np_img = np.array(image)
                predictor.set_image(np_img)
                
                pred_masks = []
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for bbox, point in zip(pred_bboxes, pred_points):
                        try:
                            masks, scores, _ = predictor.predict(
                                point_coords=[point.tolist()],
                                point_labels=[1],
                                box=bbox.tolist(),
                            )
                            mask = masks[np.argmax(scores)].astype(bool)
                        except Exception:
                            mask = np.zeros((np_img.shape[0], np_img.shape[1]), dtype=bool)
                        pred_masks.append(mask)
                
                pred_masks = np.array(pred_masks)
                gt_masks = np.array(ground_truth_masks).astype(bool)
                mask_iou = batch_mask_iou(pred_masks, gt_masks)

                mask_reward = np.zeros_like(mask_iou)
                mask_reward[(mask_iou > 0.3) & (mask_iou <= 0.5)] = 1.0
                mask_reward[(mask_iou > 0.5) & (mask_iou <= 0.7)] = 2.0
                mask_reward[(mask_iou > 0.7) & (mask_iou <= 0.8)] = 3.0
                mask_reward[(mask_iou > 0.8) & (mask_iou <= 0.9)] = 4.0
                mask_reward[mask_iou > 0.9] = 5.0
                total_matrix = mask_reward + bbox_iou_reward + bbox_l1_reward + point_reward
                cost_matrix = 8.0 - total_matrix
            else:
                total_matrix = bbox_iou_reward + bbox_l1_reward + point_reward
                cost_matrix = 3.0 - total_matrix

            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            total_reward = total_matrix[row_indices, col_indices].sum()
                
            max_length = max(len(pred_bboxes), len(gt_bboxes))
            if len(pred_bboxes) == len(gt_bboxes):
                count_match_score += 1.0
            
            max_accuracy_reward = (total_reward / max_length) + count_match_score 
            
            # ------------ For log ------------
            bbox_iou_score = bbox_iou_reward[row_indices, col_indices].sum() / max_length
            bbox_l1_score = bbox_l1_reward[row_indices, col_indices].sum() / max_length
            point_score = point_reward[row_indices, col_indices].sum() / max_length
            mask_score = mask_reward[row_indices, col_indices].sum() / max_length
            # ------------ For log ------------

    except Exception:
        pass
    return max_accuracy_reward, bbox_iou_score, bbox_l1_score, point_score, mask_score, count_match_score


def _vision_reasoner_non_repeat_reward(predict_str: str) -> float:
    non_repeat_reward = 1.0  # 初始满分
    try:
        sentences = predict_str.split('.')
        
        # 移除空句子
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 检查重复
        seen = set()
        repeats = 0
        
        for sentence in sentences:
            if sentence in seen:
                repeats += 1
            if repeats >=2:
                non_repeat_reward = 0
                break
            seen.add(sentence)
            
    except Exception:
        pass
    
    return non_repeat_reward


def batch_iou(boxes1, boxes2):
    # boxes1: (M,4), boxes2: (N,4)
    # 广播机制自动扩展维度
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)  # (M,1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)  # (N,1)
    
    xA = np.maximum(x11, np.transpose(x21))  # (M,N)
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    
    interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
    box1Area = (x12 - x11 + 1) * (y12 - y11 + 1)  # (M,1)
    box2Area = (x22 - x21 + 1) * (y22 - y21 + 1)  # (N,1)
    
    unionArea = box1Area + np.transpose(box2Area) - interArea
    iou = interArea / unionArea  # (M,N)
    return iou

def batch_l1_distance(boxes1, boxes2):
    # boxes1: (M,4), boxes2: (N,4)
    boxes1 = boxes1[:, np.newaxis, :]  # (M,1,4)
    boxes2 = boxes2[np.newaxis, :, :]  # (1,N,4)
    return np.mean(np.abs(boxes1 - boxes2), axis=2)  # (M,N)

def batch_points_distance(points1, points2):
    # points1: (M,2), points2: (N,2)
    points1 = points1[:, np.newaxis, :]  # (M,1,2)
    points2 = points2[np.newaxis, :, :]  # (1,N,2)
    
    # 计算欧氏距离
    dist = np.sqrt(np.sum((points1 - points2)**2, axis=2))  # (M,N)
    return dist

def batch_points_in_box(points, boxes):
    """
    检查每个点是否在对应的框内
    points: (M,2) - M个点的坐标
    boxes: (M,4) - M个框的坐标 [x1,y1,x2,y2]
    返回: (M,) 布尔数组
    """
    x_check = (points[:,0] >= boxes[:,0]) & (points[:,0] <= boxes[:,2])
    y_check = (points[:,1] >= boxes[:,1]) & (points[:,1] <= boxes[:,3])
    return x_check & y_check

def batch_mask_iou(pred_masks: np.ndarray, gt_masks: np.ndarray) -> np.ndarray:
    """Compute IoU between predicted and ground-truth masks."""
    pred_masks = pred_masks.astype(bool)
    gt_masks = gt_masks.astype(bool)
    intersection = np.logical_and(pred_masks[:, None], gt_masks[None]).sum(axis=(2, 3))
    union = np.logical_or(pred_masks[:, None], gt_masks[None]).sum(axis=(2, 3))
    return intersection / (union + 1e-6)

########### For Qwen3-VL models ###########

def _infer_grid_from_values(values: List[float]) -> Optional[int]:
    """
    Qwen3-VL 좌표 스케일 추정:
      - max <= 1.2   -> [0,1] 정규화
      - max <= 999.5 -> 0–999 그리드
      - max <= 1000.5-> 0–1000 그리드
      - 그 외        -> 픽셀 좌표(변환 불필요) -> None
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
    if vmax <= 999.5:
        return 999
    if vmax <= 1000.5:
        return 1000
    return None


def _scale_coord(val: float, grid: Optional[int], size: int) -> int:
    """
    상대좌표 -> 픽셀 인덱스(0..size-1). grid=None 이면 이미 픽셀로 간주.
    """
    if grid is None:
        vv = round(val)
    elif grid == 1:
        vv = round(val * (size - 1))
    else:
        vv = round(val * (size - 1) / float(grid))
    return max(0, min(size - 1, int(vv)))


def _order_box_xyxy(x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int, int, int]:
    left = min(x1, x2); right = max(x1, x2)
    top = min(y1, y2); bottom = max(y1, y2)
    return left, top, right, bottom


def _convert_qwen3_predictions_to_pixels(
    pred_bboxes: List[List[float]],
    pred_points: List[List[float]],
    image: Optional[Any],
    default_w: int = 840,
    default_h: int = 840,
    force_grid: Optional[int] = None,
    min_box_size: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Qwen3-VL 예측(상대좌표)을 이미지 픽셀 좌표로 변환.
    - 박스: [x1,y1,x2,y2] 정렬 및 최소 크기 보장
    - 포인트: [x,y] 스케일
    반환: (boxes_np[M,4], points_np[M,2])
    """
    if image is not None:
        try:
            W, H = image.size
        except Exception:
            W, H = default_w, default_h
    else:
        W, H = default_w, default_h

    # 그리드 추정(전 샘플 통합)
    vals: List[float] = []
    for b in pred_bboxes:
        if isinstance(b, (list, tuple)) and len(b) == 4:
            vals += [b[0], b[1], b[2], b[3]]
    for p in pred_points:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            vals += [p[0], p[1]]
    grid = force_grid if force_grid is not None else _infer_grid_from_values(vals)

    boxes_px: List[List[int]] = []
    points_px: List[List[int]] = []

    for i in range(len(pred_bboxes)):
        bx = pred_bboxes[i]
        pt = pred_points[i] if i < len(pred_points) else None

        if bx is not None and len(bx) == 4:
            x1 = _scale_coord(bx[0], grid, W)
            y1 = _scale_coord(bx[1], grid, H)
            x2 = _scale_coord(bx[2], grid, W)
            y2 = _scale_coord(bx[3], grid, H)
            x1, y1, x2, y2 = _order_box_xyxy(x1, y1, x2, y2)
            if x2 <= x1:
                x2 = min(W - 1, x1 + min_box_size)
            if y2 <= y1:
                y2 = min(H - 1, y1 + min_box_size)
            boxes_px.append([x1, y1, x2, y2])
        else:
            boxes_px.append([0, 0, 0, 0])

        if pt is not None and len(pt) == 2:
            px = _scale_coord(pt[0], grid, W)
            py = _scale_coord(pt[1], grid, H)
            points_px.append([px, py])
        else:
            points_px.append([0, 0])

    return np.array(boxes_px, dtype=np.int32), np.array(points_px, dtype=np.int32)

########### For Qwen3-VL models ###########