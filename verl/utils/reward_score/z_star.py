import json
import os
import re
from typing import Any, List, Optional

import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scipy.optimize import linear_sum_assignment


SAM_DEVICE = "cuda:0"
SAM_DEVICE_TYPE = "cuda" if SAM_DEVICE.startswith("cuda") else "cpu"
SAM_CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "sam2", "checkpoints", "sam2.1_hiera_large.pt")
SAM_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
_SAM_PREDICTOR: Optional[SAM2ImagePredictor] = None


def _get_sam_predictor() -> SAM2ImagePredictor:
    global _SAM_PREDICTOR
    if _SAM_PREDICTOR is None:
        _SAM_PREDICTOR = SAM2ImagePredictor(build_sam2(SAM_MODEL_CFG, SAM_CHECKPOINT))
        _SAM_PREDICTOR.model.to(SAM_DEVICE)
        _SAM_PREDICTOR.model.eval()
        for param in _SAM_PREDICTOR.model.parameters():
            param.requires_grad_(False)
    return _SAM_PREDICTOR


def vision_reasoner_format_reward(predict_str: str) -> float:
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
                        cur_reward += 1.0

                if "point_2d" in item:
                    point_2d = item["point_2d"]
                    if isinstance(point_2d, list) and len(point_2d) == 2:
                        cur_reward += 1.0

                segmentation_format_reward += cur_reward / data_cnt
        except Exception:
            pass
        return segmentation_format_reward

    segmentation_format_reward = segmentation_format(predict_str)

    return thinking_format_reward + segmentation_format_reward


def z_star_accuracy_reward(
    predict_str: str,
    ground_truth: str,
    ground_truth_masks: Optional[List[Any]] = None,
    image: Optional[Any] = None,
) -> float:
    max_accuracy_reward = 0.0
    MAX_OBJECTS = 120  # 设置上限
    predictor = _get_sam_predictor()
    
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

            pred_bboxes = np.array(pred_bboxes)
            pred_points = np.array(pred_points)
            gt_bboxes = np.array(gt_bboxes)
            gt_points = np.array(gt_points)

            iou_matrix = batch_iou(pred_bboxes, gt_bboxes)
            l1_matrix = batch_l1_distance(pred_bboxes, gt_bboxes)
            points_dist_matrix = batch_points_distance(pred_points, gt_points)
            points_in_box = batch_points_in_box(pred_points, pred_bboxes)
            iou_reward = (iou_matrix > 0.5).astype(float)
            bbox_l1_reward = (l1_matrix < 10).astype(float)
            point_reward = ((points_dist_matrix < 30) & points_in_box[:, np.newaxis]).astype(float)

            mask_reward = np.zeros_like(iou_reward)
            if ground_truth_masks is not None and image is not None and len(pred_bboxes) > 0:
                pil_img = image.convert("RGB")
                pil_img = np.array(pil_img, dtype=np.uint8)
                
                print("-*-"*50)
                predictor = _get_sam_predictor()
                print("*-*" * 100)
                predictor.set_image(pil_img)
                print("*"*100)
                pred_masks = []
                with torch.inference_mode(), torch.autocast(SAM_DEVICE_TYPE, dtype=torch.bfloat16):
                    for bbox, point in zip(pred_bboxes, pred_points):
                        try:
                            print("-"*100)
                            masks, scores, _ = predictor.predict(
                                point_coords=[point.tolist()],
                                point_labels=[1],
                                box=bbox.tolist(),
                            )
                            print("mask is predicted by SAM")
                            mask = masks[np.argmax(scores)].astype(bool)
                        except Exception:
                            mask = np.zeros((pil_img.height, pil_img.width), dtype=bool)
                        pred_masks.append(mask)
                pred_masks = np.array(pred_masks)
                gt_masks = np.array(ground_truth_masks).astype(bool)
                mask_iou = batch_mask_iou(pred_masks, gt_masks)

                mask_reward = np.zeros_like(mask_iou)
                mask_reward[(mask_iou > 0.5) & (mask_iou <= 0.7)] = 2.0
                mask_reward[(mask_iou > 0.7) & (mask_iou <= 0.8)] = 3.0
                mask_reward[mask_iou > 0.8] = 4.0
                total_matrix = mask_reward + iou_reward + bbox_l1_reward + point_reward
                cost_matrix = 7.0 - total_matrix
            else:
                total_matrix = iou_reward + bbox_l1_reward + point_reward
                cost_matrix = 3.0 - total_matrix

            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            total_reward = total_matrix[row_indices, col_indices].sum()

            max_length = max(len(pred_bboxes), len(gt_bboxes))
            max_accuracy_reward = total_reward / max_length

    except Exception:
        pass
    return max_accuracy_reward


def vision_reasoner_non_repeat_reward(predict_str: str) -> float:
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


def z_star_compute_score(
    predict_str: str,
    ground_truth: str,
    ground_truth_masks: Optional[List[Any]] = None,
    image: Optional[Any] = None,
) -> float:
    format_reward = vision_reasoner_format_reward(predict_str)
    accuracy_reward = z_star_accuracy_reward(predict_str, ground_truth, ground_truth_masks, image)
    non_repeat_reward = vision_reasoner_non_repeat_reward(predict_str)

    reward = format_reward + accuracy_reward + non_repeat_reward
    return reward


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
