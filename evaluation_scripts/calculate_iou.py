import os
import json
import glob
import re
import numpy as np
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="folder path of output files")
    return parser.parse_args()

def calculate_metrics(output_dir):
    # get all output files
    output_files = sorted(glob.glob(os.path.join(output_dir, "output_*.json")))
    
    if not output_files:
        print(f"cannot find output files in {output_dir}")
        return
    
    # for accumulating all data
    all_ious = []
    total_intersection = 0.0  # for cIoU
    total_union = 0.0         # for cIoU
    n_items = 0
    
    # response length (token count) accumulation
    total_resp_len = 0.0
    resp_len_count = 0
    
    # VReasonSeg per-type aggregation (activated only if 'reasoning_type' exists in outputs)
    # type_stats[rt] = {"sum_iou":..., "sum_inter":..., "sum_union":..., "count":...}
    type_stats = {}
    
    # self-correction metrics per turn:
    # turn=1 uses legacy keys sc_intersection/sc_union (also supported)
    sc_turn_iou_sum = {}          # turn -> sum(sc_iou)
    sc_turn_base_iou_sum = {}     # turn -> sum(base_iou) for items that have sc turn
    sc_turn_total_intersection = {}
    sc_turn_total_union = {}
    sc_turn_count = {}            # number of items that have this turn
    sc_turn_pos_delta_sum = {}    # turn -> sum(max(delta,0))
    sc_turn_neg_delta_sum = {}    # turn -> sum(min(delta,0))

    # read and process all files
    for file_path in output_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
        # process all items in each file
        for item in results:
            intersection = item['intersection']
            union = item['union']
            
            base_iou = intersection / union if union > 0 else 1.0
            
            # ------------------------------------------------------------
            # response length (token count) aggregation (if available)
            # ------------------------------------------------------------
            rl = item.get("response_length", None)
            if isinstance(rl, (int, float)) and np.isfinite(rl):
                total_resp_len += float(rl)
                resp_len_count += 1
            
            # ------------------------------------------------------------
            # VReasonSeg: per reasoning_type aggregation (only if present)
            # ------------------------------------------------------------
            rt = item.get("reasoning_type", None)
            if rt is not None:
                st = type_stats.get(rt, None)
                if st is None:
                    st = {"sum_iou": 0.0, "sum_inter": 0.0, "sum_union": 0.0, "count": 0}
                    type_stats[rt] = st
                st["sum_iou"] += float(base_iou)
                st["sum_inter"] += float(intersection)
                st["sum_union"] += float(union)
                st["count"] += 1
            
            # calculate IoU of each item
            total_intersection += intersection
            total_union += union
            n_items += 1
            
            iou = base_iou
            all_ious.append({
                'image_id': item['image_id'],
                'iou': iou
            })
            
            # ------------------------------------------------------------
            # Self-correction turns detection:
            # - legacy: sc_intersection/sc_union => turn 1
            # - multi-turn: sc_intersection_{t}, sc_union_{t}
            # ------------------------------------------------------------
            sc_turns = set()
            if ('sc_intersection' in item) and ('sc_union' in item):
                sc_turns.add(1)
            for k in item.keys():
                m = re.match(r'^sc_intersection_(\d+)$', k)
                if m:
                    t = int(m.group(1))
                    if f"sc_union_{t}" in item:
                        sc_turns.add(t)

            for t in sorted(sc_turns):
                if t == 1 and ('sc_intersection' in item) and ('sc_union' in item):
                    sc_intersection = item.get('sc_intersection', None)
                    sc_union = item.get('sc_union', None)
                else:
                    sc_intersection = item.get(f"sc_intersection_{t}", None)
                    sc_union = item.get(f"sc_union_{t}", None)
                if sc_intersection is None or sc_union is None:
                    continue

                sc_iou = sc_intersection / sc_union if sc_union > 0 else 1.0
                delta = sc_iou - base_iou
                pos = max(delta, 0.0)
                neg = min(delta, 0.0)

                sc_turn_iou_sum[t] = sc_turn_iou_sum.get(t, 0.0) + float(sc_iou)
                sc_turn_base_iou_sum[t] = sc_turn_base_iou_sum.get(t, 0.0) + float(base_iou)
                sc_turn_total_intersection[t] = sc_turn_total_intersection.get(t, 0.0) + float(sc_intersection)
                sc_turn_total_union[t] = sc_turn_total_union.get(t, 0.0) + float(sc_union)
                sc_turn_count[t] = sc_turn_count.get(t, 0) + 1
                sc_turn_pos_delta_sum[t] = sc_turn_pos_delta_sum.get(t, 0.0) + float(pos)
                sc_turn_neg_delta_sum[t] = sc_turn_neg_delta_sum.get(t, 0.0) + float(neg)
           
    
    # calculate gIoU
    gIoU = np.mean([item['iou'] for item in all_ious])
    # calculate cIoU
    # print(f"cumulative intersection: {total_intersection}")
    # print(f"cumulatvie union: {total_union}")
    cIoU = total_intersection / total_union
    
    # print the results
    print(f"Processed {n_items} samples.")
    print(f"gIoU (average of per image IoU): {gIoU:.4f}")
    print(f"cIoU (sum of intersections / sum of unions): {cIoU:.4f}")
    if resp_len_count > 0:
        avg_rl = total_resp_len / float(resp_len_count)
        print(f"Avg response length (tokens): {avg_rl:.2f}")
    else:
        print("Avg response length (tokens): N/A (no 'response_length' in outputs)")
    
    # ------------------------------------------------------------
    # ReasonSeg-X: print per reasoning_type metrics (if available)
    # ------------------------------------------------------------
    if len(type_stats) > 0:
        print("\n[ReasonSeg-X] Per reasoning_type metrics:")
        for rt in sorted(type_stats.keys()):
            st = type_stats[rt]
            cnt = int(st.get("count", 0))
            if cnt <= 0:
                continue
            g = float(st["sum_iou"]) / float(cnt)
            cu = float(st["sum_union"])
            c = (float(st["sum_inter"]) / cu) if cu > 0 else 0.0
            print(f"[reasoning_type={rt}] samples={cnt} gIoU={g:.4f} cIoU={c:.4f}")
    
    # calculate and print self-correction metrics per turn (if available)
    if len(sc_turn_count) > 0:
        for t in sorted(sc_turn_count.keys()):
            cnt = sc_turn_count[t]
            if cnt <= 0:
                continue
            sc_gIoU = sc_turn_iou_sum[t] / cnt
            sc_cIoU = (sc_turn_total_intersection[t] / sc_turn_total_union[t]) if sc_turn_total_union[t] > 0 else 0.0
            base_gIoU_on_sc = sc_turn_base_iou_sum[t] / cnt
            mean_pos = sc_turn_pos_delta_sum[t] / cnt
            mean_neg = sc_turn_neg_delta_sum[t] / cnt
            mean_delta = mean_pos + mean_neg

            print(f"[Self-correction@turn={t}] Processed {cnt} samples.")
            print(f"[Self-correction@turn={t}] gIoU (average of per image IoU): {sc_gIoU:.4f}")
            print(f"[Self-correction@turn={t}] cIoU (sum of intersections / sum of unions): {sc_cIoU:.4f}")
            print(f"[Self-correction@turn={t}] mean(+ΔIoU): {mean_pos:.6f}")
            print(f"[Self-correction@turn={t}] mean(-ΔIoU): {mean_neg:.6f}")
            print(f"[Self-correction@turn={t}] mean(ΔIoU): {mean_delta:.6f}  (should equal sc_gIoU - base_gIoU_on_sc = {(sc_gIoU - base_gIoU_on_sc):.6f})")
    

if __name__ == "__main__":
    args = parse_args()
    calculate_metrics(args.output_dir)
