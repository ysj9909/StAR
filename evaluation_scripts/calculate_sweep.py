import os
import json
import glob
import time
import numpy as np
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="folder path of sweep/output files")
    parser.add_argument("--top_k", type=int, default=30, help="print top-k configs by gIoU")
    return parser.parse_args()


def load_json_safely(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return None


def calculate_sweep_metrics(output_dir: str, top_k: int = 30):
    sweep_files = sorted(glob.glob(os.path.join(output_dir, "sweep_stats_*.json")))
    if not sweep_files:
        print(f"cannot find sweep_stats_*.json in {output_dir}")
        return

    # cfg_id -> aggregated stats
    agg_stats = {}
    cfg_params = {}

    meta_any = None
    n_shards_loaded = 0

    for fp in sweep_files:
        data = load_json_safely(fp)
        if data is None:
            continue
        n_shards_loaded += 1
        if meta_any is None:
            meta_any = {
                "reasoning_model_path": data.get("reasoning_model_path", ""),
                "test_data_path": data.get("test_data_path", ""),
                "num_samples": data.get("num_samples", None),
                "sampling_temperature": data.get("sampling_temperature", None),
                "sampling_top_p": data.get("sampling_top_p", None),
            }

        configs = data.get("configs", [])
        stats = data.get("stats", {})

        # keep params for printing
        for cfg in configs:
            cfg_id = cfg["id"]
            cfg_params[cfg_id] = cfg

        for cfg_id, st in stats.items():
            if cfg_id not in agg_stats:
                agg_stats[cfg_id] = {
                    "sum_intersection": 0.0,
                    "sum_union": 0.0,
                    "count": 0,
                    "sum_iou": 0.0,
                }
            agg_stats[cfg_id]["sum_intersection"] += float(st.get("sum_intersection", 0.0))
            agg_stats[cfg_id]["sum_union"] += float(st.get("sum_union", 0.0))
            agg_stats[cfg_id]["count"] += int(st.get("count", 0))
            # for gIoU = mean(per-image IoU)
            # we store sum_iou directly to avoid huge lists
            agg_stats[cfg_id]["sum_iou"] += float(st.get("sum_iou", 0.0))

    # compute dataset-level metrics per cfg
    rows = []
    for cfg_id, st in agg_stats.items():
        count = st["count"]
        if count <= 0:
            continue
        gIoU = st["sum_iou"] / count
        cIoU = (st["sum_intersection"] / st["sum_union"]) if st["sum_union"] > 0 else 0.0
        cfg = cfg_params.get(cfg_id, {})
        rows.append((cfg_id, gIoU, cIoU, count, cfg))

    rows.sort(key=lambda x: (x[1], x[2]), reverse=True)  # sort by gIoU then cIoU

    # write full results to a file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_txt = os.path.join(output_dir, f"{timestamp}_sweep_results.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("========== Majority-voting hyper-parameter sweep (dataset-level) ==========\n")
        f.write(f"loaded_shards={n_shards_loaded} (files={len(sweep_files)})\n")
        if meta_any is not None:
            f.write(f"reasoning_model_path={meta_any['reasoning_model_path']}\n")
            f.write(f"test_data_path={meta_any['test_data_path']}\n")
            f.write(f"num_samples={meta_any['num_samples']}, sampling_temperature={meta_any['sampling_temperature']}, sampling_top_p={meta_any['sampling_top_p']}\n")
        f.write(f"num_configs={len(rows)}\n\n")

        for cfg_id, gIoU, cIoU, count, cfg in rows:
            f.write(
                f"[SWEEP][{cfg_id}] "
                f"no_obj_thr={cfg.get('no_object_vote_threshold')}, "
                f"mask_iou_thr={cfg.get('mask_iou_cluster_threshold')}, "
                f"cluster_vote_thr={cfg.get('cluster_vote_threshold')}, "
                f"pixel_thr={cfg.get('pixel_majority_threshold')}, "
                f"agg_mode={cfg.get('cluster_agg_mode')} | "
                f"n_items={count}, gIoU={gIoU:.4f}, cIoU={cIoU:.4f}\n"
            )
        f.write("==========================================================================\n")

    # print top-k to terminal (short)
    print("========== Sweep (dataset-level) TOP results ==========")
    print(f"[INFO] Full results written to: {out_txt}")
    for rank, (cfg_id, gIoU, cIoU, count, cfg) in enumerate(rows[: max(1, top_k)], start=1):
        print(
            f"[TOP {rank:02d}] gIoU={gIoU:.4f}, cIoU={cIoU:.4f}, n={count} | "
            f"{cfg_id}"
        )
    print("======================================================")


if __name__ == "__main__":
    args = parse_args()
    calculate_sweep_metrics(args.output_dir, top_k=args.top_k)