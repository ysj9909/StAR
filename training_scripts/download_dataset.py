"""Download StAR datasets from HuggingFace Hub to local disk.

Training datasets are saved as DatasetDict (with 'train' split).
Evaluation datasets are saved as Dataset (single, no split).
This matches the format expected by the training and evaluation code.
"""
import argparse
import os
from datasets import load_dataset, DatasetDict


# keep_dict=True  -> save as DatasetDict (for training code: load_from_disk(path)['train'])
# keep_dict=False -> unwrap to Dataset   (for eval code:     load_from_disk(path))
DATASETS = {
    # Training data (DatasetDict with 'train' split)
    "ReasonSegX_train": {
        "repo": "sj9909/ReasonSegX_train",
        "local_dir": "data/ReasonSegX_train",
        "keep_dict": True,
    },
    "refcocolvis": {
        "repo": "sj9909/visionreasoner_multi_object_refcocolvis_masks_840",
        "local_dir": "data/visionreasoner_multi_object_refcocolvis_masks_840",
        "keep_dict": True,
    },
    # Evaluation data (single Dataset, no split)
    "ReasonSegX_val": {
        "repo": "sj9909/ReasonSegX_val",
        "local_dir": "data/ReasonSegX_val",
        "keep_dict": False,
    },
    "ReasonSegX_test": {
        "repo": "sj9909/ReasonSegX_test",
        "local_dir": "data/ReasonSegX_test",
        "keep_dict": False,
    },
    "ReasonSeg_refine": {
        "repo": "sj9909/ReasonSeg_refine",
        "local_dir": "data/ReasonSeg_refine",
        "keep_dict": False,
    },
    # MUSE
    "MUSE_val": {
        "repo": "sj9909/MUSE_val",
        "local_dir": "data/MUSE/val",
        "keep_dict": False,
    },
    "MUSE_test_few": {
        "repo": "sj9909/MUSE_test_few",
        "local_dir": "data/MUSE/test_few",
        "keep_dict": False,
    },
    "MUSE_test_many": {
        "repo": "sj9909/MUSE_test_many",
        "local_dir": "data/MUSE/test_many",
        "keep_dict": False,
    },
    # MMR
    "MMR_val": {
        "repo": "sj9909/MMR_val",
        "local_dir": "data/MMR/val",
        "keep_dict": False,
    },
    "MMR_test_mixed": {
        "repo": "sj9909/MMR_test_mixed",
        "local_dir": "data/MMR/test_mixed",
        "keep_dict": False,
    },
    "MMR_test_obj": {
        "repo": "sj9909/MMR_test_obj",
        "local_dir": "data/MMR/test_obj",
        "keep_dict": False,
    },
    "MMR_test_part": {
        "repo": "sj9909/MMR_test_part",
        "local_dir": "data/MMR/test_part",
        "keep_dict": False,
    },
}


def download_dataset(name, info):
    print(f"\n{'='*60}")
    print(f"Downloading: {info['repo']} -> {info['local_dir']}")

    if os.path.exists(info["local_dir"]):
        print(f"  Already exists, skipping. Delete the folder to re-download.")
        return

    os.makedirs(os.path.dirname(info["local_dir"]) or ".", exist_ok=True)
    dataset = load_dataset(info["repo"])

    if not info["keep_dict"] and isinstance(dataset, DatasetDict) and len(dataset) == 1:
        # Unwrap single-split DatasetDict to plain Dataset
        dataset = list(dataset.values())[0]

    dataset.save_to_disk(info["local_dir"])
    print(f"  Saved as {type(dataset).__name__} to: {info['local_dir']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="*", default=None,
                        help="Specific dataset names to download. If omitted, downloads all.")
    args = parser.parse_args()

    targets = args.names if args.names else list(DATASETS.keys())

    for name in targets:
        if name not in DATASETS:
            print(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
            continue
        download_dataset(name, DATASETS[name])

    print(f"\n{'='*60}")
    print("All done!")


if __name__ == "__main__":
    main()
