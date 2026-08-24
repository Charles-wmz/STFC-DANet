import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export immutable benchmark folds for official baselines.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/abide_cc400_author_npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/benchmark_splits")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[42, 3407, 2026])
    return parser.parse_args()


def split_train_val(indices: np.ndarray, labels: np.ndarray, test_size: float, seed: int):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    relative_train, relative_val = next(splitter.split(np.zeros(len(indices)), labels[indices]))
    return indices[relative_train], indices[relative_val]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = np.load(args.data_dir / "y.npy").astype(np.int64)
    metadata = pd.read_csv(args.data_dir / "metadata.csv")
    if len(metadata) != len(labels):
        raise ValueError("metadata.csv and y.npy have different sample counts.")

    subject_bytes = "\n".join(metadata["subject"].astype(str)).encode("utf-8")
    subject_sha256 = hashlib.sha256(subject_bytes).hexdigest()
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.split_seed)

    arrays = {}
    fold_summaries = []
    seen_test = np.zeros(len(labels), dtype=np.int64)
    for fold, (outer_train, test_indices) in enumerate(
        splitter.split(np.zeros_like(labels), labels),
        start=1,
    ):
        train_indices, val_indices = split_train_val(
            outer_train,
            labels,
            args.val_ratio,
            args.split_seed + fold,
        )
        arrays[f"fold_{fold}_train"] = train_indices.astype(np.int64)
        arrays[f"fold_{fold}_val"] = val_indices.astype(np.int64)
        arrays[f"fold_{fold}_test"] = test_indices.astype(np.int64)
        seen_test[test_indices] += 1
        fold_summaries.append(
            {
                "fold": fold,
                "train_size": int(len(train_indices)),
                "val_size": int(len(val_indices)),
                "test_size": int(len(test_indices)),
                "train_asd": int((labels[train_indices] == 0).sum()),
                "val_asd": int((labels[val_indices] == 0).sum()),
                "test_asd": int((labels[test_indices] == 0).sum()),
            }
        )

    if not np.all(seen_test == 1):
        raise RuntimeError("Outer test folds do not partition the dataset exactly once.")

    split_path = args.output_dir / "abide_cc400_5fold_seed42.npz"
    np.savez_compressed(split_path, **arrays)
    manifest = {
        "data_dir": str(args.data_dir),
        "num_samples": int(len(labels)),
        "num_asd": int((labels == 0).sum()),
        "num_control": int((labels == 1).sum()),
        "label_mapping": "0=ASD, 1=HC",
        "split_seed": int(args.split_seed),
        "training_seeds": [int(seed) for seed in args.training_seeds],
        "folds": int(args.folds),
        "val_ratio": float(args.val_ratio),
        "subject_order_sha256": subject_sha256,
        "split_file": split_path.name,
        "fold_summaries": fold_summaries,
    }
    with (args.output_dir / "benchmark_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
