"""
Merge per-category CO3D annotation JSON files into one dataset file.

Example:
  python merge_annotations.py \
    --inputs E:/CO3D-subset-cup/annotations.json E:/CO3D-subset-book/annotations.json \
    --out annotations/annotations_all_5sets.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


REQUIRED = ["id", "category", "instruction", "views", "gt_views"]


def load_json(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def check_sample(sample: Dict[str, Any], source: Path) -> None:
    for key in REQUIRED:
        if key not in sample:
            raise ValueError(f"Missing key '{key}' in {source}: {sample.get('id')}")
    if not isinstance(sample["views"], list) or not sample["views"]:
        raise ValueError(f"'views' must be a non-empty list in {source}: {sample['id']}")
    if not isinstance(sample["gt_views"], list):
        raise ValueError(f"'gt_views' must be a list in {source}: {sample['id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge CO3D annotation JSON files")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more annotations.json paths",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=r"E:\CO3D-combined\annotations_all_5sets.json",
    )
    parser.add_argument(
        "--prefix_id",
        action="store_true",
        default=True,
        help="Prefix ids with category:: (default on)",
    )
    parser.add_argument(
        "--no_prefix_id",
        action="store_true",
        help="Do not prefix ids with category::",
    )
    parser.add_argument(
        "--skip_missing",
        action="store_true",
        default=True,
        help="Skip samples whose image paths are missing (default on)",
    )
    parser.add_argument(
        "--keep_missing",
        action="store_true",
        help="Keep samples even if some image paths are missing",
    )
    args = parser.parse_args()

    prefix_id = not args.no_prefix_id
    skip_missing = not args.keep_missing

    merged: List[Dict[str, Any]] = []
    counts: Counter = Counter()
    seen_ids = set()
    total_in = skipped_missing = skipped_duplicate = 0

    for ann_file in map(Path, args.inputs):
        if not ann_file.exists():
            print(f"[WARN] Missing annotation file: {ann_file}")
            continue
        data = load_json(ann_file)
        print(f"[INFO] Loaded {len(data)} from {ann_file}")
        total_in += len(data)

        for sample in data:
            check_sample(sample, ann_file)
            sample = dict(sample)
            category = sample["category"]
            if prefix_id and "::" not in sample["id"]:
                sample["id"] = f"{category}::{sample['id']}"

            if sample["id"] in seen_ids:
                print(f"[WARN] Duplicate id skipped: {sample['id']}")
                skipped_duplicate += 1
                continue

            missing = [p for p in sample["views"] + sample["gt_views"] if not Path(p).exists()]
            if missing and skip_missing:
                print(f"[WARN] Skipping {sample['id']} ({len(missing)} missing files)")
                skipped_missing += 1
                continue

            seen_ids.add(sample["id"])
            counts[category] += 1
            merged.append(sample)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print("\n===== Merge Summary =====")
    print(f"Input samples:  {total_in}")
    print(f"Merged samples: {len(merged)}")
    print(f"Skipped dups:   {skipped_duplicate}")
    print(f"Skipped missing:{skipped_missing}")
    for cat, n in sorted(counts.items()):
        print(f"  {cat}: {n}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
