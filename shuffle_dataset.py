"""
Shuffle annotation dataset order.

For each sample:
1) randomly permute the views list
2) remap gt_views to the same shuffled paths
3) optionally shuffle sample order in the JSON

This removes positional bias (e.g. GT often at view_01 / index 0).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def shuffle_sample(sample: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    out = dict(sample)
    views = list(sample["views"])
    gt_set = set(sample.get("gt_views", []))

    order = list(range(len(views)))
    rng.shuffle(order)
    shuffled_views = [views[i] for i in order]
    shuffled_gt = [p for p in shuffled_views if p in gt_set]

    out["views"] = shuffled_views
    out["gt_views"] = shuffled_gt
    out["shuffle_order"] = order
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Shuffle views (and sample order) in annotations JSON")
    parser.add_argument(
        "--labels_json",
        type=str,
        default="annotations/annotations_all_5sets.json",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default="annotations/annotations_all_5sets_shuffled.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_shuffle_samples",
        action="store_true",
        help="Only shuffle views within each sample; keep sample list order",
    )
    args = parser.parse_args()

    in_path = Path(args.labels_json)
    out_path = Path(args.out_json)
    rng = random.Random(args.seed)

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {in_path}")

    shuffled = [shuffle_sample(sample, rng) for sample in data]
    if not args.no_shuffle_samples:
        rng.shuffle(shuffled)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(shuffled, f, indent=2, ensure_ascii=False)

    # Quick sanity checks
    n = len(shuffled)
    empty_gt = sum(1 for s in shuffled if not s.get("gt_views"))
    gt_first = 0
    for s in shuffled:
        if s.get("gt_views") and s["views"] and s["views"][0] in set(s["gt_views"]):
            gt_first += 1

    print("=" * 80)
    print(f"Input:  {in_path}")
    print(f"Output: {out_path}")
    print(f"Seed:   {args.seed}")
    print(f"Samples: {n}")
    print(f"Empty gt_views: {empty_gt}")
    print(f"GT at index 0 after shuffle: {gt_first}/{n} ({gt_first / max(1, n):.2%})")
    print("=" * 80)


if __name__ == "__main__":
    main()
