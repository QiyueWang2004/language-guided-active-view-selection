"""
Convert the UMD+GT RGB-D affordance dataset into the active-view-selection
annotation format used by active_view_selection.py.

For each tool (a turntable sequence of ~200-400 frames) we sample several
5-view sets at 20% increments of the sequence. Each set yields one sample per
affordance present on the tool. GT views are derived from the affordance masks:
a view is GT when its affordance area is >= gt_ratio * the largest area among
the 5 views. Samples whose best view is not clearly distinguishable (too many GT
views) or whose affordance is barely visible are dropped.

Splits follow the official umd_gt_category_split.txt (1 = train, 2 = test);
val_frac of the train instances in each category are held out as validation.
"""

import argparse
import collections
import io
import json
import os
import random
import zipfile
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


AFFORDANCES = {1: "grasp", 2: "cut", 3: "scoop", 4: "contain", 5: "pound", 6: "wgrasp"}

AFFORDANCE_PHRASES = {
    "grasp": "the handle is most clearly visible for grasping",
    "cut": "the cutting blade is most clearly visible",
    "scoop": "the scooping part is most clearly visible",
    "contain": "the opening of the container is most clearly visible",
    "pound": "the striking head is most clearly visible",
    "wgrasp": "the outer body to wrap a hand around is most clearly visible",
}


def tool_category(tool: str) -> str:
    return tool.rsplit("_", 1)[0]


def list_tool_frames(zf: zipfile.ZipFile) -> Dict[str, List[Tuple[str, str]]]:
    """tool -> sorted list of (frame_stem, rgb_member_name)."""
    frames: Dict[str, Dict[str, str]] = collections.defaultdict(dict)
    for name in zf.namelist():
        for suffix in ("_rgb.jpg", "_rgb.png"):
            if name.endswith(suffix):
                tool = name.split("/")[0]
                stem = name.split("/")[-1][: -len(suffix)]
                frames[tool].setdefault(stem, name)
    return {
        tool: sorted(stems.items(), key=lambda kv: int(kv[0].rsplit("_", 1)[-1]))
        for tool, stems in frames.items()
    }


def read_official_split(zf: zipfile.ZipFile) -> Dict[str, int]:
    name = next(n for n in zf.namelist() if n.endswith("umd_gt_category_split.txt"))
    split = {}
    for line in zf.read(name).decode(errors="ignore").splitlines():
        parts = line.split()
        if len(parts) == 2:
            split[parts[1]] = int(parts[0])
    return split


def assign_splits(tools: List[str], official: Dict[str, int], val_frac: float, seed: int) -> Dict[str, str]:
    rng = random.Random(seed)
    by_cat: Dict[str, List[str]] = collections.defaultdict(list)
    out = {}
    for tool in sorted(tools):
        if official.get(tool, 1) == 2:
            out[tool] = "test"
        else:
            by_cat[tool_category(tool)].append(tool)
    for cat, cat_tools in sorted(by_cat.items()):
        rng.shuffle(cat_tools)
        n_val = int(round(val_frac * len(cat_tools))) if len(cat_tools) >= 2 else 0
        for i, tool in enumerate(cat_tools):
            out[tool] = "val" if i < n_val else "train"
    return out


def view_set_indices(num_frames: int, num_views: int, set_idx: int, num_sets: int) -> List[int]:
    offset = int(set_idx * num_frames / (num_views * num_sets))
    return [(offset + int(i * num_frames / num_views)) % num_frames for i in range(num_views)]


def affordance_areas(mask_zip: zipfile.ZipFile, member: str) -> Dict[str, float]:
    arr = np.array(Image.open(io.BytesIO(mask_zip.read(member))))
    labels, counts = np.unique(arr, return_counts=True)
    return {
        AFFORDANCES[int(k)]: float(v) / arr.size
        for k, v in zip(labels, counts)
        if int(k) in AFFORDANCES
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--umd_zip", default=r"E:\RGB-D\UMD+GT.zip")
    parser.add_argument("--mask_zip", default=r"E:\RGB-D\UMD_GT_MASK.zip")
    parser.add_argument("--out_dir", default=r"E:\UMD-subset")
    parser.add_argument("--out_json", default=r"E:\UMD-subset\annotations_umd.json")
    parser.add_argument("--num_views", type=int, default=5)
    parser.add_argument("--train_sets", type=int, default=8,
                        help="View sets sampled per train tool (shifted start frame).")
    parser.add_argument("--eval_sets", type=int, default=4,
                        help="View sets sampled per val/test tool.")
    parser.add_argument("--gt_ratio", type=float, default=0.8)
    parser.add_argument("--min_area_frac", type=float, default=0.002,
                        help="Drop samples whose best view shows less than this image fraction.")
    parser.add_argument("--max_gt_views", type=int, default=3,
                        help="Drop samples with more GT views than this (no clear best view).")
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    zf = zipfile.ZipFile(args.umd_zip)
    mask_zip = zipfile.ZipFile(args.mask_zip)
    mask_members = set(mask_zip.namelist())

    tool_frames = list_tool_frames(zf)
    splits = assign_splits(list(tool_frames), read_official_split(zf), args.val_frac, args.seed)

    samples = []
    dropped = collections.Counter()
    for tool, frames in sorted(tool_frames.items()):
        category = tool_category(tool)
        split = splits[tool]
        num_sets = args.train_sets if split == "train" else args.eval_sets
        for set_idx in range(num_sets):
            idxs = view_set_indices(len(frames), args.num_views, set_idx, num_sets)
            view_paths, view_areas = [], []
            for i in idxs:
                stem, member = frames[i]
                mask_member = f"{tool}/{stem}_labelid.png"
                if mask_member not in mask_members:
                    break
                dst = os.path.join(args.out_dir, tool, os.path.basename(member))
                if not os.path.exists(dst):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    with open(dst, "wb") as f:
                        f.write(zf.read(member))
                view_paths.append(dst.replace("\\", "/"))
                view_areas.append(affordance_areas(mask_zip, mask_member))
            if len(view_paths) != args.num_views:
                dropped["missing_mask"] += 1
                continue

            tool_affs = sorted({a for areas in view_areas for a in areas})
            for aff in tool_affs:
                areas = [areas.get(aff, 0.0) for areas in view_areas]
                best = max(areas)
                if best < args.min_area_frac:
                    dropped[f"{aff}:too_small"] += 1
                    continue
                gt_idx = [i for i, a in enumerate(areas) if a >= args.gt_ratio * best]
                if len(gt_idx) > args.max_gt_views:
                    dropped[f"{aff}:no_clear_best"] += 1
                    continue

                order = list(range(args.num_views))
                rng.shuffle(order)
                samples.append({
                    "id": f"{category}::{tool}::{aff}::set{set_idx}",
                    "category": category,
                    "affordance": aff,
                    "tool": tool,
                    "split": split,
                    "instruction": f"find the view of the {category} where {AFFORDANCE_PHRASES[aff]}",
                    "views": [view_paths[i] for i in order],
                    "gt_views": [view_paths[i] for i in gt_idx],
                    "affordance_area": {view_paths[i]: round(areas[i], 5) for i in order},
                })

    rng.shuffle(samples)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(samples)} samples to {args.out_json}")
    for split in ("train", "val", "test"):
        sub = [s for s in samples if s["split"] == split]
        if not sub:
            continue
        tools = {s["tool"] for s in sub}
        chance = np.mean([len(s["gt_views"]) / args.num_views for s in sub])
        mean_gt = np.mean([len(s["gt_views"]) for s in sub])
        print(f"  {split:5s}: samples={len(sub):5d} tools={len(tools):3d} "
              f"mean_gt_views={mean_gt:.2f} random_chance={chance:.3f}")
    print("Samples per affordance (train/val/test):")
    for aff in AFFORDANCES.values():
        counts = [sum(1 for s in samples if s["affordance"] == aff and s["split"] == sp)
                  for sp in ("train", "val", "test")]
        print(f"  {aff:8s} {counts}")
    print("Samples per category (train/val/test):")
    for cat in sorted({s["category"] for s in samples}):
        counts = [sum(1 for s in samples if s["category"] == cat and s["split"] == sp)
                  for sp in ("train", "val", "test")]
        print(f"  {cat:12s} {counts}")
    print("Dropped:", dict(dropped))


if __name__ == "__main__":
    main()
