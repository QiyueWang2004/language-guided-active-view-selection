"""
Build (or expand) a CO3D multi-view subset for active view selection.

Samples 5 views per object from raw CO3D category folders and writes:
  <out_root>/<category>/<object_id>/view_XX.jpg
  <out_root>/annotations.json

Examples:
  python build_subset.py --category cup --objects 10
  python build_subset.py --category teddybear --object_ids teddybear_001,teddybear_002
  python build_subset.py --expand --category cup --objects 20
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_VIEW_RATIOS = [0.10, 0.30, 0.50, 0.70, 0.90]

INSTRUCTION_TEMPLATES = {
    "book": "find the view where the book is most front-facing",
    "cup": "find the view where the printed design is most visible",
    "handbag": "find the view where the handbag is most front-facing",
    "remote": "find the view where the remote is least tilted",
    "teddybear": "find the view where the face is most visible",
}


def list_images(images_dir: Path) -> List[Path]:
    return sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def uniform_sample(
    images: Sequence[Path],
    num_samples: int,
    ratios: Sequence[float] = DEFAULT_VIEW_RATIOS,
) -> List[Path]:
    n = len(images)
    if n < num_samples:
        raise ValueError(f"Not enough images to sample: {n} < {num_samples}")

    indices: List[int] = []
    seen: Set[int] = set()
    for ratio in list(ratios)[:num_samples]:
        idx = round((n - 1) * ratio)
        if idx not in seen:
            indices.append(idx)
            seen.add(idx)

    if len(indices) < num_samples:
        for i in range(n):
            if i not in seen:
                indices.append(i)
                seen.add(i)
            if len(indices) == num_samples:
                break

    indices = sorted(indices[:num_samples])
    return [images[i] for i in indices]


def copy_views(sampled: Sequence[Path], out_dir: Path) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, src in enumerate(sampled, start=1):
        dst = out_dir / f"view_{i:02d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        paths.append(str(dst).replace("\\", "/"))
    return paths


def load_annotations(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def save_annotations(path: Path, data: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def default_object_ids(category: str, count: int) -> List[str]:
    return [f"{category}_{i:03d}" for i in range(1, count + 1)]


def next_object_ids(category: str, existing_ids: Set[str], count: int) -> List[str]:
    used = []
    prefix = f"{category}_"
    for obj_id in existing_ids:
        if obj_id.startswith(prefix):
            suffix = obj_id[len(prefix) :]
            if suffix.isdigit():
                used.append(int(suffix))
    start = max(used, default=0) + 1
    return [f"{category}_{i:03d}" for i in range(start, start + count)]


def candidate_sources(
    raw_root: Path,
    category: str,
    views_per_object: int,
    exclude: Set[str],
) -> List[Tuple[str, Path, int]]:
    category_dir = raw_root / category
    if not category_dir.exists():
        raise FileNotFoundError(f"Missing raw category folder: {category_dir}")

    candidates = []
    for obj_dir in sorted(category_dir.iterdir()):
        if not obj_dir.is_dir() or obj_dir.name in exclude:
            continue
        images_dir = obj_dir / "images"
        if not images_dir.exists():
            continue
        images = list_images(images_dir)
        if len(images) < views_per_object:
            continue
        candidates.append((obj_dir.name, obj_dir, len(images)))
    return candidates


def process_object(
    category: str,
    obj_id: str,
    source_dir: Path,
    out_root: Path,
    instruction: str,
    views_per_object: int,
) -> Optional[Dict]:
    images_dir = source_dir / "images"
    if not images_dir.exists():
        print(f"[WARN] Missing images: {images_dir}")
        return None
    images = list_images(images_dir)
    if len(images) < views_per_object:
        print(f"[WARN] Not enough images in {images_dir}: {len(images)}")
        return None

    sampled = uniform_sample(images, views_per_object)
    saved = copy_views(sampled, out_root / category / obj_id)
    print(f"[INFO] {category}/{obj_id} <- {source_dir.name} ({len(images)} frames)")
    return {
        "id": obj_id,
        "category": category,
        "instruction": instruction,
        "views": saved,
        "gt_views": [],
    }


def build_fresh(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    ann_path = out_root / "annotations.json"
    instruction = args.instruction or INSTRUCTION_TEMPLATES.get(
        args.category, "find the most informative view of the object"
    )

    if args.object_ids:
        object_ids = [x.strip() for x in args.object_ids.split(",") if x.strip()]
    else:
        object_ids = default_object_ids(args.category, args.objects)

    annotations = []
    for obj_id in object_ids:
        source = Path(args.raw_root) / args.category / obj_id
        entry = process_object(
            args.category,
            obj_id,
            source,
            out_root,
            instruction,
            args.views_per_object,
        )
        if entry is not None:
            annotations.append(entry)

    save_annotations(ann_path, annotations)
    print(f"\nSaved {len(annotations)} samples -> {ann_path}")
    print(f"Subset root: {out_root}")


def expand_existing(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    ann_path = out_root / "annotations.json"
    existing = load_annotations(ann_path)
    existing_ids = {item["id"] for item in existing}
    instruction = args.instruction or INSTRUCTION_TEMPLATES.get(
        args.category, "find the most informative view of the object"
    )

    rng = random.Random(args.seed)
    candidates = candidate_sources(
        Path(args.raw_root),
        args.category,
        args.views_per_object,
        existing_ids,
    )
    if len(candidates) < args.objects:
        raise RuntimeError(
            f"{args.category}: need {args.objects} new objects, found {len(candidates)}"
        )

    selected = rng.sample(candidates, args.objects)
    new_ids = next_object_ids(args.category, existing_ids, args.objects)
    added = []
    manifest = []

    for new_id, (source_name, source_dir, num_images) in zip(new_ids, selected):
        entry = process_object(
            args.category,
            new_id,
            source_dir,
            out_root,
            instruction,
            args.views_per_object,
        )
        if entry is None:
            continue
        added.append(entry)
        manifest.append(
            {
                "new_id": new_id,
                "source_name": source_name,
                "source_dir": str(source_dir).replace("\\", "/"),
                "num_source_images": num_images,
            }
        )

    merged = existing + added
    save_annotations(ann_path, merged)
    manifest_path = out_root / "expansion_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "category": args.category,
                "seed": args.seed,
                "added_count": len(added),
                "objects": manifest,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(
        f"\n[DONE] {args.category}: {len(existing)} -> {len(merged)} (+{len(added)})"
    )
    print(f"Annotations: {ann_path}")
    print(f"Manifest:    {manifest_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build or expand a CO3D view subset")
    p.add_argument("--category", required=True, help="e.g. cup, book, teddybear")
    p.add_argument("--raw_root", default=r"E:\CO3D-data")
    p.add_argument(
        "--out_root",
        default=None,
        help="Default: E:/CO3D-subset-<category>",
    )
    p.add_argument("--objects", type=int, default=10, help="How many objects to build/add")
    p.add_argument(
        "--object_ids",
        default=None,
        help="Comma-separated IDs (fresh build only), e.g. cup_001,cup_002",
    )
    p.add_argument("--views_per_object", type=int, default=5)
    p.add_argument("--instruction", default=None)
    p.add_argument("--expand", action="store_true", help="Append new objects to an existing subset")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.out_root is None:
        args.out_root = rf"E:\CO3D-subset-{args.category}"
    return args


def main() -> None:
    args = parse_args()
    if args.expand:
        expand_existing(args)
    else:
        build_fresh(args)


if __name__ == "__main__":
    main()
