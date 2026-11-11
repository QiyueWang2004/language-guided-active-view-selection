import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


RAW_ROOT = Path(r"E:\CO3D-data")
SUBSET_ROOTS = {
    "book": Path(r"E:\CO3D-subset-book"),
    "cup": Path(r"E:\CO3D-subset-cup"),
    "handbag": Path(r"E:\CO3D-subset-handbag"),
    "remote": Path(r"E:\CO3D-subset-remote"),
    "teddybear": Path(r"E:\CO3D-subset-teddybear"),
}

INSTRUCTION_TEMPLATES = {
    "book": "find the view where the book is most front-facing",
    "cup": "find the view where the printed design is front-facing",
    "handbag": "find the view where the handbag is most front-facing",
    "remote": "find the view where the remote is least tilted",
    "teddybear": "find the view where the face is most visible",
}

VIEWS_PER_OBJECT = 5
VIEW_RATIOS = [0.0, 0.20, 0.40, 0.60, 0.80]
OBJECTS_PER_CATEGORY = 20
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42


def list_images(images_dir: Path) -> List[Path]:
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def uniform_sample(images: List[Path], num_samples: int) -> List[Path]:
    n = len(images)
    if n < num_samples:
        raise ValueError(f"Not enough images to sample: {n} < {num_samples}")

    indices = []
    seen = set()
    for ratio in VIEW_RATIOS[:num_samples]:
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


def copy_views(sampled_images: List[Path], out_dir: Path) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for i, src in enumerate(sampled_images, start=1):
        dst = out_dir / f"view_{i:02d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        saved_paths.append(str(dst).replace("\\", "/"))
    return saved_paths


def load_annotations(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_annotations(path: Path, data: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def next_object_ids(category: str, existing_ids: List[str], count: int) -> List[str]:
    used_numbers = []
    prefix = f"{category}_"
    for obj_id in existing_ids:
        if obj_id.startswith(prefix):
            suffix = obj_id[len(prefix):]
            if suffix.isdigit():
                used_numbers.append(int(suffix))

    start = max(used_numbers, default=0) + 1
    return [f"{category}_{i:03d}" for i in range(start, start + count)]


def list_candidate_sources(category: str, existing_ids: set) -> List[Tuple[str, Path, int]]:
    raw_dir = RAW_ROOT / category
    candidates = []

    for obj_dir in sorted(raw_dir.iterdir()):
        if not obj_dir.is_dir():
            continue
        if obj_dir.name in existing_ids:
            continue

        images_dir = obj_dir / "images"
        if not images_dir.exists():
            continue

        images = list_images(images_dir)
        if len(images) < VIEWS_PER_OBJECT:
            continue

        candidates.append((obj_dir.name, obj_dir, len(images)))

    return candidates


def expand_category(category: str, rng: random.Random) -> Dict:
    subset_root = SUBSET_ROOTS[category]
    annotation_file = subset_root / "annotations.json"
    existing = load_annotations(annotation_file)
    existing_ids = {item["id"] for item in existing}

    candidates = list_candidate_sources(category, existing_ids)
    if len(candidates) < OBJECTS_PER_CATEGORY:
        raise RuntimeError(
            f"{category}: need {OBJECTS_PER_CATEGORY} new objects, found {len(candidates)} candidates"
        )

    selected = rng.sample(candidates, OBJECTS_PER_CATEGORY)
    new_ids = next_object_ids(category, list(existing_ids), OBJECTS_PER_CATEGORY)
    instruction = INSTRUCTION_TEMPLATES[category]
    added = []
    manifest = []

    for new_id, (source_name, source_dir, num_images) in zip(new_ids, selected):
        images = list_images(source_dir / "images")
        sampled = uniform_sample(images, VIEWS_PER_OBJECT)
        out_obj_dir = subset_root / category / new_id
        saved_views = copy_views(sampled, out_obj_dir)

        entry = {
            "id": new_id,
            "category": category,
            "instruction": instruction,
            "views": saved_views,
            "gt_views": [],
        }
        added.append(entry)
        manifest.append(
            {
                "new_id": new_id,
                "source_name": source_name,
                "source_dir": str(source_dir).replace("\\", "/"),
                "num_source_images": num_images,
                "sampled_source_files": [p.name for p in sampled],
            }
        )
        print(f"[INFO] {category}: {source_name} -> {new_id} ({num_images} images)")

    merged = existing + added
    save_annotations(annotation_file, merged)

    manifest_path = subset_root / "expansion_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "category": category,
                "seed": SEED,
                "added_count": len(added),
                "objects": manifest,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "category": category,
        "previous_count": len(existing),
        "added_count": len(added),
        "total_count": len(merged),
        "manifest_path": str(manifest_path),
    }


def main() -> None:
    rng = random.Random(SEED)
    summaries = []

    for category in SUBSET_ROOTS:
        summary = expand_category(category, rng)
        summaries.append(summary)
        print(
            f"[DONE] {category}: {summary['previous_count']} -> {summary['total_count']} "
            f"(+{summary['added_count']})"
        )

    print("\n===== Expansion Summary =====")
    for summary in summaries:
        print(
            f"{summary['category']}: {summary['total_count']} total "
            f"(added {summary['added_count']})"
        )
        print(f"  manifest: {summary['manifest_path']}")


if __name__ == "__main__":
    main()
