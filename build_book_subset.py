import json
import shutil
from pathlib import Path
from typing import List, Dict


# ========= Config =========
RAW_ROOT = Path(r"E:\CO3D-data")
OUT_ROOT = Path(r"E:\CO3D-subset-book")
ANNOTATION_FILE = OUT_ROOT / "annotations.json"

VIEWS_PER_OBJECT = 5
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 你手动指定要保留的 object
SELECTED_OBJECTS = {
    "book": [
        "book_001", "book_002", "book_003", "book_004", "book_005",
        "book_006", "book_007", "book_008", "book_009", "book_010"
    ],
    # 以后可以继续加
    # "book": ["book_001", "book_002"],
    # "teddybear": ["teddybear_001", "teddybear_002"],
}

INSTRUCTION_TEMPLATES = {
    "book": "find the view where the book is most front-facing",
}


def list_images(images_dir: Path) -> List[Path]:
    return sorted(
        [
            p for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
    )


def uniform_sample(images: List[Path], num_samples: int) -> List[Path]:
    """
    Uniformly sample num_samples images.
    For 5 views: roughly 10%, 30%, 50%, 70%, 90%.
    """
    n = len(images)
    if n < num_samples:
        raise ValueError(f"Not enough images to sample: {n} < {num_samples}")

    ratios = [0.10, 0.30, 0.50, 0.70, 0.90]
    indices = []

    for r in ratios[:num_samples]:
        idx = round((n - 1) * r)
        indices.append(idx)

    # 去重
    unique_indices = []
    seen = set()
    for idx in indices:
        if idx not in seen:
            unique_indices.append(idx)
            seen.add(idx)

    # 如果去重后不够，补齐
    if len(unique_indices) < num_samples:
        for i in range(n):
            if i not in seen:
                unique_indices.append(i)
                seen.add(i)
            if len(unique_indices) == num_samples:
                break

    unique_indices = sorted(unique_indices[:num_samples])
    return [images[i] for i in unique_indices]


def copy_views(sampled_images: List[Path], out_dir: Path) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for i, src in enumerate(sampled_images, start=1):
        ext = src.suffix.lower()
        dst = out_dir / f"view_{i:02d}{ext}"
        shutil.copy2(src, dst)
        saved_paths.append(str(dst).replace("\\", "/"))

    return saved_paths


def build_subset() -> List[Dict]:
    annotations = []
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for category, object_ids in SELECTED_OBJECTS.items():
        category_dir = RAW_ROOT / category
        if not category_dir.exists():
            print(f"[WARNING] Missing category folder: {category_dir}")
            continue

        instruction = INSTRUCTION_TEMPLATES.get(
            category,
            "find the most informative view of the object"
        )

        for obj_id in object_ids:
            obj_dir = category_dir / obj_id
            images_dir = obj_dir / "images"

            if not images_dir.exists():
                print(f"[WARNING] Missing images folder: {images_dir}")
                continue

            images = list_images(images_dir)
            if len(images) < VIEWS_PER_OBJECT:
                print(f"[WARNING] Not enough images in {images_dir}: {len(images)}")
                continue

            print(f"[INFO] Processing {category}/{obj_id} ({len(images)} images)")
            sampled = uniform_sample(images, VIEWS_PER_OBJECT)

            out_obj_dir = OUT_ROOT / category / obj_id
            saved_views = copy_views(sampled, out_obj_dir)

            annotations.append({
                "id": obj_id,
                "category": category,
                "instruction": instruction,
                "views": saved_views,
                "gt_views": []
            })

    return annotations


def save_annotations(data: List[Dict]) -> None:
    with open(ANNOTATION_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def print_summary(data: List[Dict]) -> None:
    print("\n===== Summary =====")
    print(f"Total objects: {len(data)}")

    counts = {}
    for item in data:
        counts[item["category"]] = counts.get(item["category"], 0) + 1

    for cat, count in sorted(counts.items()):
        print(f"{cat}: {count}")


if __name__ == "__main__":
    annotations = build_subset()
    save_annotations(annotations)
    print_summary(annotations)
    print(f"\nSaved annotations to: {ANNOTATION_FILE}")
    print(f"Saved subset to: {OUT_ROOT}")