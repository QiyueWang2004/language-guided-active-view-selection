import json
import shutil
from pathlib import Path
from typing import List, Dict


# ========= Config =========
RAW_ROOT = Path(r"E:\CO3D-data")
OUT_ROOT = Path(r"E:\CO3D-subset")
ANNOTATION_FILE = OUT_ROOT / "annotations.json"

CATEGORIES = ["cup", "book", "teddybear"]   # 按你的实际类别名改
OBJECTS_PER_CATEGORY = 10
VIEWS_PER_OBJECT = 5

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

INSTRUCTION_TEMPLATES = {
    "cup": "find the view where the printed design is most visible",
    "book": "find the view where the front cover is most visible",
    "teddybear": "find the view where the face is most visible",  
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
    Uniformly sample num_samples images from a sequence.
    For 5 samples: about 10%, 30%, 50%, 70%, 90%.
    """
    n = len(images)
    if n < num_samples:
        raise ValueError(f"Not enough images to sample: {n} < {num_samples}")

    ratios = [0.10, 0.30, 0.50, 0.70, 0.90]
    indices = []

    for r in ratios[:num_samples]:
        idx = round((n - 1) * r)
        indices.append(idx)

    # remove duplicates
    unique_indices = []
    seen = set()
    for idx in indices:
        if idx not in seen:
            unique_indices.append(idx)
            seen.add(idx)

    # fill if duplicates made count too small
    if len(unique_indices) < num_samples:
        for i in range(n):
            if i not in seen:
                unique_indices.append(i)
                seen.add(i)
            if len(unique_indices) == num_samples:
                break

    unique_indices = sorted(unique_indices[:num_samples])
    return [images[i] for i in unique_indices]


def choose_object_folders(category_dir: Path, max_objects: int) -> List[Path]:
    """
    Each object folder should contain an 'images' subfolder.
    Example:
        E:/CO3D-data/cup/cup_001/images
    """
    candidates = sorted([p for p in category_dir.iterdir() if p.is_dir()])
    valid = []

    for obj_dir in candidates:
        images_dir = obj_dir / "images"
        if not images_dir.exists():
            continue

        images = list_images(images_dir)
        if len(images) >= VIEWS_PER_OBJECT:
            valid.append(obj_dir)

    return valid[:max_objects]


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

    for category in CATEGORIES:
        category_dir = RAW_ROOT / category
        if not category_dir.exists():
            print(f"[WARNING] Missing category: {category}")
            continue

        object_dirs = choose_object_folders(category_dir, OBJECTS_PER_CATEGORY)
        print(f"[INFO] {category}: selected {len(object_dirs)} objects")

        for obj_dir in object_dirs:
            obj_id = obj_dir.name
            images_dir = obj_dir / "images"
            images = list_images(images_dir)
            sampled = uniform_sample(images, VIEWS_PER_OBJECT)

            out_obj_dir = OUT_ROOT / category / obj_id
            saved_views = copy_views(sampled, out_obj_dir)

            instruction = INSTRUCTION_TEMPLATES.get(
                category,
                "find the most informative view of the object"
            )

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