import json
from pathlib import Path
from collections import Counter

# ========= Input annotation files =========
ANNOTATION_FILES = [
    Path(r"E:\CO3D-subset-cup\annotations.json"),             # cup
    Path(r"E:\CO3D-subset-book\annotations.json"),        # book
    Path(r"E:\CO3D-subset-teddybear\annotations.json"),   # teddybear
    Path(r"E:\CO3D-subset-handbag\annotations.json"),     # handbag
    Path(r"E:\CO3D-subset-remote\annotations.json"),      # remote
]

# ========= Output =========
OUT_JSON = Path(r"E:\CO3D-combined\annotations_all_5sets.json")

# If True, prefix IDs with category, e.g. "cup::cup_001"
PREFIX_ID_WITH_CATEGORY = True

# If True, skip samples with missing image files
SKIP_MISSING_FILES = True


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_sample(sample: dict, source_file: Path):
    required = ["id", "category", "instruction", "views", "gt_views"]
    for key in required:
        if key not in sample:
            raise ValueError(f"Missing key '{key}' in {source_file}: {sample}")

    if not isinstance(sample["views"], list) or len(sample["views"]) == 0:
        raise ValueError(f"'views' must be a non-empty list in {source_file}: {sample['id']}")

    if not isinstance(sample["gt_views"], list):
        raise ValueError(f"'gt_views' must be a list in {source_file}: {sample['id']}")


def file_exists(path_str: str) -> bool:
    return Path(path_str).exists()


def main():
    merged = []
    counts = Counter()
    seen_ids = set()

    total_in = 0
    skipped_missing = 0
    skipped_duplicate = 0

    for ann_file in ANNOTATION_FILES:
        if not ann_file.exists():
            print(f"[WARNING] Missing annotation file: {ann_file}")
            continue

        data = load_json(ann_file)
        print(f"[INFO] Loaded {len(data)} samples from {ann_file}")
        total_in += len(data)

        for sample in data:
            check_sample(sample, ann_file)

            sample = dict(sample)  # copy
            category = sample["category"]

            if PREFIX_ID_WITH_CATEGORY:
                sample["id"] = f"{category}::{sample['id']}"

            if sample["id"] in seen_ids:
                print(f"[WARNING] Duplicate sample id skipped: {sample['id']}")
                skipped_duplicate += 1
                continue

            missing_paths = []
            for p in sample["views"]:
                if not file_exists(p):
                    missing_paths.append(p)
            for p in sample["gt_views"]:
                if not file_exists(p):
                    missing_paths.append(p)

            if missing_paths and SKIP_MISSING_FILES:
                print(f"[WARNING] Skipping {sample['id']} because of missing files:")
                for p in missing_paths:
                    print(f"    {p}")
                skipped_missing += 1
                continue

            seen_ids.add(sample["id"])
            counts[category] += 1
            merged.append(sample)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print("\n===== Merge Summary =====")
    print(f"Total input samples: {total_in}")
    print(f"Total merged samples: {len(merged)}")
    print(f"Skipped duplicates: {skipped_duplicate}")
    print(f"Skipped missing-file samples: {skipped_missing}")

    print("\nPer-category counts:")
    for cat, n in sorted(counts.items()):
        print(f"  {cat}: {n}")

    print(f"\nSaved merged annotations to:\n{OUT_JSON}")


if __name__ == "__main__":
    main()