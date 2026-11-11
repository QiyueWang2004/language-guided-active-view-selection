"""
Evaluate the active-view annotation dataset.

Checks schema, file existence, split coverage, and GT-view statistics.
Default input: E:/CO3D-combined/annotations_all_5sets.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


REQUIRED_KEYS = ["id", "category", "instruction", "views", "gt_views"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_object_number(sample_id: str) -> Optional[int]:
    base = sample_id.split("::")[-1]
    suffix = base.rsplit("_", 1)[-1]
    if not suffix.isdigit():
        return None
    return int(suffix)


def split_name(obj_num: Optional[int], train_min: int, train_max: int,
               val_min: int, val_max: int,
               eval_min: int, eval_max: int) -> str:
    if obj_num is None:
        return "unknown"
    if eval_min <= obj_num <= eval_max:
        return "eval"
    if val_min <= obj_num <= val_max:
        return "val"
    if train_min <= obj_num <= train_max:
        return "train"
    return "other"


def load_annotations(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def gt_indices(sample: Dict[str, Any]) -> List[int]:
    views = sample["views"]
    view_to_idx = {p: i for i, p in enumerate(views)}
    return [view_to_idx[p] for p in sample["gt_views"] if p in view_to_idx]


def evaluate_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[str] = []

    for key in REQUIRED_KEYS:
        if key not in sample:
            issues.append(f"missing_key:{key}")

    if issues:
        return {
            "id": sample.get("id", "<missing>"),
            "category": sample.get("category", "unknown"),
            "issues": issues,
            "n_views": 0,
            "n_gt": 0,
            "n_missing_views": 0,
            "n_missing_gt": 0,
            "gt_indices": [],
        }

    views = sample["views"]
    gt_views = sample["gt_views"]

    if not isinstance(views, list) or len(views) == 0:
        issues.append("empty_views")
    if not isinstance(gt_views, list):
        issues.append("gt_views_not_list")
    if isinstance(gt_views, list) and len(gt_views) == 0:
        issues.append("empty_gt")

    if len(views) != len(set(views)):
        issues.append("duplicate_views")
    if len(gt_views) != len(set(gt_views)):
        issues.append("duplicate_gt")

    gt_not_in_views = [p for p in gt_views if p not in views]
    if gt_not_in_views:
        issues.append(f"gt_not_in_views:{len(gt_not_in_views)}")

    missing_views = [p for p in views if not Path(p).exists()]
    missing_gt = [p for p in gt_views if not Path(p).exists()]
    if missing_views:
        issues.append(f"missing_view_files:{len(missing_views)}")
    if missing_gt:
        issues.append(f"missing_gt_files:{len(missing_gt)}")

    for p in views:
        if Path(p).suffix.lower() not in IMAGE_EXTS:
            issues.append(f"bad_view_ext:{Path(p).suffix}")
            break

    indices = gt_indices(sample)
    return {
        "id": sample["id"],
        "category": sample.get("category", "unknown"),
        "instruction": sample.get("instruction", ""),
        "object_number": parse_object_number(sample["id"]),
        "n_views": len(views),
        "n_gt": len(gt_views),
        "n_missing_views": len(missing_views),
        "n_missing_gt": len(missing_gt),
        "gt_indices": indices,
        "issues": issues,
    }


def summarize(samples: List[Dict[str, Any]], reports: List[Dict[str, Any]],
              args: argparse.Namespace) -> Dict[str, Any]:
    issue_counter: Counter = Counter()
    for r in reports:
        for issue in r["issues"]:
            issue_counter[issue.split(":")[0]] += 1

    by_category: Dict[str, Dict[str, Any]] = {}
    by_split: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "num_samples": 0,
        "with_gt": 0,
        "empty_gt": 0,
        "categories": Counter(),
    })

    gt_index_hist: Counter = Counter()
    n_gt_hist: Counter = Counter()
    n_views_hist: Counter = Counter()
    instruction_by_category: Dict[str, set] = defaultdict(set)

    for sample, report in zip(samples, reports):
        cat = sample.get(args.group_by, report["category"])
        obj_num = report["object_number"]
        if args.split_by == "field":
            split = sample.get("split", "unknown")
        else:
            split = split_name(
                obj_num,
                args.train_obj_min, args.train_obj_max,
                args.val_obj_min, args.val_obj_max,
                args.eval_obj_min, args.eval_obj_max,
            )

        if cat not in by_category:
            by_category[cat] = {
                "num_samples": 0,
                "with_gt": 0,
                "empty_gt": 0,
                "avg_n_views": 0.0,
                "avg_n_gt": 0.0,
                "gt_index_hist": Counter(),
                "object_numbers": [],
                "issues": 0,
            }

        by_category[cat]["num_samples"] += 1
        by_category[cat]["avg_n_views"] += report["n_views"]
        by_category[cat]["avg_n_gt"] += report["n_gt"]
        by_category[cat]["object_numbers"].append(obj_num)
        if report["n_gt"] > 0:
            by_category[cat]["with_gt"] += 1
        else:
            by_category[cat]["empty_gt"] += 1
        if report["issues"]:
            by_category[cat]["issues"] += 1
        for idx in report["gt_indices"]:
            by_category[cat]["gt_index_hist"][idx] += 1
            gt_index_hist[idx] += 1

        n_gt_hist[report["n_gt"]] += 1
        n_views_hist[report["n_views"]] += 1
        instruction_by_category[cat].add(sample.get("instruction", ""))

        by_split[split]["num_samples"] += 1
        by_split[split]["categories"][cat] += 1
        if report["n_gt"] > 0:
            by_split[split]["with_gt"] += 1
        else:
            by_split[split]["empty_gt"] += 1

    for cat, stats in by_category.items():
        n = max(1, stats["num_samples"])
        stats["avg_n_views"] = stats["avg_n_views"] / n
        stats["avg_n_gt"] = stats["avg_n_gt"] / n
        nums = [x for x in stats["object_numbers"] if x is not None]
        stats["object_min"] = min(nums) if nums else None
        stats["object_max"] = max(nums) if nums else None
        stats["gt_index_hist"] = {str(k): v for k, v in sorted(stats["gt_index_hist"].items())}
        del stats["object_numbers"]

    for split_stats in by_split.values():
        split_stats["categories"] = dict(sorted(split_stats["categories"].items()))

    ids = [r["id"] for r in reports]
    duplicate_ids = [sid for sid, c in Counter(ids).items() if c > 1]

    return {
        "num_samples": len(samples),
        "num_categories": len(by_category),
        "num_with_issues": sum(1 for r in reports if r["issues"]),
        "num_empty_gt": sum(1 for r in reports if r["n_gt"] == 0),
        "num_missing_files": sum(
            1 for r in reports if r["n_missing_views"] > 0 or r["n_missing_gt"] > 0
        ),
        "duplicate_ids": duplicate_ids,
        "issue_counts": dict(issue_counter),
        "n_views_hist": {str(k): v for k, v in sorted(n_views_hist.items())},
        "n_gt_hist": {str(k): v for k, v in sorted(n_gt_hist.items())},
        "gt_index_hist": {str(k): v for k, v in sorted(gt_index_hist.items())},
        "by_category": by_category,
        "by_split": dict(by_split),
        "instructions": {k: sorted(v) for k, v in instruction_by_category.items()},
        "split_definition": (
            "split field" if args.split_by == "field" else {
                "eval": [args.eval_obj_min, args.eval_obj_max],
                "val": [args.val_obj_min, args.val_obj_max],
                "train": [args.train_obj_min, args.train_obj_max],
            }
        ),
        "random_chance_accuracy": (
            sum(r["n_gt"] / r["n_views"] for r in reports if r["n_views"]) / max(1, len(reports))
        ),
        "samples_with_issues": [
            {"id": r["id"], "category": r["category"], "issues": r["issues"]}
            for r in reports if r["issues"]
        ],
    }


def print_report(summary: Dict[str, Any]) -> None:
    print("=" * 80)
    print("Dataset Evaluation")
    print("=" * 80)
    print(f"Total samples: {summary['num_samples']}")
    print(f"Categories: {summary['num_categories']}")
    print(f"Samples with issues: {summary['num_with_issues']}")
    print(f"Empty gt_views: {summary['num_empty_gt']}")
    print(f"Samples with missing files: {summary['num_missing_files']}")
    print(f"Duplicate IDs: {len(summary['duplicate_ids'])}")
    print(f"Random-pick chance accuracy (mean |GT| / |views|): {summary['random_chance_accuracy']:.3f}")

    print("\n----- Split definition -----")
    if isinstance(summary["split_definition"], dict):
        for name, bounds in summary["split_definition"].items():
            print(f"  {name}: objects {bounds[0]:03d}-{bounds[1]:03d}")
    else:
        print(f"  {summary['split_definition']}")

    print("\n----- Per-split counts -----")
    for split, stats in sorted(summary["by_split"].items()):
        print(
            f"  {split:8s}: n={stats['num_samples']:3d} "
            f"with_gt={stats['with_gt']:3d} empty_gt={stats['empty_gt']:3d} "
            f"cats={stats['categories']}"
        )

    print("\n----- Per-category -----")
    for cat, stats in sorted(summary["by_category"].items()):
        objs = (f"objs={stats['object_min']:03d}-{stats['object_max']:03d} "
                if stats["object_min"] is not None else "")
        print(
            f"  {cat:10s}: n={stats['num_samples']:3d} "
            f"{objs}"
            f"avg_views={stats['avg_n_views']:.1f} avg_gt={stats['avg_n_gt']:.2f} "
            f"empty_gt={stats['empty_gt']} issues={stats['issues']}"
        )
        print(f"             gt_index_hist={stats['gt_index_hist']}")

    print("\n----- View / GT histograms -----")
    print(f"  n_views_hist: {summary['n_views_hist']}")
    print(f"  n_gt_hist:    {summary['n_gt_hist']}")
    print(f"  gt_index_hist:{summary['gt_index_hist']}")

    if summary["issue_counts"]:
        print("\n----- Issue counts -----")
        for k, v in sorted(summary["issue_counts"].items()):
            print(f"  {k}: {v}")

    if summary["samples_with_issues"]:
        print("\n----- Samples with issues (up to 20) -----")
        for item in summary["samples_with_issues"][:20]:
            print(f"  {item['id']}: {item['issues']}")

    print("\n----- Instructions -----")
    for cat, texts in sorted(summary["instructions"].items()):
        for t in texts:
            print(f"  [{cat}] {t}")


def save_plots(summary: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-category sample counts
    cats = sorted(summary["by_category"].keys())
    counts = [summary["by_category"][c]["num_samples"] for c in cats]
    with_gt = [summary["by_category"][c]["with_gt"] for c in cats]

    plt.figure(figsize=(8, 5))
    x = range(len(cats))
    width = 0.35
    plt.bar([i - width / 2 for i in x], counts, width=width, label="Total")
    plt.bar([i + width / 2 for i in x], with_gt, width=width, label="With GT")
    plt.xticks(list(x), cats, rotation=15, ha="right")
    plt.ylabel("Count")
    plt.title("Samples per category")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "samples_per_category.png", dpi=200)
    plt.close()

    # GT index histogram (0-based view index among annotated GT views)
    hist = {int(k): v for k, v in summary["gt_index_hist"].items()}
    if hist:
        idxs = sorted(hist.keys())
        vals = [hist[i] for i in idxs]
        plt.figure(figsize=(7, 4))
        plt.bar([str(i + 1) for i in idxs], vals)  # show as 1-based view numbers
        plt.xlabel("GT view index (1-based)")
        plt.ylabel("Count")
        plt.title("Ground-truth view position distribution")
        plt.tight_layout()
        plt.savefig(out_dir / "gt_index_histogram.png", dpi=200)
        plt.close()

    # Split counts
    splits = sorted(summary["by_split"].keys())
    split_counts = [summary["by_split"][s]["num_samples"] for s in splits]
    plt.figure(figsize=(6, 4))
    plt.bar(splits, split_counts)
    plt.ylabel("Count")
    plt.title("Samples per split")
    plt.tight_layout()
    plt.savefig(out_dir / "samples_per_split.png", dpi=200)
    plt.close()

    # Per-category GT index heat-ish bars
    max_idx = 0
    for cat in cats:
        for k in summary["by_category"][cat]["gt_index_hist"]:
            max_idx = max(max_idx, int(k))
    if cats:
        fig, axes = plt.subplots(1, len(cats), figsize=(3.2 * len(cats), 3.5), squeeze=False)
        for ax, cat in zip(axes[0], cats):
            hist_c = {int(k): v for k, v in summary["by_category"][cat]["gt_index_hist"].items()}
            idxs = list(range(max_idx + 1))
            vals = [hist_c.get(i, 0) for i in idxs]
            ax.bar([str(i + 1) for i in idxs], vals)
            ax.set_title(cat)
            ax.set_xlabel("View #")
            ax.set_ylabel("GT count")
        fig.suptitle("GT view positions by category", y=1.05)
        fig.tight_layout()
        fig.savefig(out_dir / "gt_index_by_category.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate active-view annotation dataset")
    parser.add_argument(
        "--labels_json",
        type=str,
        default=r"E:\CO3D-combined\annotations_all_5sets.json",
    )
    parser.add_argument("--out_dir", type=str, default=r"E:\CO3D-combined\dataset_eval")
    parser.add_argument("--split_by", choices=["object_number", "field"], default="object_number")
    parser.add_argument("--group_by", type=str, default="category")
    parser.add_argument("--train_obj_min", type=int, default=16)
    parser.add_argument("--train_obj_max", type=int, default=30)
    parser.add_argument("--val_obj_min", type=int, default=11)
    parser.add_argument("--val_obj_max", type=int, default=15)
    parser.add_argument("--eval_obj_min", type=int, default=1)
    parser.add_argument("--eval_obj_max", type=int, default=10)
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    labels_path = Path(args.labels_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_annotations(labels_path)
    reports = [evaluate_sample(s) for s in samples]
    summary = summarize(samples, reports, args)
    summary["labels_json"] = str(labels_path)

    print_report(summary)

    summary_path = out_dir / "dataset_eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    sample_reports_path = out_dir / "sample_reports.json"
    with open(sample_reports_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)

    if not args.no_plots:
        save_plots(summary, out_dir)

    print("\n" + "=" * 80)
    print(f"Saved summary to: {summary_path}")
    print(f"Saved per-sample reports to: {sample_reports_path}")
    if not args.no_plots:
        print(f"Saved plots to: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
