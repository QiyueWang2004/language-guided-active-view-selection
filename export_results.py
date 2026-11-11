"""
Export compact CSVs / plots from evaluation output directories.

Examples:
  python export_results.py summary --out_dir results/umd
  python export_results.py category --out_dir results/co3d_shuffled
  python export_results.py methods --out_dir outputs_all5
  python export_results.py plot-topk --out_dir results/umd
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def find_budget_summary(out_dir: Path) -> Path:
    for name in (
        "budget_sweep_summary_compact.json",
        "budget_sweep_summary.json",
        "trained_active_learned_reranker_budget_sweep.json",
    ):
        path = out_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No budget summary JSON found under {out_dir}")


def cmd_summary(out_dir: Path) -> None:
    src = find_budget_summary(out_dir)
    data = load_json(src)
    rows = []
    for budget, metrics in data.items():
        rows.append(
            {
                "budget": budget,
                "accuracy": metrics.get("accuracy"),
                "accuracy_std": metrics.get("accuracy_std"),
                "top_1_accuracy": metrics.get("top_1_accuracy"),
                "top_2_accuracy": metrics.get("top_2_accuracy"),
                "top_3_accuracy": metrics.get("top_3_accuracy"),
                "avg_steps": metrics.get("avg_steps"),
                "early_stop_rate": metrics.get("early_stop_rate"),
                "num_samples": metrics.get("num_samples"),
                "num_trials": metrics.get("num_trials"),
            }
        )
    write_csv(out_dir / "budget_sweep_summary.csv", rows)


def cmd_category(out_dir: Path) -> None:
    src = find_budget_summary(out_dir)
    data = load_json(src)
    rows = []
    for budget, metrics in data.items():
        for category, vals in (metrics.get("per_category") or {}).items():
            rows.append(
                {
                    "budget": budget,
                    "category": category,
                    "num_samples": vals.get("num_samples"),
                    "accuracy": vals.get("accuracy"),
                    "avg_steps": vals.get("avg_steps"),
                }
            )
    write_csv(out_dir / "per_category_summary.csv", rows)


def cmd_methods(out_dir: Path, budgets: List[int]) -> None:
    rows = []

    def add_constant(method: str, filename: str) -> None:
        path = out_dir / filename
        if not path.exists():
            print(f"[WARN] missing {path}")
            return
        metrics = load_json(path)
        acc = metrics.get("accuracy")
        for b in budgets:
            rows.append({"method": method, "budget": b, "accuracy": acc})

    def add_per_budget(method: str, pattern: str) -> None:
        for b in budgets:
            path = out_dir / pattern.format(budget=b)
            if not path.exists():
                # also accept compact sweep file
                continue
            metrics = load_json(path)
            rows.append({"method": method, "budget": b, "accuracy": metrics.get("accuracy")})

    # Prefer compact sweep files when present
    sweep_map = {
        "learned_reranker": "learned_reranker_budget_sweep.json",
        "trained_active_clip": "trained_active_clip_budget_sweep.json",
        "trained_active_learned_reranker": "trained_active_learned_reranker_budget_sweep.json",
    }
    for method, fname in sweep_map.items():
        path = out_dir / fname
        if path.exists():
            data = load_json(path)
            for b, metrics in data.items():
                rows.append(
                    {
                        "method": method,
                        "budget": int(b) if str(b).isdigit() else b,
                        "accuracy": metrics.get("accuracy"),
                        "accuracy_std": metrics.get("accuracy_std"),
                    }
                )

    add_constant("random", "random.json")
    add_constant("exhaustive_clip", "exhaustive_clip.json")
    add_per_budget("active_policy", "active_policy_budget_{budget}.json")

    write_csv(out_dir / "method_vs_budget.csv", rows)


def cmd_plot_topk(out_dir: Path, budgets: List[int]) -> None:
    src = find_budget_summary(out_dir)
    data = load_json(src)
    xs, top1, top2, top3 = [], [], [], []
    for b in budgets:
        key = str(b)
        if key not in data:
            continue
        m = data[key]
        xs.append(b)
        top1.append(m.get("top_1_accuracy", m.get("accuracy", 0.0)))
        top2.append(m.get("top_2_accuracy", 0.0))
        top3.append(m.get("top_3_accuracy", 0.0))

    if not xs:
        raise RuntimeError(f"No budget entries found in {src}")

    plt.figure(figsize=(8, 5))
    plt.plot(xs, top1, marker="o", linewidth=2, label="Top-1")
    plt.plot(xs, top2, marker="s", linewidth=2, label="Top-2")
    plt.plot(xs, top3, marker="^", linewidth=2, label="Top-3")
    plt.xlabel("Observation budget")
    plt.ylabel("Accuracy")
    plt.title("Top-k Accuracy vs Observation Budget")
    plt.xticks(xs)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    out = out_dir / "topk_vs_budget.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CSVs/plots from eval outputs")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--out_dir", type=str, required=True)
        p.add_argument("--budgets", type=str, default="1,2,3,4,5")

    for name in ("summary", "category", "methods", "plot-topk"):
        add_common(sub.add_parser(name))

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]

    if args.command == "summary":
        cmd_summary(out_dir)
    elif args.command == "category":
        cmd_category(out_dir)
    elif args.command == "methods":
        cmd_methods(out_dir, budgets)
    elif args.command == "plot-topk":
        cmd_plot_topk(out_dir, budgets)


if __name__ == "__main__":
    main()
