"""
Ablation study for the active view selection pipeline.

The pipeline has two stages:
  1) view selection: which views to observe within the budget
  2) final reranking: which observed view to output

Each configuration combines one selector and one reranker. Every configuration is
compared against the baseline (random selection + CLIP rerank) per category and
per budget. All configurations share the same random initial view per sample and
trial, so the deltas are paired comparisons.

Every reported gain is accuracy(method) - accuracy(baseline), where the method
replaces one or both baseline stages with the named component.
"""

import argparse
import csv
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from active_view_selection import (
    ActiveSelectionPolicy,
    ActiveViewDataset,
    CLIPScorer,
    FeatureCache,
    SampleItem,
    TrainedActiveSelectionPolicy,
    ViewReranker,
    add_split_args,
    rank_observed_views,
    split_dataset,
    train_view_reranker,
    train_view_selector,
)


BASELINE = "random+clip"
BASELINE_LABEL = "Random selection + CLIP rerank"
GAIN_LEGEND_TITLE = (
    f"Baseline = {BASELINE_LABEL}\n"
    "Each value = accuracy(method) - accuracy(baseline)"
)

# (key, label, selector, reranker)
CONFIGS: List[Tuple[str, str, str, str]] = [
    ("random+clip", f"Baseline: {BASELINE_LABEL}", "random", "clip"),
    ("random+learned", "Random selection + Learned rerank", "random", "learned"),
    ("clip_rel+clip", "CLIP-relevance selection + CLIP rerank", "clip_rel", "clip"),
    ("clip_rel_nov+clip", "CLIP-relevance+novelty selection + CLIP rerank", "clip_rel_nov", "clip"),
    ("clip_rel_nov+learned", "CLIP-relevance+novelty selection + Learned rerank", "clip_rel_nov", "learned"),
    ("trained+clip", "Trained selection + CLIP rerank", "trained", "clip"),
    ("trained+learned", "Trained selection + Learned rerank (full pipeline)", "trained", "learned"),
]

# (key, label, method config, baseline config). Line 1 of the label names what
# differs from the baseline; line 2 gives the full method configuration.
COMPONENTS: List[Tuple[str, str, str, str]] = [
    ("learned_rerank",
     "Learned rerank vs baseline\n[Random selection + Learned rerank]",
     "random+learned", BASELINE),
    ("clip_rel_selection",
     "CLIP-relevance selection vs baseline\n[CLIP-relevance selection + CLIP rerank]",
     "clip_rel+clip", BASELINE),
    ("clip_rel_nov_selection",
     "CLIP-relevance+novelty selection vs baseline\n[CLIP-relevance+novelty selection + CLIP rerank]",
     "clip_rel_nov+clip", BASELINE),
    ("trained_selection",
     "Trained selection vs baseline\n[Trained selection + CLIP rerank]",
     "trained+clip", BASELINE),
    ("clip_rel_nov_selection_learned_rerank",
     "CLIP-relevance+novelty selection & learned rerank vs baseline\n"
     "[CLIP-relevance+novelty selection + Learned rerank]",
     "clip_rel_nov+learned", BASELINE),
    ("full_pipeline",
     "Trained selection & learned rerank (full pipeline) vs baseline\n"
     "[Trained selection + Learned rerank]",
     "trained+learned", BASELINE),
]

CONFIG_LABELS = {key: label for key, label, _, _ in CONFIGS}
COMPONENT_LABELS = {key: label for key, label, _, _ in COMPONENTS}


def flat_label(label: str) -> str:
    return label.replace("\n", " ")


# ============================================================
# Models
# ============================================================

def load_view_mlp(path: str, device: torch.device) -> Tuple[ViewReranker, Dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    info = ckpt["train_info"]
    model = ViewReranker(feat_dim=info["feat_dim"], hidden_dim=info["hidden_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, info


def prepare_models(
    args: argparse.Namespace,
    dataset: ActiveViewDataset,
    feature_cache: FeatureCache,
    device: torch.device,
) -> Tuple[ViewReranker, ViewReranker, Dict[str, Any]]:
    if args.load_models_from:
        reranker, reranker_info = load_view_mlp(os.path.join(args.load_models_from, "reranker.pt"), device)
        selector, selector_info = load_view_mlp(os.path.join(args.load_models_from, "view_selector.pt"), device)
        print(f"Loaded reranker and view selector from {args.load_models_from}")
        return reranker, selector, {"reranker": reranker_info, "selector": selector_info, "loaded_from": args.load_models_from}

    train_dataset, val_dataset, _, split_desc = split_dataset(dataset, args)
    train_kwargs = dict(
        train_samples=train_dataset.samples,
        val_samples=val_dataset.samples,
        feature_cache=feature_cache,
        device=device,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )

    print(f"Training reranker and view selector ({split_desc})")
    reranker, reranker_info = train_view_reranker(**train_kwargs)
    selector, selector_info = train_view_selector(**train_kwargs)
    print(f"Reranker best val accuracy: {reranker_info['best_val_accuracy']:.4f}")
    print(f"View selector best val accuracy: {selector_info['best_val_accuracy']:.4f}")

    torch.save({"state_dict": reranker.state_dict(), "train_info": reranker_info},
               os.path.join(args.out_dir, "reranker.pt"))
    torch.save({"state_dict": selector.state_dict(), "train_info": selector_info},
               os.path.join(args.out_dir, "view_selector.pt"))
    return reranker, selector, {"reranker": reranker_info, "selector": selector_info}


# ============================================================
# Evaluation
# ============================================================

def run_pipeline(
    samples: List[SampleItem],
    initial_indices: List[int],
    selector_kind: str,
    reranker_mode: str,
    budget: int,
    selection_rng: random.Random,
    selectors: Dict[str, Any],
    feature_cache: FeatureCache,
    reranker: ViewReranker,
    device: torch.device,
) -> List[Dict[str, Any]]:
    per_sample = []
    for sample, init_idx in zip(samples, initial_indices):
        num_views = len(sample.views)
        seen = [init_idx]

        while len(seen) < min(budget, num_views):
            if selector_kind == "random":
                candidates = [i for i in range(num_views) if i not in seen]
                next_idx = selection_rng.choice(candidates)
            else:
                next_idx = selectors[selector_kind].select_next(sample.query, sample.views, seen)
                if next_idx is None:
                    break
            seen.append(next_idx)

        ranked, _ = rank_observed_views(
            sample=sample,
            seen_indices=seen,
            feature_cache=feature_cache,
            reranker_mode=reranker_mode,
            device=device,
            reranker=reranker,
        )
        per_sample.append(
            {
                "sample_id": sample.sample_id,
                "category": sample.category,
                "selected_indices": seen,
                "final_index": ranked[0],
                "correct": bool(sample.views[ranked[0]].is_positive),
                "gt_observed": any(sample.views[i].is_positive for i in seen),
            }
        )
    return per_sample


def summarize_trial(per_sample: List[Dict[str, Any]], categories: List[str]) -> Dict[str, Dict[str, float]]:
    out = {}
    for cat in categories + ["overall"]:
        items = per_sample if cat == "overall" else [x for x in per_sample if x["category"] == cat]
        n = max(1, len(items))
        out[cat] = {
            "accuracy": sum(x["correct"] for x in items) / n,
            "gt_coverage": sum(x["gt_observed"] for x in items) / n,
            "num_samples": len(items),
        }
    return out


def mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std())


def aggregate(
    trials: Dict[str, Dict[int, List[Dict[str, Dict[str, float]]]]],
    budgets: List[int],
    categories: List[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cats = categories + ["overall"]

    config_summary: Dict[str, Any] = {}
    for cfg_key, _, _, _ in CONFIGS:
        config_summary[cfg_key] = {}
        for b in budgets:
            config_summary[cfg_key][str(b)] = {}
            for cat in cats:
                acc = [t[cat]["accuracy"] for t in trials[cfg_key][b]]
                cov = [t[cat]["gt_coverage"] for t in trials[cfg_key][b]]
                base_acc = [t[cat]["accuracy"] for t in trials[BASELINE][b]]
                delta = [a - c for a, c in zip(acc, base_acc)]
                acc_m, acc_s = mean_std(acc)
                cov_m, cov_s = mean_std(cov)
                d_m, d_s = mean_std(delta)
                config_summary[cfg_key][str(b)][cat] = {
                    "accuracy_mean": acc_m,
                    "accuracy_std": acc_s,
                    "gt_coverage_mean": cov_m,
                    "gt_coverage_std": cov_s,
                    "delta_vs_baseline_mean": d_m,
                    "delta_vs_baseline_std": d_s,
                    "num_samples": trials[cfg_key][b][0][cat]["num_samples"],
                }

    component_summary: Dict[str, Any] = {}
    for comp_key, label, method_cfg, baseline_cfg in COMPONENTS:
        component_summary[comp_key] = {
            "label": flat_label(label),
            "method": method_cfg,
            "baseline": baseline_cfg,
        }
        for b in budgets:
            component_summary[comp_key][str(b)] = {}
            for cat in cats:
                gains = [
                    m[cat]["accuracy"] - base[cat]["accuracy"]
                    for m, base in zip(trials[method_cfg][b], trials[baseline_cfg][b])
                ]
                cov_gains = [
                    m[cat]["gt_coverage"] - base[cat]["gt_coverage"]
                    for m, base in zip(trials[method_cfg][b], trials[baseline_cfg][b])
                ]
                g_m, g_s = mean_std(gains)
                c_m, c_s = mean_std(cov_gains)
                component_summary[comp_key][str(b)][cat] = {
                    "accuracy_gain_mean": g_m,
                    "accuracy_gain_std": g_s,
                    "gt_coverage_gain_mean": c_m,
                    "gt_coverage_gain_std": c_s,
                }

    return config_summary, component_summary


# ============================================================
# Reporting
# ============================================================

def write_csvs(
    out_dir: str,
    config_summary: Dict[str, Any],
    component_summary: Dict[str, Any],
    budgets: List[int],
    categories: List[str],
) -> None:
    cats = categories + ["overall"]

    with open(os.path.join(out_dir, "ablation_configs.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "config", "label", "budget", "category",
            "accuracy_mean", "accuracy_std",
            "delta_vs_baseline_mean", "delta_vs_baseline_std",
            "gt_coverage_mean",
        ])
        for cfg_key, label, _, _ in CONFIGS:
            for b in budgets:
                for cat in cats:
                    s = config_summary[cfg_key][str(b)][cat]
                    writer.writerow([
                        cfg_key, label, b, cat,
                        f"{s['accuracy_mean']:.4f}", f"{s['accuracy_std']:.4f}",
                        f"{s['delta_vs_baseline_mean']:.4f}", f"{s['delta_vs_baseline_std']:.4f}",
                        f"{s['gt_coverage_mean']:.4f}",
                    ])

    with open(os.path.join(out_dir, "ablation_components.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "component", "comparison", "method_config", "baseline_config", "budget", "category",
            "accuracy_gain_vs_baseline_mean", "accuracy_gain_vs_baseline_std",
            "gt_coverage_gain_vs_baseline_mean",
        ])
        for comp_key, label, method_cfg, baseline_cfg in COMPONENTS:
            for b in budgets:
                for cat in cats:
                    s = component_summary[comp_key][str(b)][cat]
                    writer.writerow([
                        comp_key, flat_label(label), method_cfg, baseline_cfg, b, cat,
                        f"{s['accuracy_gain_mean']:.4f}", f"{s['accuracy_gain_std']:.4f}",
                        f"{s['gt_coverage_gain_mean']:.4f}",
                    ])


def save_delta_heatmap(
    component_summary: Dict[str, Any],
    budget: int,
    categories: List[str],
    out_path: str,
) -> None:
    cats = categories + ["overall"]
    comp_keys = [key for key, _, _, _ in COMPONENTS]
    matrix = np.array([
        [component_summary[k][str(budget)][c]["accuracy_gain_mean"] * 100 for c in cats]
        for k in comp_keys
    ])
    limit = max(1.0, float(np.abs(matrix).max()))

    fig, ax = plt.subplots(figsize=(1.6 * len(cats) + 6, 0.9 * len(comp_keys) + 2))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats)
    ax.set_yticks(range(len(comp_keys)))
    ax.set_yticklabels([COMPONENT_LABELS[k] for k in comp_keys], fontsize=8)
    for i in range(len(comp_keys)):
        for j in range(len(cats)):
            ax.text(j, i, f"{matrix[i, j]:+.1f}", ha="center", va="center", fontsize=9)
    ax.set_title(
        f"Accuracy gain vs baseline ({BASELINE_LABEL}), budget = {budget}\n"
        "Each cell = accuracy(method) - accuracy(baseline), in percentage points"
    )
    fig.colorbar(im, ax=ax, label="Accuracy gain vs baseline (pp)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def save_component_bars(
    component_summary: Dict[str, Any],
    budget: int,
    categories: List[str],
    out_path: str,
) -> None:
    cats = categories + ["overall"]
    comp_keys = [key for key, _, _, _ in COMPONENTS]
    x = np.arange(len(cats))
    width = 0.8 / len(comp_keys)

    fig, ax = plt.subplots(figsize=(2.2 * len(cats) + 3, 5.5))
    for i, comp in enumerate(comp_keys):
        means = [component_summary[comp][str(budget)][c]["accuracy_gain_mean"] * 100 for c in cats]
        stds = [component_summary[comp][str(budget)][c]["accuracy_gain_std"] * 100 for c in cats]
        ax.bar(x - 0.4 + width * (i + 0.5), means, width, yerr=stds, capsize=2,
               label=COMPONENT_LABELS[comp])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Accuracy gain vs baseline (pp)")
    ax.set_title(f"Accuracy gain vs baseline ({BASELINE_LABEL}) by category, budget = {budget}")
    ax.legend(fontsize=8, title=GAIN_LEGEND_TITLE, title_fontsize=8,
              loc="upper left", bbox_to_anchor=(1.01, 1.0), labelspacing=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_component_gain_vs_budget(
    component_summary: Dict[str, Any],
    budgets: List[int],
    categories: List[str],
    out_path: str,
) -> None:
    cats = categories + ["overall"]
    cols = 3
    rows = int(np.ceil(len(cats) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.8 * rows), squeeze=False)

    for ax, cat in zip(axes.flat, cats):
        for comp_key, label, _, _ in COMPONENTS:
            gains = [component_summary[comp_key][str(b)][cat]["accuracy_gain_mean"] * 100 for b in budgets]
            ax.plot(budgets, gains, marker="o", label=label)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(cat)
        ax.set_xticks(budgets)
        ax.set_xlabel("Observation budget")
        ax.set_ylabel("Accuracy gain vs baseline (pp)")
    for ax in list(axes.flat)[len(cats):]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8,
               title=GAIN_LEGEND_TITLE, title_fontsize=9,
               bbox_to_anchor=(0.5, 0.0), labelspacing=1.0)
    fig.suptitle(f"Accuracy gain vs baseline ({BASELINE_LABEL}) across budgets")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def print_budget_report(
    config_summary: Dict[str, Any],
    component_summary: Dict[str, Any],
    budget: int,
    categories: List[str],
) -> None:
    cats = categories + ["overall"]
    name_width = 64
    header = f"{'':{name_width}s}" + "".join(f"{c:>12s}" for c in cats)

    print("=" * len(header))
    print(f"Accuracy per configuration (budget = {budget}); "
          f"gain vs baseline ({BASELINE_LABEL}) in pp in parentheses")
    print("=" * len(header))
    print(header)
    for cfg_key, label, _, _ in CONFIGS:
        cells = []
        for c in cats:
            s = config_summary[cfg_key][str(budget)][c]
            if cfg_key == BASELINE:
                cells.append(f"{s['accuracy_mean']:.3f}")
            else:
                cells.append(f"{s['accuracy_mean']:.3f}({s['delta_vs_baseline_mean'] * 100:+.0f})")
        print(f"{label[:name_width]:{name_width}s}" + "".join(f"{x:>12s}" for x in cells))

    print()
    print(f"Accuracy gain vs baseline ({BASELINE_LABEL}) in pp, budget = {budget}")
    print(header)
    for comp_key, label, _, _ in COMPONENTS:
        name = label.split("\n")[0]
        cells = [f"{component_summary[comp_key][str(budget)][c]['accuracy_gain_mean'] * 100:+.1f}" for c in cats]
        print(f"{name[:name_width]:{name_width}s}" + "".join(f"{x:>12s}" for x in cells))
    print("=" * len(header))


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Per-category ablation study for active view selection")
    parser.add_argument("--labels_json", type=str, default=r"E:\CO3D-combined\annotations_all_5sets_shuffled.json")
    parser.add_argument("--out_dir", type=str, default="outputs_ablation")
    parser.add_argument("--load_models_from", type=str, default=None,
                        help="Directory with reranker.pt and view_selector.pt; trains new models if omitted")
    parser.add_argument("--budget", type=int, default=3, help="Budget printed in the console report")
    parser.add_argument("--budget_sweep", type=str, default="1,2,3,4,5")
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--w_novelty", type=float, default=0.5)
    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="openai")

    add_split_args(parser)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    plot_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    dataset = ActiveViewDataset(args.labels_json, group_field=args.group_by)
    scorer = CLIPScorer(model_name=args.clip_model, pretrained=args.clip_pretrained)
    feature_cache = FeatureCache(scorer)
    device = torch.device(scorer.device)

    reranker, selector_model, model_info = prepare_models(args, dataset, feature_cache, device)

    eval_samples = [s for s in split_dataset(dataset, args)[2].samples if len(s.views) > 0]
    categories = sorted({s.category for s in eval_samples})
    budgets = [int(x) for x in args.budget_sweep.split(",") if x.strip()]
    print(f"Eval samples: {len(eval_samples)} | categories: {categories} | budgets: {budgets}")

    selectors = {
        "clip_rel": ActiveSelectionPolicy(feature_cache, w_relevance=1.0, w_novelty=0.0),
        "clip_rel_nov": ActiveSelectionPolicy(feature_cache, w_relevance=1.0, w_novelty=args.w_novelty),
        "trained": TrainedActiveSelectionPolicy(feature_cache, selector_model, device),
    }

    trials: Dict[str, Dict[int, List[Dict[str, Dict[str, float]]]]] = {
        key: {b: [] for b in budgets} for key, _, _, _ in CONFIGS
    }

    total_runs = args.num_trials * len(budgets) * len(CONFIGS)
    with tqdm(total=total_runs, desc="Ablation runs") as pbar:
        for trial_idx in range(args.num_trials):
            trial_seed = args.seed + trial_idx
            init_rng = random.Random(trial_seed)
            initial_indices = [init_rng.randrange(len(s.views)) for s in eval_samples]

            for budget in budgets:
                for cfg_key, _, selector_kind, reranker_mode in CONFIGS:
                    # Same selection seed for configs that differ only in reranker.
                    selection_rng = random.Random(trial_seed * 1000 + budget)
                    per_sample = run_pipeline(
                        samples=eval_samples,
                        initial_indices=initial_indices,
                        selector_kind=selector_kind,
                        reranker_mode=reranker_mode,
                        budget=budget,
                        selection_rng=selection_rng,
                        selectors=selectors,
                        feature_cache=feature_cache,
                        reranker=reranker,
                        device=device,
                    )
                    trials[cfg_key][budget].append(summarize_trial(per_sample, categories))
                    pbar.update(1)

    config_summary, component_summary = aggregate(trials, budgets, categories)

    report_budget = args.budget if args.budget in budgets else budgets[len(budgets) // 2]
    print_budget_report(config_summary, component_summary, report_budget, categories)

    with open(os.path.join(args.out_dir, "ablation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "labels_json": args.labels_json,
                "baseline": BASELINE,
                "configs": {k: {"label": l, "selector": s, "reranker": r} for k, l, s, r in CONFIGS},
                "baseline_label": BASELINE_LABEL,
                "components": {
                    k: {"label": flat_label(l), "method": m, "baseline": base}
                    for k, l, m, base in COMPONENTS
                },
                "budgets": budgets,
                "categories": categories,
                "num_trials": args.num_trials,
                "num_eval_samples": len(eval_samples),
                "w_novelty": args.w_novelty,
                "model_info": model_info,
                "config_summary": config_summary,
                "component_summary": component_summary,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    write_csvs(args.out_dir, config_summary, component_summary, budgets, categories)

    for b in budgets:
        save_delta_heatmap(component_summary, b, categories,
                           os.path.join(plot_dir, f"delta_vs_baseline_heatmap_budget_{b}.png"))
        save_component_bars(component_summary, b, categories,
                            os.path.join(plot_dir, f"component_gain_budget_{b}.png"))
    save_component_gain_vs_budget(component_summary, budgets, categories,
                                  os.path.join(plot_dir, "component_gain_vs_budget.png"))

    print(f"Saved ablation results to: {args.out_dir}")


if __name__ == "__main__":
    main()
