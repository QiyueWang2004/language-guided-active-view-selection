import os
import json
import matplotlib.pyplot as plt

OUT_DIR = r"outputs_all5"
BUDGETS = [1, 2, 3, 4, 5]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_topk_vs_budget():
    top1 = []
    top2 = []
    top3 = []

    for b in BUDGETS:
        path = os.path.join(OUT_DIR, f"active_policy_budget_{b}.json")
        metrics = load_json(path)
        top1.append(metrics.get("top_1_accuracy", 0.0))
        top2.append(metrics.get("top_2_accuracy", 0.0))
        top3.append(metrics.get("top_3_accuracy", 0.0))

    plt.figure(figsize=(8, 5))
    plt.plot(BUDGETS, top1, marker="o", linewidth=2, label="Top-1")
    plt.plot(BUDGETS, top2, marker="s", linewidth=2, label="Top-2")
    plt.plot(BUDGETS, top3, marker="^", linewidth=2, label="Top-3")

    plt.xlabel("Observation budget")
    plt.ylabel("Accuracy")
    plt.title("Top-k Accuracy vs Observation Budget")
    plt.xticks(BUDGETS)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "topk_vs_budget.png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_topk_vs_budget()