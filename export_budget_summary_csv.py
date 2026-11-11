import pandas as pd
import matplotlib.pyplot as plt

csv_path = r"outputs_all3\budget_sweep_summary.csv"
df = pd.read_csv(csv_path)

df["budget"] = df["budget"].astype(int)
df = df.sort_values("budget")

plt.figure(figsize=(8, 5))
plt.plot(df["budget"], df["top_1_accuracy"], marker="o", linewidth=2, label="Top-1")
plt.plot(df["budget"], df["top_2_accuracy"], marker="s", linewidth=2, label="Top-2")
plt.plot(df["budget"], df["top_3_accuracy"], marker="^", linewidth=2, label="Top-3")

plt.xlabel("Observation budget", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title("Top-k Accuracy vs Observation Budget", fontsize=13)
plt.xticks(df["budget"])
plt.ylim(0, 1.0)
plt.legend()
plt.tight_layout()
plt.savefig(r"outputs_all3\topk_vs_budget.png", dpi=300)
plt.show()