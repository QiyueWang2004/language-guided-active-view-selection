import json
import csv
from pathlib import Path

out_dir = Path("outputs_all3")
out_csv = out_dir / "method_vs_budget.csv"

rows = []
budgets = [1, 2, 3, 4, 5]

# Active policy: one value per budget
for budget in budgets:
    with open(out_dir / f"active_policy_budget_{budget}.json", "r", encoding="utf-8") as f:
        m = json.load(f)
    rows.append({
        "method": "active_policy",
        "budget": budget,
        "accuracy": m.get("accuracy"),
    })

# Random: one constant line
with open(out_dir / "random.json", "r", encoding="utf-8") as f:
    random_metrics = json.load(f)

for budget in budgets:
    rows.append({
        "method": "random",
        "budget": budget,
        "accuracy": random_metrics.get("accuracy"),
    })

# Exhaustive CLIP: one constant line
with open(out_dir / "exhaustive_clip.json", "r", encoding="utf-8") as f:
    exhaustive_metrics = json.load(f)

for budget in budgets:
    rows.append({
        "method": "exhaustive_clip",
        "budget": budget,
        "accuracy": exhaustive_metrics.get("accuracy"),
    })

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["method", "budget", "accuracy"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved to: {out_csv}")