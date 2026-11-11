import json
import csv
from pathlib import Path

summary_json = Path("outputs_all3/budget_sweep_summary.json")
out_csv = Path("outputs_all3/per_category_summary.csv")

with open(summary_json, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []
for budget, metrics in data.items():
    per_cat = metrics.get("per_category", {})
    for category, vals in per_cat.items():
        rows.append({
            "budget": budget,
            "category": category,
            "num_samples": vals.get("num_samples"),
            "accuracy": vals.get("accuracy"),
            "avg_steps": vals.get("avg_steps"),
        })

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved per-category CSV to: {out_csv}")