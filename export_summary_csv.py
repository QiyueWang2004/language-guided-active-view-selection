import json
import csv
from pathlib import Path

summary_json = Path("outputs_all3/budget_sweep_summary.json")
out_csv = Path("outputs_all3/budget_sweep_summary.csv")

with open(summary_json, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []
for budget, metrics in data.items():
    rows.append({
        "budget": budget,
        "accuracy": metrics.get("accuracy"),
        "top_1_accuracy": metrics.get("top_1_accuracy"),
        "top_2_accuracy": metrics.get("top_2_accuracy"),
        "top_3_accuracy": metrics.get("top_3_accuracy"),
        "avg_steps": metrics.get("avg_steps"),
        "early_stop_rate": metrics.get("early_stop_rate"),
        "num_samples": metrics.get("num_samples"),
    })

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved CSV summary to: {out_csv}")