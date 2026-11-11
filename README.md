# Language-Guided Active View Selection

UNSW Thesis C project (COMP4952) — **Language-Guided Active View Selection for Vision-Language Perception**.

Given several photos of the same object and a short English instruction (e.g. *“find the view where the handle is most clearly visible”*), the system chooses which views to open under a budget, then returns the best observed view.

## Pipeline

1. **Frozen CLIP** encodes the instruction and candidate views  
2. **Active view selector** (heuristic CLIP relevance+novelty, or a trained MLP) picks the next view under budget \(k\)  
3. **Final reranker** (CLIP cosine, or a trained language-conditioned MLP) scores only the observed views and returns the answer  

Correct if the predicted view is in the ground-truth set for that instruction.

## Key results

| Dataset | Setting | Headline |
|---|---|---|
| **CO3D** (shuffled instructions) | Full pipeline | ~**53.4%** @ budget 2; learned reranker is the main gain vs CLIP-only (~36%) |
| **UMD+GT** (affordance views) | 20 random-start trials | Trained active + learned reranker **66.9% ± 1.1%** @B2, **68.0% ± 0.8%** @B3, **69.6%** @B5 |

Plots and compact metrics live under [`results/`](results/):

- `results/co3d_shuffled/` — CO3D method comparison plots + checkpoints  
- `results/umd/` — UMD 20-trial plots + checkpoints  
- `results/ablation_co3d/`, `results/ablation_umd/` — component-gain ablations  

## Repository layout

```
active_view_selection.py   # train + evaluate main pipeline
ablation_study.py          # paired ablations vs random+CLIP baseline
build_umd_annotations.py   # UMD mask → GT view labels
build_*_subset.py          # CO3D category subset builders
shuffle_dataset.py         # shuffle candidate view order
evaluate_dataset.py        # annotation QA
plot_comparison.py / export_*.py
annotations/               # CO3D label JSONs (paths may need remapping)
results/                   # plots, compact metrics, .pt checkpoints
requirements.txt
```

Raw multi-view image folders (`book/`, `cup/`, …) are **not** in this repo (≈200 GB locally). Keep images on disk and point the annotation JSON paths at them.

## Setup

```bash
pip install -r requirements.txt
```

Needs PyTorch with CUDA recommended. CLIP weights download via `open_clip` on first run (`ViT-B-32` / `openai`).

## Data

### CO3D annotations

[`annotations/annotations_all_5sets_shuffled.json`](annotations/annotations_all_5sets_shuffled.json) lists samples with:

- `instruction`, `views` (5 paths), `gt_views`  
- Absolute paths like `E:/CO3D-subset-cup/cup/...` — **rewrite** these to your local image root before training  

Object split by ID number (default in code): eval `1–10`, val `11–15`, train `16–30`.

### UMD+GT

Build labels from UMD affordance masks with:

```bash
python build_umd_annotations.py
```

(Adjust paths inside the script for your UMD subset location.)

## Run evaluation / training

Example — CO3D shuffled, budget sweep, train selector + reranker:

```bash
python active_view_selection.py \
  --labels_json annotations/annotations_all_5sets_shuffled.json \
  --out_dir outputs_shuffled \
  --budget_sweep 1,2,3,4,5 \
  --num_trials 10
```

Load existing checkpoints:

```bash
python active_view_selection.py \
  --labels_json annotations/annotations_all_5sets_shuffled.json \
  --load_models_from results/co3d_shuffled \
  --out_dir outputs_eval \
  --budget_sweep 1,2,3,4,5 \
  --num_trials 10
```

Ablations:

```bash
python ablation_study.py \
  --labels_json annotations/annotations_all_5sets_shuffled.json \
  --out_dir outputs_ablation
```

## Methods compared

| Name | Selection | Final ranking |
|---|---|---|
| Random | random extra views | CLIP |
| Exhaustive CLIP | all views | CLIP |
| Active policy | CLIP relevance + novelty | CLIP |
| Learned reranker | random | trained MLP |
| Trained active + CLIP | trained selector | CLIP |
| Full pipeline | trained selector | trained MLP |

## Author

**Qiyue Wang** — UNSW Computer Engineering · Thesis C · Supervisor: Dr Francisco Cruz
