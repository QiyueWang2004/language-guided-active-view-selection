import os
import json
import math
import random
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader

# pip install open_clip_torch pillow tqdm matplotlib
import open_clip
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================
# Expected label format
# ============================================================
# This script assumes you have a JSON file like this:
#
# [
#   {
#     "id": "teddybear_001",
#     "category": "teddybear",
#     "instruction": "find the view where the face is most visible",
#     "views": [
#       "E:/CO3D-subset-teddybear/teddybear/teddybear_001/view_01.jpg",
#       "E:/CO3D-subset-teddybear/teddybear/teddybear_001/view_02.jpg",
#       "E:/CO3D-subset-teddybear/teddybear/teddybear_001/view_03.jpg"
#     ],
#     "gt_views": [
#       "E:/CO3D-subset-teddybear/teddybear/teddybear_001/view_01.jpg"
#     ]
#   }
# ]
#
# Required fields:
# - id
# - category
# - instruction
# - views           (list of image paths)
# - gt_views        (list of one or more positive image paths)
# ============================================================

@dataclass
class ViewItem:
    image_path: str
    is_positive: bool


@dataclass
class SampleItem:
    sample_id: str
    category: str
    query: str
    views: List[ViewItem]
    split: Optional[str] = None


class ActiveViewDataset(Dataset):
    def __init__(self, json_path: str, group_field: str = "category"):
        self.group_field = group_field
        self.samples = self._load_json(json_path)

    def _load_json(self, json_path: str) -> List[SampleItem]:
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        samples = []
        for entry in raw:
            gt_set = set(entry["gt_views"])
            views = [
                ViewItem(
                    image_path=path,
                    is_positive=(path in gt_set),
                )
                for path in entry["views"]
            ]

            samples.append(
                SampleItem(
                    sample_id=entry["id"],
                    category=entry.get(self.group_field, entry.get("category", "unknown")),
                    query=entry["instruction"],
                    views=views,
                    split=entry.get("split"),
                )
            )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> SampleItem:
        return self.samples[idx]


class CLIPScorer:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer([text]).to(self.device)
        text_feat = self.model.encode_text(tokens)
        text_feat = F.normalize(text_feat, dim=-1)
        return text_feat.squeeze(0)

    @torch.no_grad()
    def encode_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        image_feat = self.model.encode_image(image_tensor)
        image_feat = F.normalize(image_feat, dim=-1)
        return image_feat.squeeze(0)

    @torch.no_grad()
    def score(self, text_feat: torch.Tensor, image_feat: torch.Tensor) -> float:
        return float(torch.sum(text_feat * image_feat).item())


class FeatureCache:
    def __init__(self, scorer: CLIPScorer):
        self.scorer = scorer
        self.image_cache: Dict[str, torch.Tensor] = {}
        self.text_cache: Dict[str, torch.Tensor] = {}

    def get_text_feat(self, text: str) -> torch.Tensor:
        if text not in self.text_cache:
            self.text_cache[text] = self.scorer.encode_text(text)
        return self.text_cache[text]

    def get_image_feat(self, image_path: str) -> torch.Tensor:
        if image_path not in self.image_cache:
            self.image_cache[image_path] = self.scorer.encode_image(image_path)
        return self.image_cache[image_path]


class ActiveSelectionPolicy:
    def __init__(
        self,
        feature_cache: FeatureCache,
        w_relevance: float = 1.0,
        w_novelty: float = 0.5,
        stop_threshold: Optional[float] = None,
    ):
        self.feature_cache = feature_cache
        self.w_relevance = w_relevance
        self.w_novelty = w_novelty
        self.stop_threshold = stop_threshold

    def _novelty(self, cand_feat: torch.Tensor, seen_feats: List[torch.Tensor]) -> float:
        if len(seen_feats) == 0:
            return 1.0
        sims = [float(torch.sum(cand_feat * feat).item()) for feat in seen_feats]
        return 1.0 - max(sims)

    def score_candidate(
        self,
        query: str,
        candidate_view: ViewItem,
        seen_views: List[ViewItem],
    ) -> float:
        text_feat = self.feature_cache.get_text_feat(query)
        cand_feat = self.feature_cache.get_image_feat(candidate_view.image_path)
        relevance = float(torch.sum(text_feat * cand_feat).item())

        seen_feats = [self.feature_cache.get_image_feat(v.image_path) for v in seen_views]
        novelty = self._novelty(cand_feat, seen_feats)

        score = (
            self.w_relevance * relevance
            + self.w_novelty * novelty
        )
        return score

    def select_next(
        self,
        query: str,
        all_views: List[ViewItem],
        seen_indices: List[int],
    ) -> int:
        seen_views = [all_views[i] for i in seen_indices]
        best_idx = None
        best_score = -1e9
        for idx, view in enumerate(all_views):
            if idx in seen_indices:
                continue
            s = self.score_candidate(query, view, seen_views)
            if s > best_score:
                best_score = s
                best_idx = idx
        return best_idx

    def best_observed_view(self, query: str, observed_views: List[ViewItem]) -> Tuple[int, float]:
        text_feat = self.feature_cache.get_text_feat(query)
        best_idx = -1
        best_score = -1e9
        for i, view in enumerate(observed_views):
            feat = self.feature_cache.get_image_feat(view.image_path)
            s = float(torch.sum(text_feat * feat).item())
            if s > best_score:
                best_score = s
                best_idx = i
        return best_idx, best_score

    def observed_scores(self, query: str, observed_views: List[ViewItem]) -> List[float]:
        text_feat = self.feature_cache.get_text_feat(query)
        scores = []
        for view in observed_views:
            feat = self.feature_cache.get_image_feat(view.image_path)
            s = float(torch.sum(text_feat * feat).item())
            scores.append(s)
        return scores

    def should_stop(self, query: str, observed_views: List[ViewItem]) -> bool:
        if self.stop_threshold is None:
            return False
        scores = self.observed_scores(query, observed_views)
        if len(scores) == 0:
            return False
        return max(scores) >= self.stop_threshold


# ============================================================
# Baselines
# ============================================================

def random_policy_step(all_views: List[ViewItem], seen_indices: List[int]) -> int:
    candidates = [i for i in range(len(all_views)) if i not in seen_indices]
    return random.choice(candidates)


def parse_object_number(sample_id: str) -> int:
    base = sample_id.split("::")[-1]
    return int(base.rsplit("_", 1)[-1])


def filter_samples_by_object_number(
    samples: List[SampleItem],
    min_num: int,
    max_num: int,
) -> List[SampleItem]:
    return [
        s for s in samples
        if min_num <= parse_object_number(s.sample_id) <= max_num
    ]


def make_dataset_subset(dataset: ActiveViewDataset, min_num: int, max_num: int) -> ActiveViewDataset:
    subset = ActiveViewDataset.__new__(ActiveViewDataset)
    subset.samples = filter_samples_by_object_number(dataset.samples, min_num, max_num)
    return subset


def make_split_subset(dataset: ActiveViewDataset, split: str) -> ActiveViewDataset:
    subset = ActiveViewDataset.__new__(ActiveViewDataset)
    subset.samples = [s for s in dataset.samples if s.split == split]
    return subset


def add_split_args(parser) -> None:
    parser.add_argument("--split_by", choices=["object_number", "field"], default="object_number",
                        help="object_number: split by the trailing object number in the sample id; "
                             "field: use each sample's 'split' value (train/val/test)")
    parser.add_argument("--group_by", type=str, default="category",
                        help="JSON field used to group per-category results (e.g. category, affordance)")
    parser.add_argument("--train_obj_min", type=int, default=16)
    parser.add_argument("--train_obj_max", type=int, default=30)
    parser.add_argument("--val_obj_min", type=int, default=11)
    parser.add_argument("--val_obj_max", type=int, default=15)
    parser.add_argument("--eval_obj_min", type=int, default=1)
    parser.add_argument("--eval_obj_max", type=int, default=10)


def split_dataset(dataset: ActiveViewDataset, args) -> Tuple[ActiveViewDataset, ActiveViewDataset, ActiveViewDataset, str]:
    """Returns (train, val, eval, description)."""
    if args.split_by == "field":
        return (
            make_split_subset(dataset, "train"),
            make_split_subset(dataset, "val"),
            make_split_subset(dataset, "test"),
            "split field (train / val / test)",
        )
    return (
        make_dataset_subset(dataset, args.train_obj_min, args.train_obj_max),
        make_dataset_subset(dataset, args.val_obj_min, args.val_obj_max),
        make_dataset_subset(dataset, args.eval_obj_min, args.eval_obj_max),
        f"objects train {args.train_obj_min}-{args.train_obj_max}, "
        f"val {args.val_obj_min}-{args.val_obj_max}, eval {args.eval_obj_min}-{args.eval_obj_max}",
    )


def exhaustive_clip_best_view(query: str, all_views: List[ViewItem], feature_cache: FeatureCache) -> int:
    text_feat = feature_cache.get_text_feat(query)
    best_idx = -1
    best_score = -1e9
    for i, view in enumerate(all_views):
        feat = feature_cache.get_image_feat(view.image_path)
        s = float(torch.sum(text_feat * feat).item())
        if s > best_score:
            best_score = s
            best_idx = i
    return best_idx


# ============================================================
# Learned reranker (CLIP features + MLP)
# ============================================================

class ViewReranker(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, text_feat: torch.Tensor, image_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([text_feat, image_feat], dim=-1)
        return self.net(x).squeeze(-1)


def _samples_with_labels(samples: List[SampleItem]) -> List[SampleItem]:
    return [s for s in samples if any(v.is_positive for v in s.views)]


def extract_view_tensors(
    samples: List[SampleItem],
    feature_cache: FeatureCache,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    text_feats: List[torch.Tensor] = []
    image_feats: List[torch.Tensor] = []
    labels: List[float] = []

    for sample in samples:
        if not any(v.is_positive for v in sample.views):
            continue
        text_feat = feature_cache.get_text_feat(sample.query)
        for view in sample.views:
            text_feats.append(text_feat)
            image_feats.append(feature_cache.get_image_feat(view.image_path))
            labels.append(1.0 if view.is_positive else 0.0)

    if not labels:
        raise ValueError("No labeled views found for reranker training.")

    return (
        torch.stack(text_feats),
        torch.stack(image_feats),
        torch.tensor(labels, dtype=torch.float32),
    )


@torch.no_grad()
def reranker_object_accuracy(
    reranker: ViewReranker,
    samples: List[SampleItem],
    feature_cache: FeatureCache,
    device: torch.device,
) -> float:
    reranker.eval()
    correct = 0
    total = 0

    for sample in samples:
        if not any(v.is_positive for v in sample.views):
            continue
        text_feat = feature_cache.get_text_feat(sample.query).unsqueeze(0).to(device)
        best_idx = -1
        best_score = -1e9
        for i, view in enumerate(sample.views):
            image_feat = feature_cache.get_image_feat(view.image_path).unsqueeze(0).to(device)
            score = float(reranker(text_feat, image_feat).item())
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx >= 0 and sample.views[best_idx].is_positive:
            correct += 1
        total += 1

    return correct / max(1, total)


def train_view_mlp(
    train_samples: List[SampleItem],
    val_samples: List[SampleItem],
    feature_cache: FeatureCache,
    device: torch.device,
    hidden_dim: int = 256,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 10,
    model_name: str = "view_mlp",
) -> Tuple[ViewReranker, Dict[str, Any]]:
    train_samples = _samples_with_labels(train_samples)
    val_samples = _samples_with_labels(val_samples)

    text_feats, image_feats, labels = extract_view_tensors(train_samples, feature_cache)
    feat_dim = text_feats.shape[-1]
    pos_count = float(labels.sum().item())
    neg_count = float(len(labels) - pos_count)
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], device=device)

    train_loader = DataLoader(
        TensorDataset(text_feats, image_feats, labels),
        batch_size=batch_size,
        shuffle=True,
    )

    model = ViewReranker(feat_dim=feat_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_acc = -1.0
    best_state = None
    stale_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_text, batch_image, batch_labels in train_loader:
            batch_text = batch_text.to(device)
            batch_image = batch_image.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_text, batch_image)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

        val_acc = reranker_object_accuracy(model, val_samples, feature_cache, device)
        avg_loss = total_loss / max(1, num_batches)
        history.append({"epoch": epoch, "train_loss": avg_loss, "val_accuracy": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    train_info = {
        "model_name": model_name,
        "feat_dim": feat_dim,
        "hidden_dim": hidden_dim,
        "train_objects": len(train_samples),
        "val_objects": len(val_samples),
        "train_views": len(labels),
        "best_val_accuracy": best_val_acc,
        "epochs_ran": len(history),
        "history": history,
    }
    return model, train_info


def train_view_reranker(
    train_samples: List[SampleItem],
    val_samples: List[SampleItem],
    feature_cache: FeatureCache,
    device: torch.device,
    hidden_dim: int = 256,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 10,
) -> Tuple[ViewReranker, Dict[str, Any]]:
    return train_view_mlp(
        train_samples=train_samples,
        val_samples=val_samples,
        feature_cache=feature_cache,
        device=device,
        hidden_dim=hidden_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        model_name="reranker",
    )


def train_view_selector(
    train_samples: List[SampleItem],
    val_samples: List[SampleItem],
    feature_cache: FeatureCache,
    device: torch.device,
    hidden_dim: int = 256,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 10,
) -> Tuple[ViewReranker, Dict[str, Any]]:
    return train_view_mlp(
        train_samples=train_samples,
        val_samples=val_samples,
        feature_cache=feature_cache,
        device=device,
        hidden_dim=hidden_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        model_name="active_selector",
    )


def reranker_score_view(
    reranker: ViewReranker,
    text_feat: torch.Tensor,
    image_feat: torch.Tensor,
    device: torch.device,
) -> float:
    reranker.eval()
    with torch.no_grad():
        text_in = text_feat.unsqueeze(0).to(device)
        image_in = image_feat.unsqueeze(0).to(device)
        return float(reranker(text_in, image_in).item())


class TrainedActiveSelectionPolicy:
    """Greedy next-view selection using a trained CLIP+MLP scorer."""

    def __init__(
        self,
        feature_cache: FeatureCache,
        selector: ViewReranker,
        device: torch.device,
    ):
        self.feature_cache = feature_cache
        self.selector = selector
        self.device = device

    def select_next(
        self,
        query: str,
        all_views: List[ViewItem],
        seen_indices: List[int],
    ) -> Optional[int]:
        text_feat = self.feature_cache.get_text_feat(query)
        best_idx = None
        best_score = -1e9
        for idx, view in enumerate(all_views):
            if idx in seen_indices:
                continue
            image_feat = self.feature_cache.get_image_feat(view.image_path)
            score = reranker_score_view(self.selector, text_feat, image_feat, self.device)
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def should_stop(self, query: str, observed_views: List[ViewItem]) -> bool:
        return False


def clip_score_view(
    text_feat: torch.Tensor,
    image_feat: torch.Tensor,
) -> float:
    return float(torch.sum(text_feat * image_feat).item())


def rank_observed_views(
    sample: SampleItem,
    seen_indices: List[int],
    feature_cache: FeatureCache,
    reranker_mode: str,
    device: torch.device,
    reranker: Optional[ViewReranker] = None,
) -> Tuple[List[int], List[Tuple[int, float]]]:
    text_feat = feature_cache.get_text_feat(sample.query)
    scored: List[Tuple[int, float]] = []

    for global_i in seen_indices:
        image_feat = feature_cache.get_image_feat(sample.views[global_i].image_path)
        if reranker_mode == "learned":
            if reranker is None:
                raise ValueError("reranker is required when reranker_mode='learned'")
            score = reranker_score_view(reranker, text_feat, image_feat, device)
        elif reranker_mode == "clip":
            score = clip_score_view(text_feat, image_feat)
        else:
            raise ValueError(f"Unknown reranker_mode: {reranker_mode}")

        scored.append((global_i, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scored], scored


# ============================================================
# Evaluation
# ============================================================

def pick_initial_index(num_views: int, random_initial: bool, fixed_index: int = 0) -> int:
    if num_views <= 0:
        return 0
    if random_initial:
        return random.randrange(num_views)
    return min(fixed_index, num_views - 1)


def aggregate_trial_metrics(trial_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trial_metrics:
        return {}

    scalar_keys = ["accuracy", "avg_steps", "early_stop_rate"]
    for metrics in trial_metrics:
        for key in metrics:
            if key.startswith("top_") and key.endswith("_accuracy"):
                scalar_keys.append(key)
    scalar_keys = sorted(set(scalar_keys))

    aggregated: Dict[str, Any] = {"num_trials": len(trial_metrics), "trials": trial_metrics}
    for key in scalar_keys:
        values = [m[key] for m in trial_metrics if m.get(key) is not None]
        if not values:
            continue
        aggregated[key] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values))

    aggregated["num_samples"] = trial_metrics[-1].get("num_samples")
    aggregated["results"] = trial_metrics[-1].get("results", [])
    return aggregated


def run_evaluation_trials(
    eval_fn,
    num_trials: int = 10,
    base_seed: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    trial_metrics = []
    for trial_idx in range(num_trials):
        if base_seed is not None:
            trial_seed = base_seed + trial_idx
            random.seed(trial_seed)
            np.random.seed(trial_seed)
            torch.manual_seed(trial_seed)
        metrics = eval_fn(**kwargs)
        metrics["trial_idx"] = trial_idx
        if base_seed is not None:
            metrics["trial_seed"] = base_seed + trial_idx
        trial_metrics.append(metrics)
    return aggregate_trial_metrics(trial_metrics)


def evaluate_sequential_policy(
    dataset: ActiveViewDataset,
    feature_cache: FeatureCache,
    active_policy: ActiveSelectionPolicy,
    budget: int = 3,
    initial_index: int = 0,
    random_initial: bool = True,
    topk: Tuple[int, ...] = (1, 2, 3),
    verbose: bool = False,
) -> Dict[str, float]:
    topk_hits = {k: 0 for k in topk}
    success = 0
    total_steps = 0
    early_stop_count = 0
    results = []

    for sample in tqdm(dataset.samples, desc=f"Evaluating active policy @ budget={budget}"):
        if len(sample.views) == 0:
            continue

        seen_indices = [pick_initial_index(len(sample.views), random_initial, initial_index)]
        early_stop = False

        while len(seen_indices) < min(budget, len(sample.views)):
            observed_views = [sample.views[i] for i in seen_indices]
            if active_policy.should_stop(sample.query, observed_views):
                early_stop = True
                early_stop_count += 1
                break

            next_idx = active_policy.select_next(sample.query, sample.views, seen_indices)
            if next_idx is None:
                break
            seen_indices.append(next_idx)

        observed_views = [sample.views[i] for i in seen_indices]
        text_feat = feature_cache.get_text_feat(sample.query)

        scored = []
        for local_i, global_i in enumerate(seen_indices):
            feat = feature_cache.get_image_feat(sample.views[global_i].image_path)
            s = float(torch.sum(text_feat * feat).item())
            scored.append((global_i, s))

        scored.sort(key=lambda x: x[1], reverse=True)
        ranked_indices = [x[0] for x in scored]

        final_global_idx = ranked_indices[0]
        final_view = sample.views[final_global_idx]
        is_correct = final_view.is_positive
        success += int(is_correct)
        total_steps += len(seen_indices)

        for k in topk:
            top_indices = ranked_indices[: min(k, len(ranked_indices))]
            if any(sample.views[idx].is_positive for idx in top_indices):
                topk_hits[k] += 1

        result = {
            "sample_id": sample.sample_id,
            "category": sample.category,
            "query": sample.query,
            "selected_indices": seen_indices,
            "selected_image_paths": [sample.views[i].image_path for i in seen_indices],
            "ranked_observed_indices": ranked_indices,
            "ranked_observed_image_paths": [sample.views[i].image_path for i in ranked_indices],
            "final_index": final_global_idx,
            "final_image_path": final_view.image_path,
            "gt_image_paths": [v.image_path for v in sample.views if v.is_positive],
            "correct": is_correct,
            "steps": len(seen_indices),
            "early_stop": early_stop,
            "scores": [
                {
                    "index": idx,
                    "image_path": sample.views[idx].image_path,
                    "score": score,
                    "is_positive": sample.views[idx].is_positive,
                }
                for idx, score in scored
            ],
        }
        results.append(result)

        if verbose:
            print("-" * 90)
            print(f"Sample: {sample.sample_id}")
            print(f"Instruction: {sample.query}")
            print(f"Selected indices: {seen_indices}")
            print(f"Final: {final_view.image_path}")
            print(f"GT: {[v.image_path for v in sample.views if v.is_positive]}")
            print(f"Correct: {is_correct}")
            print(f"Steps: {len(seen_indices)} | Early stop: {early_stop}")

    n = max(1, len(results))
    metrics = {
        "accuracy": success / n,
        "avg_steps": total_steps / n,
        "num_samples": n,
        "early_stop_rate": early_stop_count / n,
        "results": results,
    }
    for k in topk:
        metrics[f"top_{k}_accuracy"] = topk_hits[k] / n
    return metrics


def evaluate_baseline_random(
    dataset: ActiveViewDataset,
    feature_cache: FeatureCache,
    budget: int = 3,
    initial_index: int = 0,
    random_initial: bool = True,
) -> Dict[str, float]:
    success = 0
    total_steps = 0
    results = []

    for sample in tqdm(dataset.samples, desc="Evaluating random policy"):
        if len(sample.views) == 0:
            continue

        seen_indices = [pick_initial_index(len(sample.views), random_initial, initial_index)]
        while len(seen_indices) < min(budget, len(sample.views)):
            next_idx = random_policy_step(sample.views, seen_indices)
            seen_indices.append(next_idx)

        text_feat = feature_cache.get_text_feat(sample.query)
        best_idx = -1
        best_score = -1e9
        for idx in seen_indices:
            feat = feature_cache.get_image_feat(sample.views[idx].image_path)
            s = float(torch.sum(text_feat * feat).item())
            if s > best_score:
                best_score = s
                best_idx = idx

        final_view = sample.views[best_idx]
        is_correct = final_view.is_positive
        success += int(is_correct)
        total_steps += len(seen_indices)

        results.append(
            {
                "sample_id": sample.sample_id,
                "category": sample.category,
                "query": sample.query,
                "selected_indices": seen_indices,
                "final_index": best_idx,
                "final_image_path": final_view.image_path,
                "correct": is_correct,
                "steps": len(seen_indices),
                "final_score": best_score,
            }
        )

    n = max(1, len(results))
    return {
        "accuracy": success / n,
        "avg_steps": total_steps / n,
        "num_samples": n,
        "results": results,
    }


def evaluate_exhaustive_clip(
    dataset: ActiveViewDataset,
    feature_cache: FeatureCache,
) -> Dict[str, float]:
    success = 0
    results = []

    for sample in tqdm(dataset.samples, desc="Evaluating exhaustive CLIP"):
        if len(sample.views) == 0:
            continue
        best_idx = exhaustive_clip_best_view(sample.query, sample.views, feature_cache)
        final_view = sample.views[best_idx]
        is_correct = final_view.is_positive
        success += int(is_correct)
        results.append(
            {
                "sample_id": sample.sample_id,
                "category": sample.category,
                "query": sample.query,
                "final_index": best_idx,
                "final_image_path": final_view.image_path,
                "correct": is_correct,
                "steps": len(sample.views),
            }
        )

    n = max(1, len(results))
    return {
        "accuracy": success / n,
        "avg_steps": None,
        "num_samples": n,
        "results": results,
    }


def evaluate_baseline_learned_reranker(
    dataset: ActiveViewDataset,
    feature_cache: FeatureCache,
    reranker: ViewReranker,
    device: torch.device,
    budget: int = 3,
    initial_index: int = 0,
    random_initial: bool = True,
) -> Dict[str, float]:
    success = 0
    total_steps = 0
    results = []

    for sample in tqdm(dataset.samples, desc="Evaluating learned reranker"):
        if len(sample.views) == 0:
            continue

        seen_indices = [pick_initial_index(len(sample.views), random_initial, initial_index)]
        while len(seen_indices) < min(budget, len(sample.views)):
            next_idx = random_policy_step(sample.views, seen_indices)
            seen_indices.append(next_idx)

        text_feat = feature_cache.get_text_feat(sample.query)
        best_idx = -1
        best_score = -1e9
        for idx in seen_indices:
            image_feat = feature_cache.get_image_feat(sample.views[idx].image_path)
            s = reranker_score_view(reranker, text_feat, image_feat, device)
            if s > best_score:
                best_score = s
                best_idx = idx

        final_view = sample.views[best_idx]
        is_correct = final_view.is_positive
        success += int(is_correct)
        total_steps += len(seen_indices)

        results.append(
            {
                "sample_id": sample.sample_id,
                "category": sample.category,
                "query": sample.query,
                "selected_indices": seen_indices,
                "final_index": best_idx,
                "final_image_path": final_view.image_path,
                "correct": is_correct,
                "steps": len(seen_indices),
                "final_score": best_score,
            }
        )

    n = max(1, len(results))
    return {
        "accuracy": success / n,
        "avg_steps": total_steps / n,
        "num_samples": n,
        "results": results,
    }


def evaluate_trained_active_policy(
    dataset: ActiveViewDataset,
    feature_cache: FeatureCache,
    selector: ViewReranker,
    device: torch.device,
    budget: int = 3,
    initial_index: int = 0,
    random_initial: bool = True,
    reranker_mode: str = "clip",
    reranker: Optional[ViewReranker] = None,
    topk: Tuple[int, ...] = (1, 2, 3),
) -> Dict[str, float]:
    if reranker_mode == "learned" and reranker is None:
        raise ValueError("reranker is required when reranker_mode='learned'")

    active_policy = TrainedActiveSelectionPolicy(
        feature_cache=feature_cache,
        selector=selector,
        device=device,
    )

    label = "learned reranker" if reranker_mode == "learned" else "CLIP reranker"
    topk_hits = {k: 0 for k in topk}
    success = 0
    total_steps = 0
    results = []

    for sample in tqdm(
        dataset.samples,
        desc=f"Evaluating trained active + {label} @ budget={budget}",
    ):
        if len(sample.views) == 0:
            continue

        seen_indices = [pick_initial_index(len(sample.views), random_initial, initial_index)]

        while len(seen_indices) < min(budget, len(sample.views)):
            next_idx = active_policy.select_next(sample.query, sample.views, seen_indices)
            if next_idx is None:
                break
            seen_indices.append(next_idx)

        ranked_indices, scored = rank_observed_views(
            sample=sample,
            seen_indices=seen_indices,
            feature_cache=feature_cache,
            reranker_mode=reranker_mode,
            device=device,
            reranker=reranker,
        )

        final_global_idx = ranked_indices[0]
        final_view = sample.views[final_global_idx]
        is_correct = final_view.is_positive
        success += int(is_correct)
        total_steps += len(seen_indices)

        for k in topk:
            top_indices = ranked_indices[: min(k, len(ranked_indices))]
            if any(sample.views[idx].is_positive for idx in top_indices):
                topk_hits[k] += 1

        results.append(
            {
                "sample_id": sample.sample_id,
                "category": sample.category,
                "query": sample.query,
                "selected_indices": seen_indices,
                "selected_image_paths": [sample.views[i].image_path for i in seen_indices],
                "ranked_observed_indices": ranked_indices,
                "ranked_observed_image_paths": [sample.views[i].image_path for i in ranked_indices],
                "final_index": final_global_idx,
                "final_image_path": final_view.image_path,
                "gt_image_paths": [v.image_path for v in sample.views if v.is_positive],
                "correct": is_correct,
                "steps": len(seen_indices),
                "reranker_mode": reranker_mode,
                "scores": [
                    {
                        "index": idx,
                        "image_path": sample.views[idx].image_path,
                        "score": score,
                        "is_positive": sample.views[idx].is_positive,
                    }
                    for idx, score in scored
                ],
            }
        )

    n = max(1, len(results))
    metrics = {
        "accuracy": success / n,
        "avg_steps": total_steps / n,
        "num_samples": n,
        "reranker_mode": reranker_mode,
        "results": results,
    }
    for k in topk:
        metrics[f"top_{k}_accuracy"] = topk_hits[k] / n
    return metrics


# ============================================================
# Utility functions
# ============================================================

def print_summary(name: str, metrics: Dict[str, Any]) -> None:
    print("=" * 90)
    print(f"Method: {name}")
    if metrics.get("num_trials", 1) > 1:
        print(f"Trials: {metrics['num_trials']} (random initial view per sample)")
    print(f"Accuracy: {metrics['accuracy']:.4f}", end="")
    if metrics.get("accuracy_std") is not None:
        print(f" +/- {metrics['accuracy_std']:.4f}")
    else:
        print()
    for key, value in metrics.items():
        if key.startswith("top_") and key.endswith("_accuracy") and not key.endswith("_accuracy_std"):
            std_key = f"{key}_std"
            if metrics.get(std_key) is not None:
                print(f"{key}: {value:.4f} +/- {metrics[std_key]:.4f}")
            else:
                print(f"{key}: {value:.4f}")
    if metrics.get("avg_steps") is not None:
        line = f"Average steps: {metrics['avg_steps']:.2f}"
        if metrics.get("avg_steps_std") is not None:
            line += f" +/- {metrics['avg_steps_std']:.2f}"
        print(line)
    if metrics.get("early_stop_rate") is not None:
        line = f"Early stop rate: {metrics['early_stop_rate']:.4f}"
        if metrics.get("early_stop_rate_std") is not None:
            line += f" +/- {metrics['early_stop_rate_std']:.4f}"
        print(line)
    print("=" * 90)


def save_results(path: str, metrics: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


# ============================================================
# Visualization and reporting
# ============================================================

def _safe_open_image(path: str, size: Tuple[int, int] = (224, 224)) -> Image.Image:
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        img = Image.new("RGB", size, color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "Missing image", fill=(0, 0, 0))
        return img
    return img.resize(size)


def make_contact_sheet(
    result: Dict[str, Any],
    out_path: str,
    thumb_size: Tuple[int, int] = (224, 224),
) -> None:
    selected_paths = result["selected_image_paths"]
    gt_paths = set(result["gt_image_paths"])
    instruction = result["query"]
    sample_id = result["sample_id"]

    cols = max(1, len(selected_paths))
    width = cols * thumb_size[0]
    header_h = 100
    footer_h = 40
    height = header_h + thumb_size[1] + footer_h

    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    wrapped = textwrap.fill(f"{sample_id}: {instruction}", width=70)
    draw.text((10, 10), wrapped, fill=(0, 0, 0))

    for i, path in enumerate(selected_paths):
        img = _safe_open_image(path, thumb_size)
        x = i * thumb_size[0]
        y = header_h
        canvas.paste(img, (x, y))

        is_gt = path in gt_paths
        color = (0, 180, 0) if is_gt else (200, 0, 0)
        draw.rectangle([x + 2, y + 2, x + thumb_size[0] - 3, y + thumb_size[1] - 3], outline=color, width=4)
        draw.text((x + 8, y + thumb_size[1] + 8), f"step {i+1}", fill=(0, 0, 0))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)


def save_visualizations(metrics: Dict[str, Any], out_dir: str, limit: int = 20) -> None:
    vis_dir = os.path.join(out_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    for i, result in enumerate(metrics.get("results", [])[:limit]):
        out_path = os.path.join(vis_dir, f"{i:03d}_{result['sample_id']}.png")
        make_contact_sheet(result, out_path)


def save_budget_curve_plot(budget_to_metrics: Dict[int, Dict[str, Any]], out_path: str) -> None:
    budgets = sorted(budget_to_metrics.keys())
    accuracies = [budget_to_metrics[b]["accuracy"] for b in budgets]
    avg_steps = [budget_to_metrics[b]["avg_steps"] for b in budgets]

    plt.figure(figsize=(7, 5))
    plt.plot(budgets, accuracies, marker="o", label="Top-1 accuracy")
    plt.plot(budgets, avg_steps, marker="s", label="Average steps")
    plt.xlabel("Budget")
    plt.ylabel("Value")
    plt.title("Budget sweep")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _per_category_accuracy(metrics: Dict[str, Any], category: str) -> float:
    per_cat = metrics.get("per_category")
    if per_cat and category in per_cat:
        return float(per_cat[category]["accuracy"])

    items = [r for r in metrics.get("results", []) if r.get("category") == category]
    if not items:
        return 0.0
    return sum(int(x["correct"]) for x in items) / len(items)


def collect_eval_categories(*metrics_list: Dict[str, Any]) -> List[str]:
    categories = set()
    for metrics in metrics_list:
        for item in metrics.get("results", []):
            if item.get("category"):
                categories.add(item["category"])
        for category in metrics.get("per_category", {}):
            categories.add(category)
    return sorted(categories)


def _method_accuracy_curve(
    budgets: List[int],
    budget_to_metrics: Optional[Dict[int, Dict[str, Any]]],
    flat_metrics: Optional[Dict[str, Any]],
    category: Optional[str],
    value_key: str = "accuracy",
) -> Optional[List[float]]:
    if budget_to_metrics is not None:
        if category is None:
            return [float(budget_to_metrics[b].get(value_key, 0.0)) for b in budgets]
        if value_key != "accuracy":
            return None
        return [_per_category_accuracy(budget_to_metrics[b], category) for b in budgets]
    if flat_metrics is not None:
        if category is None:
            value = float(flat_metrics.get(value_key, 0.0))
        else:
            if value_key != "accuracy":
                return None
            value = _per_category_accuracy(flat_metrics, category)
        return [value] * len(budgets)
    return None


def _plot_accuracy_with_std(
    budgets: List[int],
    means: Optional[List[float]],
    stds: Optional[List[float]],
    **plot_kwargs,
) -> None:
    if means is None:
        return
    if stds is not None and any(float(s) > 0 for s in stds):
        plt.errorbar(
            budgets,
            means,
            yerr=stds,
            capsize=3,
            elinewidth=1.2,
            markeredgewidth=1.0,
            **plot_kwargs,
        )
    else:
        plt.plot(budgets, means, **plot_kwargs)


def save_method_comparison_plot(
    budget_to_metrics: Dict[int, Dict[str, Any]],
    random_metrics: Dict[str, Any],
    exhaustive_metrics: Dict[str, Any],
    out_path: str,
    budget_to_reranker_metrics: Optional[Dict[int, Dict[str, Any]]] = None,
    budget_to_trained_active_clip: Optional[Dict[int, Dict[str, Any]]] = None,
    budget_to_trained_active_learned: Optional[Dict[int, Dict[str, Any]]] = None,
    budget_to_random_metrics: Optional[Dict[int, Dict[str, Any]]] = None,
    category: Optional[str] = None,
) -> None:
    budgets = sorted(budget_to_metrics.keys())
    show_std = category is None

    active_acc = _method_accuracy_curve(budgets, budget_to_metrics, None, category)
    active_std = (
        _method_accuracy_curve(budgets, budget_to_metrics, None, category, "accuracy_std")
        if show_std else None
    )
    random_acc = _method_accuracy_curve(
        budgets, budget_to_random_metrics, random_metrics, category
    )
    random_std = (
        _method_accuracy_curve(
            budgets, budget_to_random_metrics, random_metrics, category, "accuracy_std"
        )
        if show_std else None
    )
    exhaustive_acc = _method_accuracy_curve(budgets, None, exhaustive_metrics, category)
    exhaustive_std = (
        _method_accuracy_curve(budgets, None, exhaustive_metrics, category, "accuracy_std")
        if show_std else None
    )
    reranker_acc = _method_accuracy_curve(budgets, budget_to_reranker_metrics, None, category)
    reranker_std = (
        _method_accuracy_curve(
            budgets, budget_to_reranker_metrics, None, category, "accuracy_std"
        )
        if show_std else None
    )
    trained_active_clip_acc = _method_accuracy_curve(
        budgets, budget_to_trained_active_clip, None, category
    )
    trained_active_clip_std = (
        _method_accuracy_curve(
            budgets, budget_to_trained_active_clip, None, category, "accuracy_std"
        )
        if show_std else None
    )
    trained_active_learned_acc = _method_accuracy_curve(
        budgets, budget_to_trained_active_learned, None, category
    )
    trained_active_learned_std = (
        _method_accuracy_curve(
            budgets, budget_to_trained_active_learned, None, category, "accuracy_std"
        )
        if show_std else None
    )

    plt.figure(figsize=(10, 5))

    _plot_accuracy_with_std(
        budgets, active_acc, active_std, marker="o", linewidth=2, label="Active policy"
    )
    _plot_accuracy_with_std(
        budgets, random_acc, random_std, marker="o", linestyle="--", label="Random"
    )
    _plot_accuracy_with_std(
        budgets,
        exhaustive_acc,
        exhaustive_std,
        marker="o",
        linestyle="--",
        label="Exhaustive CLIP",
    )
    if reranker_acc is not None:
        _plot_accuracy_with_std(
            budgets,
            reranker_acc,
            reranker_std,
            marker="o",
            linestyle="--",
            label="Learned reranker",
        )
    if trained_active_clip_acc is not None:
        _plot_accuracy_with_std(
            budgets,
            trained_active_clip_acc,
            trained_active_clip_std,
            marker="o",
            linestyle="-.",
            linewidth=2,
            label="Trained active + CLIP",
        )
    if trained_active_learned_acc is not None:
        _plot_accuracy_with_std(
            budgets,
            trained_active_learned_acc,
            trained_active_learned_std,
            marker="o",
            linestyle="-.",
            linewidth=2,
            label="Trained active + learned reranker",
        )

    plt.xlabel("Observation budget")
    plt.ylabel("Accuracy")
    title = "Active View Selection vs Baselines"
    if category is not None:
        title += f" ({category})"
    elif show_std:
        n_trials = next(
            (
                budget_to_metrics[b].get("num_trials")
                for b in budgets
                if budget_to_metrics[b].get("num_trials")
            ),
            None,
        )
        if n_trials:
            title += f" (mean ± std over {n_trials} trials)"
    plt.title(title)
    plt.xticks(budgets)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_method_comparison_plots_by_category(
    out_dir: str,
    budget_to_metrics: Dict[int, Dict[str, Any]],
    random_metrics: Dict[str, Any],
    exhaustive_metrics: Dict[str, Any],
    budget_to_reranker_metrics: Optional[Dict[int, Dict[str, Any]]] = None,
    budget_to_trained_active_clip: Optional[Dict[int, Dict[str, Any]]] = None,
    budget_to_trained_active_learned: Optional[Dict[int, Dict[str, Any]]] = None,
    budget_to_random_metrics: Optional[Dict[int, Dict[str, Any]]] = None,
) -> List[str]:
    categories = collect_eval_categories(
        random_metrics,
        exhaustive_metrics,
        *budget_to_metrics.values(),
        *(budget_to_random_metrics or {}).values(),
        *(budget_to_reranker_metrics or {}).values(),
        *(budget_to_trained_active_clip or {}).values(),
        *(budget_to_trained_active_learned or {}).values(),
    )

    category_dir = os.path.join(out_dir, "per_category")
    saved_paths = []
    for category in categories:
        out_path = os.path.join(category_dir, f"method_comparison_vs_budget_{category}.png")
        save_method_comparison_plot(
            budget_to_metrics=budget_to_metrics,
            random_metrics=random_metrics,
            exhaustive_metrics=exhaustive_metrics,
            out_path=out_path,
            budget_to_reranker_metrics=budget_to_reranker_metrics,
            budget_to_trained_active_clip=budget_to_trained_active_clip,
            budget_to_trained_active_learned=budget_to_trained_active_learned,
            budget_to_random_metrics=budget_to_random_metrics,
            category=category,
        )
        saved_paths.append(out_path)
    return saved_paths


def save_topk_vs_budget_plot(
    budget_to_metrics: Dict[int, Dict[str, Any]],
    out_path: str,
    title: str,
    topk: Tuple[int, ...] = (1, 2, 3),
) -> None:
    budgets = sorted(budget_to_metrics.keys())
    markers = ["o", "s", "^", "D", "v"]

    plt.figure(figsize=(8, 5))
    for i, k in enumerate(topk):
        key = f"top_{k}_accuracy"
        values = [
            float(budget_to_metrics[b].get(key, budget_to_metrics[b].get("accuracy", 0.0)))
            for b in budgets
        ]
        plt.plot(
            budgets,
            values,
            marker=markers[i % len(markers)],
            linewidth=2,
            label=f"Top-{k}",
        )

    plt.xlabel("Observation budget")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.xticks(budgets)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_topk_vs_budget_plots(
    out_dir: str,
    budget_to_metrics: Dict[int, Dict[str, Any]],
    topk: Tuple[int, ...],
    budget_to_trained_active_clip: Optional[Dict[int, Dict[str, Any]]] = None,
    budget_to_trained_active_learned: Optional[Dict[int, Dict[str, Any]]] = None,
) -> List[str]:
    series: List[Tuple[str, Dict[int, Dict[str, Any]]]] = [
        ("Active policy", budget_to_metrics),
    ]
    if budget_to_trained_active_clip:
        series.append(("Trained active + CLIP", budget_to_trained_active_clip))
    if budget_to_trained_active_learned:
        series.append(("Trained active + learned reranker", budget_to_trained_active_learned))

    saved_paths = []
    if len(series) == 1:
        out_path = os.path.join(out_dir, "topk_vs_budget.png")
        save_topk_vs_budget_plot(
            budget_to_metrics=budget_to_metrics,
            out_path=out_path,
            title="Top-k Accuracy vs Observation Budget (Active policy)",
            topk=topk,
        )
        saved_paths.append(out_path)
        return saved_paths

    fig, axes = plt.subplots(1, len(series), figsize=(6 * len(series), 5), squeeze=False)
    budgets = sorted(budget_to_metrics.keys())
    markers = ["o", "s", "^", "D", "v"]

    for ax, (name, metrics_by_budget) in zip(axes[0], series):
        for i, k in enumerate(topk):
            key = f"top_{k}_accuracy"
            values = [
                float(metrics_by_budget[b].get(key, metrics_by_budget[b].get("accuracy", 0.0)))
                for b in budgets
            ]
            ax.plot(
                budgets,
                values,
                marker=markers[i % len(markers)],
                linewidth=2,
                label=f"Top-{k}",
            )
        ax.set_xlabel("Observation budget")
        ax.set_ylabel("Accuracy")
        ax.set_title(name)
        ax.set_xticks(budgets)
        ax.set_ylim(0, 1.0)
        ax.legend()

    fig.suptitle("Top-k Accuracy vs Observation Budget", y=1.02)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "topk_vs_budget.png")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(out_path)
    return saved_paths


def save_method_bar_plot(
    budget_to_metrics: Dict[int, Dict[str, Any]],
    random_metrics: Dict[str, Any],
    exhaustive_metrics: Dict[str, Any],
    selected_budget: int,
    out_path: str,
    budget_to_reranker_metrics: Optional[Dict[int, Dict[str, Any]]] = None,
    budget_to_trained_active_clip: Optional[Dict[int, Dict[str, Any]]] = None,
    budget_to_trained_active_learned: Optional[Dict[int, Dict[str, Any]]] = None,
) -> None:
    active_acc = budget_to_metrics[selected_budget]["accuracy"]

    methods = ["Active policy", "Random", "Exhaustive CLIP"]
    accuracies = [
        active_acc,
        random_metrics["accuracy"],
        exhaustive_metrics["accuracy"],
    ]
    if budget_to_reranker_metrics is not None and selected_budget in budget_to_reranker_metrics:
        methods.append("Learned reranker")
        accuracies.append(budget_to_reranker_metrics[selected_budget]["accuracy"])
    if budget_to_trained_active_clip is not None and selected_budget in budget_to_trained_active_clip:
        methods.append("Trained active + CLIP")
        accuracies.append(budget_to_trained_active_clip[selected_budget]["accuracy"])
    if budget_to_trained_active_learned is not None and selected_budget in budget_to_trained_active_learned:
        methods.append("Trained active + learned reranker")
        accuracies.append(budget_to_trained_active_learned[selected_budget]["accuracy"])

    plt.figure(figsize=(10, 5))
    plt.bar(methods, accuracies)
    plt.ylabel("Accuracy")
    plt.title(f"Method Comparison at Budget = {selected_budget}")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def compute_per_category(metrics: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in metrics.get("results", []):
        grouped.setdefault(item["category"], []).append(item)

    out = {}
    for category, items in grouped.items():
        n = len(items)
        acc = sum(int(x["correct"]) for x in items) / max(1, n)
        avg_steps = sum(x["steps"] for x in items) / max(1, n)
        out[category] = {
            "num_samples": n,
            "accuracy": acc,
            "avg_steps": avg_steps,
        }
    return out


# ============================================================
# Main
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_json", type=str, required=True)
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--initial_index", type=int, default=0, help="Used only when --no_random_initial is set")
    parser.add_argument("--no_random_initial", action="store_true", help="Start every sample from --initial_index instead of a random view")
    parser.add_argument("--num_trials", type=int, default=10, help="Number of repeated evaluations with random initial views")
    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="openai")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--w_relevance", type=float, default=1.0)
    parser.add_argument("--w_novelty", type=float, default=0.5)

    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--budget_sweep", type=str, default="1,2,3,4,5")
    parser.add_argument("--topk", type=str, default="1,2,3")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save_visualizations", action="store_true")
    parser.add_argument("--vis_limit", type=int, default=20)
    parser.add_argument("--stop_threshold", type=float, default=None)

    add_split_args(parser)
    parser.add_argument("--reranker_hidden_dim", type=int, default=256)
    parser.add_argument("--reranker_epochs", type=int, default=50)
    parser.add_argument("--reranker_batch_size", type=int, default=64)
    parser.add_argument("--reranker_lr", type=float, default=1e-3)
    parser.add_argument("--reranker_patience", type=int, default=10)
    parser.add_argument("--skip_reranker", action="store_true")
    parser.add_argument(
        "--load_models_from",
        type=str,
        default=None,
        help="Directory with reranker.pt and view_selector.pt; skips training when set",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    dataset = ActiveViewDataset(args.labels_json, group_field=args.group_by)
    scorer = CLIPScorer(model_name=args.clip_model, pretrained=args.clip_pretrained)
    feature_cache = FeatureCache(scorer)
    device = torch.device(scorer.device)

    train_dataset, val_dataset, eval_dataset, split_desc = split_dataset(dataset, args)

    print(
        f"Split ({split_desc}): train={len(train_dataset)} val={len(val_dataset)} eval={len(eval_dataset)}"
    )

    reranker = None
    reranker_train_info: Dict[str, Any] = {}
    view_selector = None
    selector_train_info: Dict[str, Any] = {}
    if args.load_models_from and not args.skip_reranker:
        reranker_path = os.path.join(args.load_models_from, "reranker.pt")
        selector_path = os.path.join(args.load_models_from, "view_selector.pt")
        reranker_ckpt = torch.load(reranker_path, map_location=device, weights_only=False)
        selector_ckpt = torch.load(selector_path, map_location=device, weights_only=False)
        reranker_train_info = reranker_ckpt["train_info"]
        selector_train_info = selector_ckpt["train_info"]
        reranker = ViewReranker(
            feat_dim=reranker_train_info["feat_dim"],
            hidden_dim=reranker_train_info["hidden_dim"],
        )
        view_selector = ViewReranker(
            feat_dim=selector_train_info["feat_dim"],
            hidden_dim=selector_train_info["hidden_dim"],
        )
        reranker.load_state_dict(reranker_ckpt["state_dict"])
        view_selector.load_state_dict(selector_ckpt["state_dict"])
        reranker.to(device).eval()
        view_selector.to(device).eval()
        print(f"Loaded reranker and view selector from {args.load_models_from}")
    elif not args.skip_reranker:
        print(f"Training learned reranker ({split_desc})")
        reranker, reranker_train_info = train_view_reranker(
            train_samples=train_dataset.samples,
            val_samples=val_dataset.samples,
            feature_cache=feature_cache,
            device=device,
            hidden_dim=args.reranker_hidden_dim,
            epochs=args.reranker_epochs,
            batch_size=args.reranker_batch_size,
            lr=args.reranker_lr,
            patience=args.reranker_patience,
        )
        print(f"Reranker best val accuracy: {reranker_train_info['best_val_accuracy']:.4f}")
        torch.save(
            {
                "state_dict": reranker.state_dict(),
                "train_info": reranker_train_info,
            },
            os.path.join(args.out_dir, "reranker.pt"),
        )
        save_results(
            os.path.join(args.out_dir, "reranker_train_info.json"),
            reranker_train_info,
        )

        print(f"Training active view selector ({split_desc})")
        view_selector, selector_train_info = train_view_selector(
            train_samples=train_dataset.samples,
            val_samples=val_dataset.samples,
            feature_cache=feature_cache,
            device=device,
            hidden_dim=args.reranker_hidden_dim,
            epochs=args.reranker_epochs,
            batch_size=args.reranker_batch_size,
            lr=args.reranker_lr,
            patience=args.reranker_patience,
        )
        print(f"Active selector best val accuracy: {selector_train_info['best_val_accuracy']:.4f}")
        torch.save(
            {
                "state_dict": view_selector.state_dict(),
                "train_info": selector_train_info,
            },
            os.path.join(args.out_dir, "view_selector.pt"),
        )
        save_results(
            os.path.join(args.out_dir, "view_selector_train_info.json"),
            selector_train_info,
        )

    topk = tuple(int(x.strip()) for x in args.topk.split(",") if x.strip())
    budgets = [int(x.strip()) for x in args.budget_sweep.split(",") if x.strip()]
    random_initial = not args.no_random_initial

    active_policy = ActiveSelectionPolicy(
        feature_cache=feature_cache,
        w_relevance=args.w_relevance,
        w_novelty=args.w_novelty,
        stop_threshold=args.stop_threshold,
    )

    budget_to_metrics = {}
    for budget in budgets:
        active_metrics = run_evaluation_trials(
            evaluate_sequential_policy,
            num_trials=args.num_trials,
            base_seed=args.seed,
            dataset=eval_dataset,
            feature_cache=feature_cache,
            active_policy=active_policy,
            budget=budget,
            initial_index=args.initial_index,
            random_initial=random_initial,
            topk=topk,
            verbose=args.verbose,
        )
        active_metrics["per_category"] = compute_per_category(active_metrics)
        budget_to_metrics[budget] = active_metrics
        print_summary(f"active_policy_budget_{budget}", active_metrics)
        save_results(os.path.join(args.out_dir, f"active_policy_budget_{budget}.json"), active_metrics)

        if args.save_visualizations:
            save_visualizations(active_metrics, os.path.join(args.out_dir, f"budget_{budget}"), limit=args.vis_limit)

    budget_to_random_metrics: Dict[int, Dict[str, Any]] = {}
    for budget in budgets:
        random_budget_metrics = run_evaluation_trials(
            evaluate_baseline_random,
            num_trials=args.num_trials,
            base_seed=args.seed,
            dataset=eval_dataset,
            feature_cache=feature_cache,
            budget=budget,
            initial_index=args.initial_index,
            random_initial=random_initial,
        )
        random_budget_metrics["per_category"] = compute_per_category(random_budget_metrics)
        budget_to_random_metrics[budget] = random_budget_metrics
        print_summary(f"random_budget_{budget}", random_budget_metrics)
        save_results(
            os.path.join(args.out_dir, f"random_budget_{budget}.json"),
            random_budget_metrics,
        )
    random_metrics = budget_to_random_metrics.get(
        args.budget, next(iter(budget_to_random_metrics.values()))
    )
    save_results(os.path.join(args.out_dir, "random.json"), random_metrics)

    exhaustive_metrics = run_evaluation_trials(
        evaluate_exhaustive_clip,
        num_trials=1,
        dataset=eval_dataset,
        feature_cache=feature_cache,
    )
    print_summary("exhaustive_clip", exhaustive_metrics)
    exhaustive_metrics["per_category"] = compute_per_category(exhaustive_metrics)
    save_results(os.path.join(args.out_dir, "exhaustive_clip.json"), exhaustive_metrics)

    reranker_metrics = None
    budget_to_reranker_metrics: Dict[int, Dict[str, Any]] = {}
    if reranker is not None:
        for budget in budgets:
            reranker_metrics = run_evaluation_trials(
                evaluate_baseline_learned_reranker,
                num_trials=args.num_trials,
                base_seed=args.seed,
                dataset=eval_dataset,
                feature_cache=feature_cache,
                reranker=reranker,
                device=device,
                budget=budget,
                initial_index=args.initial_index,
                random_initial=random_initial,
            )
            reranker_metrics["per_category"] = compute_per_category(reranker_metrics)
            budget_to_reranker_metrics[budget] = reranker_metrics
            print_summary(f"learned_reranker_budget_{budget}", reranker_metrics)
            save_results(
                os.path.join(args.out_dir, f"learned_reranker_budget_{budget}.json"),
                reranker_metrics,
            )

        reranker_sweep_summary = {
            str(b): budget_to_reranker_metrics[b]
            for b in sorted(budget_to_reranker_metrics.keys())
        }
        reranker_sweep_summary["train_info"] = reranker_train_info
        save_results(
            os.path.join(args.out_dir, "learned_reranker_budget_sweep.json"),
            reranker_sweep_summary,
        )

    budget_to_trained_active_clip: Dict[int, Dict[str, Any]] = {}
    budget_to_trained_active_learned: Dict[int, Dict[str, Any]] = {}
    if view_selector is not None:
        for budget in budgets:
            trained_active_clip_metrics = run_evaluation_trials(
                evaluate_trained_active_policy,
                num_trials=args.num_trials,
                base_seed=args.seed,
                dataset=eval_dataset,
                feature_cache=feature_cache,
                selector=view_selector,
                device=device,
                budget=budget,
                initial_index=args.initial_index,
                random_initial=random_initial,
                reranker_mode="clip",
                topk=topk,
            )
            trained_active_clip_metrics["per_category"] = compute_per_category(trained_active_clip_metrics)
            budget_to_trained_active_clip[budget] = trained_active_clip_metrics
            print_summary(f"trained_active_clip_budget_{budget}", trained_active_clip_metrics)
            save_results(
                os.path.join(args.out_dir, f"trained_active_clip_budget_{budget}.json"),
                trained_active_clip_metrics,
            )

        trained_active_clip_sweep = {
            str(b): budget_to_trained_active_clip[b]
            for b in sorted(budget_to_trained_active_clip.keys())
        }
        trained_active_clip_sweep["selector_train_info"] = selector_train_info
        save_results(
            os.path.join(args.out_dir, "trained_active_clip_budget_sweep.json"),
            trained_active_clip_sweep,
        )

        if reranker is not None:
            for budget in budgets:
                trained_active_learned_metrics = run_evaluation_trials(
                    evaluate_trained_active_policy,
                    num_trials=args.num_trials,
                    base_seed=args.seed,
                    dataset=eval_dataset,
                    feature_cache=feature_cache,
                    selector=view_selector,
                    device=device,
                    budget=budget,
                    initial_index=args.initial_index,
                    random_initial=random_initial,
                    reranker_mode="learned",
                    reranker=reranker,
                    topk=topk,
                )
                trained_active_learned_metrics["per_category"] = compute_per_category(
                    trained_active_learned_metrics
                )
                budget_to_trained_active_learned[budget] = trained_active_learned_metrics
                print_summary(
                    f"trained_active_learned_reranker_budget_{budget}",
                    trained_active_learned_metrics,
                )
                save_results(
                    os.path.join(
                        args.out_dir,
                        f"trained_active_learned_reranker_budget_{budget}.json",
                    ),
                    trained_active_learned_metrics,
                )

            trained_active_learned_sweep = {
                str(b): budget_to_trained_active_learned[b]
                for b in sorted(budget_to_trained_active_learned.keys())
            }
            trained_active_learned_sweep["selector_train_info"] = selector_train_info
            trained_active_learned_sweep["reranker_train_info"] = reranker_train_info
            save_results(
                os.path.join(args.out_dir, "trained_active_learned_reranker_budget_sweep.json"),
                trained_active_learned_sweep,
            )

    save_budget_curve_plot(
        budget_to_metrics,
        os.path.join(args.out_dir, "budget_sweep.png"),
    )

    save_method_comparison_plot(
        budget_to_metrics,
        random_metrics,
        exhaustive_metrics,
        os.path.join(args.out_dir, "method_comparison_vs_budget.png"),
        budget_to_reranker_metrics=budget_to_reranker_metrics or None,
        budget_to_trained_active_clip=budget_to_trained_active_clip or None,
        budget_to_trained_active_learned=budget_to_trained_active_learned or None,
        budget_to_random_metrics=budget_to_random_metrics or None,
    )

    category_plot_paths = save_method_comparison_plots_by_category(
        out_dir=args.out_dir,
        budget_to_metrics=budget_to_metrics,
        random_metrics=random_metrics,
        exhaustive_metrics=exhaustive_metrics,
        budget_to_reranker_metrics=budget_to_reranker_metrics or None,
        budget_to_trained_active_clip=budget_to_trained_active_clip or None,
        budget_to_trained_active_learned=budget_to_trained_active_learned or None,
        budget_to_random_metrics=budget_to_random_metrics or None,
    )

    topk_plot_paths = save_topk_vs_budget_plots(
        out_dir=args.out_dir,
        budget_to_metrics=budget_to_metrics,
        topk=topk,
        budget_to_trained_active_clip=budget_to_trained_active_clip or None,
        budget_to_trained_active_learned=budget_to_trained_active_learned or None,
    )

    if args.budget in budget_to_metrics:
        save_method_bar_plot(
            budget_to_metrics,
            random_metrics,
            exhaustive_metrics,
            selected_budget=args.budget,
            out_path=os.path.join(args.out_dir, f"method_comparison_budget_{args.budget}.png"),
            budget_to_reranker_metrics=budget_to_reranker_metrics or None,
            budget_to_trained_active_clip=budget_to_trained_active_clip or None,
            budget_to_trained_active_learned=budget_to_trained_active_learned or None,
        )

    save_results(
        os.path.join(args.out_dir, "budget_sweep_summary.json"),
        {str(k): v for k, v in budget_to_metrics.items()},
    )

    print(f"Saved results to: {args.out_dir}")
    if category_plot_paths:
        print(f"Saved {len(category_plot_paths)} per-category comparison plots to: "
              f"{os.path.join(args.out_dir, 'per_category')}")
    if topk_plot_paths:
        print(f"Saved top-k vs budget plot(s): {', '.join(topk_plot_paths)}")


if __name__ == "__main__":
    main()
