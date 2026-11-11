import json
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


@dataclass
class ViewScore:
    image_path: str
    score: float
    rank: int = -1


class CLIPMultiViewEvaluator:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = None,
        use_fast_processor: bool = True,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_name = model_name

        self.processor = CLIPProcessor.from_pretrained(
            model_name,
            use_fast=use_fast_processor
        )
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _load_images(self, image_paths: List[str]) -> List[Image.Image]:
        images = []
        for path in image_paths:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Image not found: {path}")
            img = Image.open(path).convert("RGB")
            images.append(img)
        return images

    @torch.no_grad()
    def score_views(self, image_paths: List[str], instruction: str) -> List[ViewScore]:
        if not image_paths:
            raise ValueError("image_paths is empty")
        if not instruction.strip():
            raise ValueError("instruction is empty")

        images = self._load_images(image_paths)

        inputs = self.processor(
            text=[instruction],
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        outputs = self.model(**inputs)

        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        scores_tensor = (image_embeds @ text_embeds.T).squeeze(-1)

        results = [
            ViewScore(image_path=path, score=float(score.item()))
            for path, score in zip(image_paths, scores_tensor)
        ]

        ranked = sorted(results, key=lambda x: x.score, reverse=True)
        for i, item in enumerate(ranked, start=1):
            item.rank = i

        rank_map = {item.image_path: item.rank for item in ranked}
        for item in results:
            item.rank = rank_map[item.image_path]

        return results

    def exhaustive_best_view(
        self,
        image_paths: List[str],
        instruction: str,
    ) -> Tuple[str, List[ViewScore]]:
        scores = self.score_views(image_paths, instruction)
        best = max(scores, key=lambda x: x.score)
        return best.image_path, scores

    def fixed_order_first_hit(
        self,
        image_paths: List[str],
        instruction: str,
        threshold: float = 0.21,
    ) -> Dict:
        scores = self.score_views(image_paths, instruction)

        checked = []
        for i, item in enumerate(scores, start=1):
            checked.append(item)
            if item.score >= threshold:
                return {
                    "selected_view": item.image_path,
                    "steps": i,
                    "hit": True,
                    "checked_scores": checked,
                }

        best = max(scores, key=lambda x: x.score)
        return {
            "selected_view": best.image_path,
            "steps": len(scores),
            "hit": False,
            "checked_scores": checked,
        }


def load_annotations(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("annotations.json must contain a list")

    return data


def validate_sample(sample: Dict) -> None:
    required_keys = ["id", "instruction", "views", "gt_views"]
    for key in required_keys:
        if key not in sample:
            raise ValueError(f"Missing key '{key}' in sample: {sample}")

    if not isinstance(sample["views"], list) or len(sample["views"]) == 0:
        raise ValueError(f"Sample {sample['id']} has empty views")

    if not isinstance(sample["gt_views"], list):
        raise ValueError(f"Sample {sample['id']} gt_views must be a list")


def evaluate_exhaustive(
    evaluator: CLIPMultiViewEvaluator,
    dataset: List[Dict],
) -> Dict:
    top1_correct = 0
    top2_correct = 0
    total_steps = 0
    details = []

    for sample in dataset:
        validate_sample(sample)

        pred_view, scores = evaluator.exhaustive_best_view(
            image_paths=sample["views"],
            instruction=sample["instruction"],
        )

        ranked_scores = sorted(scores, key=lambda x: x.rank)
        top1_view = ranked_scores[0].image_path
        top2_views = [s.image_path for s in ranked_scores[:2]]

        is_top1_correct = top1_view in sample["gt_views"]
        is_top2_correct = any(v in sample["gt_views"] for v in top2_views)

        if is_top1_correct:
            top1_correct += 1
        if is_top2_correct:
            top2_correct += 1

        steps = len(sample["views"])
        total_steps += steps

        details.append({
            "id": sample["id"],
            "instruction": sample["instruction"],
            "pred_view": pred_view,
            "top1_view": top1_view,
            "top2_views": top2_views,
            "gt_views": sample["gt_views"],
            "correct": is_top1_correct,
            "top2_correct": is_top2_correct,
            "steps": steps,
            "scores": ranked_scores,
        })

    total_samples = len(dataset)
    top1_accuracy = top1_correct / total_samples if total_samples > 0 else 0.0
    top2_accuracy = top2_correct / total_samples if total_samples > 0 else 0.0
    avg_steps = total_steps / total_samples if total_samples > 0 else 0.0

    return {
        "method": "exhaustive_best_view",
        "accuracy": top1_accuracy,      # 保持原字段名兼容
        "top1_accuracy": top1_accuracy,
        "top2_accuracy": top2_accuracy,
        "avg_steps": avg_steps,
        "details": details,
    }


def evaluate_fixed_order(
    evaluator: CLIPMultiViewEvaluator,
    dataset: List[Dict],
    threshold: float = 0.21,
) -> Dict:
    correct = 0
    total_steps = 0
    details = []

    for sample in dataset:
        validate_sample(sample)

        result = evaluator.fixed_order_first_hit(
            image_paths=sample["views"],
            instruction=sample["instruction"],
            threshold=threshold,
        )

        pred_view = result["selected_view"]
        steps = result["steps"]
        is_correct = pred_view in sample["gt_views"]

        if is_correct:
            correct += 1
        total_steps += steps

        details.append({
            "id": sample["id"],
            "instruction": sample["instruction"],
            "pred_view": pred_view,
            "gt_views": sample["gt_views"],
            "correct": is_correct,
            "steps": steps,
            "hit": result["hit"],
            "scores": result["checked_scores"],
        })

    total_samples = len(dataset)
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    avg_steps = total_steps / total_samples if total_samples > 0 else 0.0

    return {
        "method": "fixed_order_first_hit",
        "accuracy": accuracy,
        "avg_steps": avg_steps,
        "details": details,
    }


def print_summary(results: Dict) -> None:
    print("\n" + "=" * 90)
    print(f"Method: {results['method']}")
    print(f"Accuracy: {results['accuracy']:.4f}")

    if "top1_accuracy" in results:
        print(f"Top-1 Accuracy: {results['top1_accuracy']:.4f}")
    if "top2_accuracy" in results:
        print(f"Top-2 Accuracy: {results['top2_accuracy']:.4f}")

    print(f"Average steps: {results['avg_steps']:.2f}")
    print("=" * 90)

    for item in results["details"]:
        print(f"\nSample: {item['id']}")
        print(f"Instruction: {item['instruction']}")
        print(f"Predicted:   {item['pred_view']}")
        print(f"GT Views:    {item['gt_views']}")
        print(f"Correct:     {item['correct']}")

        if "top2_correct" in item:
            print(f"Top-2 Correct: {item['top2_correct']}")
            print(f"Top-2 Views:   {item['top2_views']}")

        print(f"Steps:       {item['steps']}")

        if "hit" in item:
            print(f"Early stop:  {item['hit']}")

        print("Scores:")
        for s in item["scores"]:
            print(f"  rank={s.rank:<2} score={s.score:.4f} path={s.image_path}")


def save_results_json(results: Dict, save_path: str) -> None:
    serializable = {
        "method": results["method"],
        "accuracy": results["accuracy"],
        "avg_steps": results["avg_steps"],
        "details": []
    }

    if "top1_accuracy" in results:
        serializable["top1_accuracy"] = results["top1_accuracy"]
    if "top2_accuracy" in results:
        serializable["top2_accuracy"] = results["top2_accuracy"]

    for item in results["details"]:
        item_copy = item.copy()
        item_copy["scores"] = [
            {
                "image_path": s.image_path,
                "score": s.score,
                "rank": s.rank,
            }
            for s in item["scores"]
        ]
        serializable["details"].append(item_copy)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def main():
    annotation_path = r"E:\CO3D-subset-teddybear\annotations.json"
    fixed_threshold = 0.21

    dataset = load_annotations(annotation_path)
    print(f"Loaded {len(dataset)} samples from {annotation_path}")

    evaluator = CLIPMultiViewEvaluator()
    print(f"Using device: {evaluator.device}")
    print(f"Model: {evaluator.model_name}")

    exhaustive_results = evaluate_exhaustive(evaluator, dataset)
    print_summary(exhaustive_results)
    save_results_json(
        exhaustive_results,
        r"E:\CO3D-subset-teddybear\results_exhaustive.json"
    )

    fixed_results = evaluate_fixed_order(
        evaluator,
        dataset,
        threshold=fixed_threshold,
    )
    print_summary(fixed_results)
    save_results_json(
        fixed_results,
        r"E:\CO3D-subset-teddybear\results_fixed_order.json"
    )

    print("\nSaved:")
    print(r"  E:\CO3D-subset-teddybear\results_exhaustive.json")
    print(r"  E:\CO3D-subset-teddybear\results_fixed_order.json")


if __name__ == "__main__":
    main()