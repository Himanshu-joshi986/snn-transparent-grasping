"""
evaluation/evaluate.py - Real Model Evaluation
"""

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.dta_snn import DTASNN
from training.dataset import EventDataset, get_val_augmentations
from utils.metrics import compute_metrics, MetricsTracker


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model(checkpoint: str, cfg: dict, device: torch.device) -> DTASNN:
    model = DTASNN(cfg["model"]).to(device)
    state = torch.load(checkpoint, map_location=device)
    # Support both raw state_dict and checkpoint dicts
    state_dict = state.get("model_state_dict", state.get("state_dict", state))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate(model, loader, device, threshold=0.5):
    tracker = MetricsTracker()
    total_time = 0.0

    with torch.no_grad():
        for batch in loader:
            events = batch["events"].to(device)   # (B, T, C, H, W)
            masks  = batch["mask"].to(device)     # (B, 1, H, W)

            # Model expects (T, B, C, H, W)
            x = events.permute(1, 0, 2, 3, 4).contiguous()

            t0 = time.perf_counter()
            preds = model(x)                      # (B, 1, H, W)
            total_time += time.perf_counter() - t0

            metrics = compute_metrics(preds, masks, threshold=threshold)
            tracker.update(metrics, count=events.size(0))

    results = tracker.mean()
    results["latency_ms"] = (total_time / len(loader.dataset)) * 1000
    results["num_samples"] = len(loader.dataset)
    return results


def main():
    parser = argparse.ArgumentParser(description="SNN-DTA Model Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--config", type=str, default="configs/dta.yaml")
    parser.add_argument("--split", type=str, default="real",
                        choices=["real", "synthetic"],
                        help="Which test split to evaluate on")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--output_format", nargs="+", default=["json", "csv"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    image_size = tuple(cfg["data"]["image_size"])
    T = cfg["data"]["time_steps"]

    if args.split == "real":
        rgb_dir   = cfg["data"]["real_rgb"]
        mask_dir  = cfg["data"]["real_masks"]
        event_dir = cfg["data"]["real_events"]
    else:
        rgb_dir   = cfg["data"]["synthetic_rgb"]
        mask_dir  = cfg["data"]["synthetic_masks"]
        event_dir = cfg["data"]["synthetic_events"]

    dataset = EventDataset(
        rgb_dir=rgb_dir,
        mask_dir=mask_dir,
        event_dir=event_dir,
        image_size=image_size,
        time_steps=T,
        split="test",
        transform=get_val_augmentations(image_size),
    )

    if len(dataset) == 0:
        print(f"No samples found in {rgb_dir}. Check your data paths.")
        return

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 2),
        pin_memory=device.type == "cuda",
    )

    print("=" * 60)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Split      : {args.split}  ({len(dataset)} samples)")
    print(f"Device     : {device}")
    print("=" * 60)

    model = load_model(args.checkpoint, cfg, device)
    results = evaluate(model, loader, device, threshold=args.threshold)

    print(f"IoU        : {results['iou']:.4f}")
    print(f"F1         : {results['f1']:.4f}")
    print(f"Precision  : {results['precision']:.4f}")
    print(f"Recall     : {results['recall']:.4f}")
    print(f"Accuracy   : {results['accuracy']:.4f}")
    print(f"Latency    : {results['latency_ms']:.2f} ms/sample")
    print(f"Samples    : {results['num_samples']}")
    print("=" * 60)

    results["checkpoint"] = args.checkpoint
    results["split"] = args.split

    if "json" in args.output_format:
        out = Path(args.output_dir) / "evaluation_metrics.json"
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON → {out}")

    if "csv" in args.output_format:
        out = Path(args.output_dir) / "evaluation_metrics.csv"
        with open(out, "w") as f:
            f.write("metric,value\n")
            for k, v in results.items():
                f.write(f"{k},{v}\n")
        print(f"Saved CSV  → {out}")


if __name__ == "__main__":
    main()
