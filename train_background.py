#!/usr/bin/env python3
"""Single-model training entrypoint built on the benchmark framework."""

from __future__ import annotations

import argparse
from pathlib import Path

from mad.benchmark import run_benchmark
from mad.utils import timestamp, write_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one detector model")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--model-id", type=str, default="single_run")
    parser.add_argument("--dataset-yaml", type=Path, default=Path("data/processed/yolo_dataset/dataset.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/single_model"))
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--workspace-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = {
        "study_name": f"single_model_{args.model_id}",
        "dataset_yaml": str(args.dataset_yaml),
        "output_dir": str(args.output_dir),
        "workspace_root": str(args.workspace_root) if args.workspace_root is not None else None,
        "device": args.device,
        "seed": args.seed,
        "seeds": [args.seed],
        "train": {
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "optimizer": "auto",
            "lr0": 0.01,
            "lrf": 0.01,
            "cos_lr": True,
            "weight_decay": 0.0005,
            "patience": max(20, args.epochs // 3),
            "close_mosaic": min(15, max(5, args.epochs // 8)),
            "cache": False,
            "amp": True,
            "deterministic": True,
        },
        "models": [{"id": args.model_id, "weights": args.model}],
    }

    cfg_path = Path("artifacts") / f"single_model_{timestamp()}.yaml"
    write_yaml(cfg_path, cfg)
    summary = run_benchmark(cfg_path)

    print("Single model training/evaluation completed")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
