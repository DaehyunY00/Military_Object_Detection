from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from mad.runtime import get_workspace_root, resolve_workspace_path
from mad.utils import configure_ultralytics_env, ensure_dir, resolve_device, timestamp, write_json, write_markdown


def _markdown_metrics_table(metrics: dict[str, Any]) -> str:
    rows = [
        "# Evaluation Summary",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    for key, value in metrics.items():
        rows.append(f"| {key} | {value} |")
    rows.append("")
    return "\n".join(rows)


def evaluate_model(
    model_path: Path,
    dataset_yaml: Path,
    split: str = "test",
    device: str = "auto",
    output_dir: Path | None = None,
) -> dict[str, Any]:
    configure_ultralytics_env()
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    resolved_device = resolve_device(device)
    val_kwargs: dict[str, Any] = {
        "data": str(dataset_yaml),
        "split": split,
        "device": resolved_device,
        "plots": True,
        "verbose": False,
    }
    if output_dir is not None:
        ensure_dir(output_dir)
        val_kwargs.update(
            {
                "project": str(output_dir.parent),
                "name": output_dir.name,
                "exist_ok": True,
            }
        )

    results = model.val(**val_kwargs)

    box = getattr(results, "box", None)
    metrics = {
        "model_path": str(model_path.resolve()),
        "dataset_yaml": str(dataset_yaml.resolve()),
        "split": split,
        "device": resolved_device,
        "map50": float(getattr(box, "map50", 0.0)) if box else None,
        "map50_95": float(getattr(box, "map", 0.0)) if box else None,
        "precision": float(getattr(box, "mp", 0.0)) if box else None,
        "recall": float(getattr(box, "mr", 0.0)) if box else None,
        "save_dir": str(Path(getattr(results, "save_dir", "")).resolve()) if getattr(results, "save_dir", None) else None,
    }

    if output_dir is not None:
        write_json(output_dir / "metrics.json", metrics)
        write_markdown(output_dir / "metrics.md", _markdown_metrics_table(metrics))

    return metrics


def predict_images(
    model_path: Path,
    source_path: Path,
    output_dir: Path,
    conf: float = 0.25,
    device: str = "auto",
) -> dict[str, Any]:
    configure_ultralytics_env()
    from ultralytics import YOLO

    ensure_dir(output_dir)
    model = YOLO(str(model_path))
    resolved_device = resolve_device(device)

    result = model.predict(
        source=str(source_path),
        conf=conf,
        device=resolved_device,
        save=True,
        project=str(output_dir),
        name="predictions",
        exist_ok=True,
    )

    rows = []
    for r in result:
        image_path = Path(getattr(r, "path", ""))
        for box in getattr(r, "boxes", []):
            rows.append(
                {
                    "image": str(image_path),
                    "class_id": int(box.cls.item()),
                    "confidence": float(box.conf.item()),
                    "x1": float(box.xyxy[0][0].item()),
                    "y1": float(box.xyxy[0][1].item()),
                    "x2": float(box.xyxy[0][2].item()),
                    "y2": float(box.xyxy[0][3].item()),
                }
            )

    csv_path = output_dir / "predictions" / "detections.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    summary = {
        "model_path": str(model_path.resolve()),
        "source_path": str(source_path.resolve()),
        "device": resolved_device,
        "detections_csv": str(csv_path.resolve()),
        "prediction_dir": str((output_dir / "predictions").resolve()),
        "detection_count": len(rows),
    }
    write_json(output_dir / "predictions" / "summary.json", summary)
    return summary


def _resolve_runtime_paths(
    *,
    workspace_root: Path | None,
    model_path: Path,
    dataset_yaml: Path | None = None,
    source_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Path | None]:
    resolved_workspace = get_workspace_root(workspace_root)
    return {
        "workspace_root": resolved_workspace,
        "model_path": resolve_workspace_path(model_path, resolved_workspace),
        "dataset_yaml": resolve_workspace_path(dataset_yaml, resolved_workspace) if dataset_yaml is not None else None,
        "source_path": resolve_workspace_path(source_path, resolved_workspace) if source_path is not None else None,
        "output_dir": resolve_workspace_path(output_dir, resolved_workspace) if output_dir is not None else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained detector or run prediction.")
    sub = parser.add_subparsers(dest="command", required=True)

    eval_parser = sub.add_parser("eval", help="Evaluate model on dataset split")
    eval_parser.add_argument("--model", type=Path, required=True)
    eval_parser.add_argument("--dataset-yaml", type=Path, default=Path("data/processed/yolo_dataset/dataset.yaml"))
    eval_parser.add_argument("--split", type=str, default="test")
    eval_parser.add_argument("--device", type=str, default="auto")
    eval_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    eval_parser.add_argument("--workspace-root", type=Path, default=None)

    pred_parser = sub.add_parser("predict", help="Predict images in a folder")
    pred_parser.add_argument("--model", type=Path, required=True)
    pred_parser.add_argument("--source", type=Path, default=None, help="Image file or directory")
    pred_parser.add_argument("--image-dir", type=Path, default=None, help="Deprecated alias for --source")
    pred_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/inference"))
    pred_parser.add_argument("--conf", type=float, default=0.25)
    pred_parser.add_argument("--device", type=str, default="auto")
    pred_parser.add_argument("--workspace-root", type=Path, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "eval":
        resolved = _resolve_runtime_paths(
            workspace_root=args.workspace_root,
            model_path=args.model,
            dataset_yaml=args.dataset_yaml,
            output_dir=args.output_dir,
        )
        output_dir = ensure_dir(Path(resolved["output_dir"]) / f"eval_{Path(resolved['model_path']).stem}_{timestamp()}")
        metrics = evaluate_model(
            Path(resolved["model_path"]),
            Path(resolved["dataset_yaml"]),
            split=args.split,
            device=args.device,
            output_dir=output_dir,
        )
        for key, value in metrics.items():
            print(f"{key}: {value}")
        print(f"metrics_dir: {output_dir}")
    elif args.command == "predict":
        source = args.source or args.image_dir
        if source is None:
            raise ValueError("Provide --source (or legacy --image-dir) for prediction.")
        resolved = _resolve_runtime_paths(
            workspace_root=args.workspace_root,
            model_path=args.model,
            source_path=source,
            output_dir=args.output_dir,
        )
        summary = predict_images(
            Path(resolved["model_path"]),
            Path(resolved["source_path"]),
            Path(resolved["output_dir"]),
            conf=args.conf,
            device=args.device,
        )
        print(f"detections_csv: {summary['detections_csv']}")
        print(f"prediction_dir: {summary['prediction_dir']}")


if __name__ == "__main__":
    main()
