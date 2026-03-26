from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from mad.runtime import collect_runtime_metadata, get_workspace_root, resolve_workspace_path
from mad.utils import ensure_dir, read_yaml, seed_everything, sorted_name_values, write_json, write_yaml

VALID_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
SPLIT_MAP = {
    "train": "train",
    "training": "train",
    "val": "val",
    "validation": "val",
    "valid": "val",
    "test": "test",
}


@dataclass
class DatasetBuildConfig:
    annotations_csv: Path
    images_dir: Path
    output_dir: Path
    force: bool = False
    symlink: bool = False
    workspace_root: Path | None = None
    max_images_per_split: int | None = None
    shuffle: bool = False
    seed: int = 42
    validate: bool = True


def _resolve_image_path(images_dir: Path, filename: str) -> Path | None:
    candidate = images_dir / filename
    if candidate.exists():
        return candidate

    stem = Path(filename).stem
    for suffix in VALID_IMAGE_SUFFIXES:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
        candidate = images_dir / f"{stem}{suffix.upper()}"
        if candidate.exists():
            return candidate
    return None


def _copy_or_link(src: Path, dst: Path, use_symlink: bool) -> None:
    if dst.exists():
        return
    if use_symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def _clip_bbox(xmin: float, ymin: float, xmax: float, ymax: float, width: float, height: float) -> tuple[float, float, float, float] | None:
    x1 = max(0.0, min(xmin, width - 1.0))
    y1 = max(0.0, min(ymin, height - 1.0))
    x2 = max(0.0, min(xmax, width - 1.0))
    y2 = max(0.0, min(ymax, height - 1.0))

    if x2 <= x1 + 1.0 or y2 <= y1 + 1.0:
        return None
    return x1, y1, x2, y2


def _normalize_split(raw_split: str) -> str | None:
    split = str(raw_split).strip().lower()
    return SPLIT_MAP.get(split)


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _iter_image_files(images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in images_dir.glob("*")
        if path.suffix.lower() in VALID_IMAGE_SUFFIXES and path.is_file()
    )


def validate_yolo_dataset(dataset_yaml_path: Path) -> dict:
    dataset_yaml_path = dataset_yaml_path.resolve()
    dataset_cfg = read_yaml(dataset_yaml_path)
    dataset_root = Path(dataset_cfg.get("path", dataset_yaml_path.parent))
    if not dataset_root.is_absolute():
        dataset_root = (dataset_yaml_path.parent / dataset_root).resolve()

    names = dataset_cfg.get("names", [])
    if isinstance(names, dict):
        names = sorted_name_values(names)
    class_count = int(dataset_cfg.get("nc", len(names)))

    summary = {
        "valid": True,
        "dataset_yaml": str(dataset_yaml_path),
        "dataset_root": str(dataset_root),
        "class_count": class_count,
        "split_summary": {},
        "warnings": [],
        "errors": [],
    }

    for split in ("train", "val", "test"):
        split_value = dataset_cfg.get(split)
        if split_value is None:
            summary["warnings"].append(f"{split} split is missing from dataset.yaml")
            continue

        split_path = Path(split_value)
        if not split_path.is_absolute():
            split_path = (dataset_root / split_path).resolve()

        if split_path.suffix.lower() == ".txt":
            listed_images: list[Path] = []
            if split_path.exists():
                listed_images = [Path(line.strip()) for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            label_dir = None
            image_count = len(listed_images)
            label_count = None
        else:
            label_dir = split_path.parent / "labels"
            image_count = len(_iter_image_files(split_path)) if split_path.exists() else 0
            label_count = len(list(label_dir.glob("*.txt"))) if label_dir.exists() else 0

        split_info = {
            "path": str(split_path),
            "image_count": image_count,
            "label_count": label_count,
        }

        if not split_path.exists():
            summary["errors"].append(f"{split} images path does not exist: {split_path}")
        elif image_count == 0:
            summary["warnings"].append(f"{split} split has no images: {split_path}")

        if label_dir is not None and label_dir.exists():
            invalid_labels = 0
            missing_labels = 0
            for image_path in _iter_image_files(split_path):
                label_path = label_dir / f"{image_path.stem}.txt"
                if not label_path.exists():
                    missing_labels += 1
                    continue
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        invalid_labels += 1
                        continue
                    try:
                        class_id = int(float(parts[0]))
                        coords = [float(value) for value in parts[1:]]
                    except ValueError:
                        invalid_labels += 1
                        continue
                    if class_id < 0 or class_id >= class_count:
                        invalid_labels += 1
                        continue
                    xc, yc, width, height = coords
                    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
                        invalid_labels += 1

            split_info["missing_labels"] = missing_labels
            split_info["invalid_label_rows"] = invalid_labels
            if missing_labels:
                summary["warnings"].append(f"{split} split has {missing_labels} images without labels")
            if invalid_labels:
                summary["errors"].append(f"{split} split has {invalid_labels} invalid label rows")

        summary["split_summary"][split] = split_info

    summary["valid"] = len(summary["errors"]) == 0
    return summary


def build_yolo_dataset(config: DatasetBuildConfig) -> dict:
    workspace_root = get_workspace_root(config.workspace_root)
    annotations_csv = resolve_workspace_path(config.annotations_csv, workspace_root)
    images_dir = resolve_workspace_path(config.images_dir, workspace_root)
    output_dir = resolve_workspace_path(config.output_dir, workspace_root)
    seed_metadata = seed_everything(config.seed, deterministic=False)

    if not annotations_csv.exists():
        raise FileNotFoundError(f"Annotations not found: {annotations_csv}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if output_dir.exists() and config.force:
        shutil.rmtree(output_dir)

    for split in ("train", "val", "test"):
        ensure_dir(output_dir / split / "images")
        ensure_dir(output_dir / split / "labels")

    resolved_config = asdict(config)
    resolved_config["annotations_csv"] = str(annotations_csv.resolve())
    resolved_config["images_dir"] = str(images_dir.resolve())
    resolved_config["output_dir"] = str(output_dir.resolve())
    resolved_config["workspace_root"] = str(workspace_root.resolve())
    write_yaml(output_dir / "resolved_config.yaml", resolved_config)
    write_json(
        output_dir / "runtime.json",
        collect_runtime_metadata(
            workspace_root=workspace_root,
            extra={
                "task": "prepare_dataset",
                "annotations_csv": str(annotations_csv.resolve()),
                "images_dir": str(images_dir.resolve()),
                "output_dir": str(output_dir.resolve()),
            },
        ),
    )
    write_json(output_dir / "seed_plan.json", {"seed": int(config.seed), "deterministic": False, **seed_metadata})

    df = pd.read_csv(annotations_csv)
    _validate_columns(
        df,
        required=["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax", "split"],
    )

    df["split_norm"] = df["split"].map(_normalize_split)
    df = df[df["split_norm"].notna()].copy()

    classes = sorted(df["class"].unique().tolist())
    class_to_id = {name: idx for idx, name in enumerate(classes)}

    stats = {
        "total_annotations": int(len(df)),
        "class_count": int(len(classes)),
        "processed_images": 0,
        "missing_images": 0,
        "invalid_boxes": 0,
        "empty_label_images": 0,
        "split_image_count": {"train": 0, "val": 0, "test": 0},
        "split_box_count": {"train": 0, "val": 0, "test": 0},
    }

    grouped_items = list(df.groupby(["split_norm", "filename"], sort=False))
    if config.shuffle:
        random.Random(config.seed).shuffle(grouped_items)

    selected_per_split = {"train": 0, "val": 0, "test": 0}

    for (split, filename), group in tqdm(grouped_items, total=len(grouped_items), desc="Building YOLO dataset"):
        if config.max_images_per_split is not None and selected_per_split[split] >= config.max_images_per_split:
            continue

        image_path = _resolve_image_path(images_dir, str(filename))
        if image_path is None:
            stats["missing_images"] += 1
            continue

        width = float(group.iloc[0]["width"])
        height = float(group.iloc[0]["height"])

        yolo_lines: list[str] = []
        for _, row in group.iterrows():
            clipped = _clip_bbox(
                xmin=float(row["xmin"]),
                ymin=float(row["ymin"]),
                xmax=float(row["xmax"]),
                ymax=float(row["ymax"]),
                width=width,
                height=height,
            )
            if clipped is None:
                stats["invalid_boxes"] += 1
                continue

            x1, y1, x2, y2 = clipped
            x_center = ((x1 + x2) / 2.0) / width
            y_center = ((y1 + y2) / 2.0) / height
            box_w = (x2 - x1) / width
            box_h = (y2 - y1) / height
            class_id = class_to_id[str(row["class"])]
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

        if not yolo_lines:
            stats["empty_label_images"] += 1
            continue

        stem = Path(str(filename)).stem
        image_out = output_dir / split / "images" / f"{stem}{image_path.suffix.lower()}"
        label_out = output_dir / split / "labels" / f"{stem}.txt"

        _copy_or_link(image_path, image_out, config.symlink)
        label_out.write_text("\n".join(yolo_lines), encoding="utf-8")

        selected_per_split[split] += 1
        stats["processed_images"] += 1
        stats["split_image_count"][split] += 1
        stats["split_box_count"][split] += len(yolo_lines)

    dataset_yaml = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(classes),
        "names": classes,
    }

    dataset_yaml_path = output_dir / "dataset.yaml"
    write_yaml(dataset_yaml_path, dataset_yaml)
    write_json(output_dir / "class_to_id.json", class_to_id)
    write_json(output_dir / "build_summary.json", stats)

    validation_summary = validate_yolo_dataset(dataset_yaml_path) if config.validate else None
    if validation_summary is not None:
        write_json(output_dir / "validation_summary.json", validation_summary)

    return {
        "dataset_yaml": str(dataset_yaml_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "workspace_root": str(workspace_root.resolve()),
        "class_count": len(classes),
        "resolved_config": str((output_dir / "resolved_config.yaml").resolve()),
        "stats": stats,
        "validation": validation_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build YOLO dataset from labels_with_split.csv")
    parser.add_argument("--annotations-csv", type=Path, default=Path("data/labels_with_split.csv"))
    parser.add_argument("--images-dir", type=Path, default=Path("data/dataset 2"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/yolo_dataset"))
    parser.add_argument("--workspace-root", type=Path, default=None, help="Resolve relative data/output paths against this workspace root")
    parser.add_argument("--force", action="store_true", help="Delete output directory before rebuilding")
    parser.add_argument("--symlink", action="store_true", help="Use symlink instead of copying images")
    parser.add_argument("--smoke", action="store_true", help="Build a small deterministic subset for quick validation")
    parser.add_argument("--max-images-per-split", type=int, default=None, help="Optional cap on images per split")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle grouped examples before selecting a subset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_images_per_split = args.max_images_per_split
    if args.smoke and max_images_per_split is None:
        max_images_per_split = 32

    result = build_yolo_dataset(
        DatasetBuildConfig(
            annotations_csv=args.annotations_csv,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            force=args.force,
            symlink=args.symlink,
            workspace_root=args.workspace_root,
            max_images_per_split=max_images_per_split,
            shuffle=args.shuffle,
            seed=args.seed,
            validate=not args.skip_validation,
        )
    )
    print("Dataset ready")
    print(f"- dataset_yaml: {result['dataset_yaml']}")
    print(f"- class_count: {result['class_count']}")
    print(f"- workspace_root: {result['workspace_root']}")
    print(f"- processed_images: {result['stats']['processed_images']}")
    if result.get("validation") is not None:
        print(f"- validation_valid: {result['validation']['valid']}")


if __name__ == "__main__":
    main()
