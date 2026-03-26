#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mad.benchmark import run_benchmark
from mad.dataset_builder import DatasetBuildConfig, build_yolo_dataset
from mad.kaggle_dataset import (
    DEFAULT_KAGGLE_DATASET_ID,
    KaggleYOLOBuildConfig,
    build_kaggle_yolo_dataset,
    default_prepared_dataset_yaml,
)
from mad.runtime import resolve_workspace_path
from mad.synthetic_augmentation import SyntheticConfig, generate_augmented_dataset
from mad.utils import ensure_dir, read_yaml, timestamp, write_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full study: base benchmark + diffusion augmentation benchmark")
    parser.add_argument("--dataset-yaml", type=Path, default=None)
    parser.add_argument("--benchmark-config", type=Path, default=Path("configs/benchmark_sota.yaml"))
    parser.add_argument("--workspace-root", type=Path, default=None)
    parser.add_argument("--build-dataset-if-missing", action="store_true")
    parser.add_argument("--dataset-source", choices=["kaggle", "manual"], default="kaggle")
    parser.add_argument("--kaggle-dataset-id", type=str, default=DEFAULT_KAGGLE_DATASET_ID)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--raw-dataset-dir", type=Path, default=None)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--class-names-path", type=Path, default=None)
    parser.add_argument("--annotations-csv", type=Path, default=Path("data/labels_with_split.csv"))
    parser.add_argument("--images-dir", type=Path, default=Path("data/dataset 2"))
    parser.add_argument("--synthetic-count", type=int, default=2000)
    parser.add_argument("--synthetic-mode", type=str, choices=["auto", "diffusion", "procedural"], default="auto")
    parser.add_argument("--disable-diffusion", action="store_true")
    parser.add_argument("--strict-diffusion", action="store_true")
    parser.add_argument("--skip-augmentation", action="store_true")
    parser.add_argument("--diffusion-model-id", type=str, default="stabilityai/stable-diffusion-2-1-base")
    return parser.parse_args()


def _ensure_dataset(dataset_yaml: Path, args: argparse.Namespace) -> Path:
    dataset_yaml = resolve_workspace_path(dataset_yaml, args.workspace_root) if dataset_yaml is not None else default_prepared_dataset_yaml(
        args.kaggle_dataset_id,
        cache_root=args.cache_root,
    )
    if dataset_yaml.exists():
        return dataset_yaml
    if not args.build_dataset_if_missing:
        raise FileNotFoundError(f"dataset_yaml not found: {dataset_yaml}. Use --build-dataset-if-missing.")

    if args.dataset_source == "manual":
        result = build_yolo_dataset(
            DatasetBuildConfig(
                annotations_csv=args.annotations_csv,
                images_dir=args.images_dir,
                output_dir=dataset_yaml.parent,
                force=False,
                symlink=False,
                workspace_root=args.workspace_root,
            )
        )
    else:
        result = build_kaggle_yolo_dataset(
            KaggleYOLOBuildConfig(
                dataset_id=args.kaggle_dataset_id,
                output_dir=dataset_yaml.parent,
                cache_root=args.cache_root,
                raw_dataset_dir=args.raw_dataset_dir,
                force_download=args.force_download,
                force_rebuild=False,
                validate=True,
                seed=42,
                class_names_path=args.class_names_path,
            )
        )
    return Path(result["dataset_yaml"])


def main() -> None:
    args = parse_args()
    dataset_yaml = _ensure_dataset(args.dataset_yaml, args)

    base_cfg = read_yaml(args.benchmark_config)
    base_cfg["dataset_yaml"] = str(dataset_yaml.resolve())
    if args.workspace_root is not None:
        base_cfg["workspace_root"] = str(args.workspace_root)

    # 임시 config는 output_dir/metadata/ 아래에 저장한다.
    # (artifacts/ 루트 오염 방지 및 실험별 추적 가능)
    base_out_dir = ensure_dir(Path(base_cfg.get("output_dir", "experiments/detector_benchmark")))
    base_cfg_path = ensure_dir(base_out_dir / "metadata") / f"benchmark_base_{timestamp()}.yaml"
    write_yaml(base_cfg_path, base_cfg)
    base_summary = run_benchmark(base_cfg_path)
    print("[1/3] Base benchmark done")
    for key, value in base_summary.items():
        print(f"- {key}: {value}")

    if args.skip_augmentation:
        return

    synthetic_mode = "procedural" if args.disable_diffusion else args.synthetic_mode
    # workspace_root 기준으로 Drive에 저장 (Colab 런타임 종료 시 유실 방지)
    _ws_root = Path(base_cfg.get("workspace_root", "data/processed")).parent if args.workspace_root is None else args.workspace_root
    synth_dir = Path(_ws_root) / "data" / "processed" / f"augmented_{synthetic_mode}_{timestamp()}"
    synth_meta = generate_augmented_dataset(
        SyntheticConfig(
            dataset_yaml=dataset_yaml,
            output_dir=synth_dir,
            synthetic_count=args.synthetic_count,
            mode=synthetic_mode,
            diffusion_model_id=args.diffusion_model_id,
            workspace_root=args.workspace_root,
            allow_fallback=not args.strict_diffusion,
            fallback_mode="procedural",
            seed=base_cfg.get("seed", 42),  # 명시적 seed 전달 (재현성 보장)
        )
    )
    print("[2/3] Synthetic dataset done")
    print(f"- output_dataset_yaml: {synth_meta['output_dataset_yaml']}")

    aug_cfg = dict(base_cfg)
    aug_cfg["dataset_yaml"] = synth_meta["output_dataset_yaml"]
    aug_out_dir = Path(base_cfg.get("output_dir", "experiments/detector_benchmark")) / "with_synthetic"
    aug_cfg["output_dir"] = str(aug_out_dir)
    aug_cfg_path = ensure_dir(aug_out_dir / "metadata") / f"benchmark_aug_{timestamp()}.yaml"
    write_yaml(aug_cfg_path, aug_cfg)

    aug_summary = run_benchmark(aug_cfg_path)
    print("[3/3] Augmented benchmark done")
    for key, value in aug_summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
