from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mad.runtime import collect_runtime_metadata, get_data_cache_root
from mad.utils import ensure_dir, seed_everything, timestamp, write_json, write_yaml

DEFAULT_KAGGLE_DATASET_ID = "a2015003713/militaryaircraftdetectiondataset"
VALID_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


@dataclass
class KaggleYOLOBuildConfig:
    dataset_id: str = DEFAULT_KAGGLE_DATASET_ID
    output_dir: Path | None = None
    cache_root: Path | None = None
    raw_dataset_dir: Path | None = None
    force_download: bool = False
    force_rebuild: bool = False
    validate: bool = True
    use_hf_probe: bool = True
    max_images_per_split: int | None = None
    seed: int = 42
    class_names_path: Path | None = None


def dataset_slug(dataset_id: str) -> str:
    return dataset_id.replace("/", "__").replace(":", "_")


def default_kaggle_dataset_root(dataset_id: str, cache_root: str | Path | None = None) -> Path:
    root = get_data_cache_root(cache_root)
    return ensure_dir(root / "datasets" / dataset_slug(dataset_id))


def default_prepared_dataset_dir(dataset_id: str, cache_root: str | Path | None = None) -> Path:
    return default_kaggle_dataset_root(dataset_id, cache_root) / "yolo_dataset"


def default_prepared_dataset_yaml(dataset_id: str = DEFAULT_KAGGLE_DATASET_ID, cache_root: str | Path | None = None) -> Path:
    return default_prepared_dataset_dir(dataset_id, cache_root) / "dataset.yaml"


def _load_kagglehub() -> tuple[Any, Any, Any]:
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "kagglehub is required for Kaggle-backed dataset preparation. "
            "Install it with `python -m pip install -r requirements-colab.txt`."
        ) from exc

    load_fn = getattr(kagglehub, "load_dataset", None) or getattr(kagglehub, "dataset_load", None)
    return kagglehub, KaggleDatasetAdapter, load_fn


def probe_kaggle_hf_dataset(dataset_id: str) -> dict[str, Any]:
    """Best-effort metadata probe using the KaggleHub HF adapter."""
    try:
        _, adapter, load_fn = _load_kagglehub()
    except Exception as exc:
        return {
            "attempted": False,
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
        }

    if load_fn is None:
        return {
            "attempted": False,
            "status": "unsupported",
            "error": "Neither kagglehub.load_dataset nor kagglehub.dataset_load is available.",
        }

    attempts: list[dict[str, Any]] = []
    for file_path in ("", "metadata.jsonl"):
        try:
            try:
                dataset = load_fn(adapter.HUGGING_FACE, dataset_id, file_path=file_path)
            except TypeError:
                dataset = load_fn(adapter.HUGGING_FACE, dataset_id, file_path)

            num_rows = None
            try:
                num_rows = int(len(dataset))
            except Exception:
                num_rows = None
            return {
                "attempted": True,
                "status": "ok",
                "file_path": file_path,
                "num_rows": num_rows,
                "type": type(dataset).__name__,
            }
        except Exception as exc:
            attempts.append(
                {
                    "file_path": file_path,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return {
        "attempted": True,
        "status": "failed",
        "attempts": attempts,
    }


def download_kaggle_dataset(
    dataset_id: str,
    *,
    cache_root: str | Path | None = None,
    force_download: bool = False,
) -> dict[str, Any]:
    kagglehub, _, _ = _load_kagglehub()
    raw_dir = ensure_dir(default_kaggle_dataset_root(dataset_id, cache_root) / "raw")

    try:
        downloaded_path = kagglehub.dataset_download(
            dataset_id,
            output_dir=str(raw_dir),
            force_download=force_download,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Kaggle dataset download failed. Make sure KaggleHub is authenticated. "
            "In Colab, set a `KAGGLE_API_TOKEN` secret or run `kagglehub.login()`."
        ) from exc

    resolved_path = Path(downloaded_path).expanduser().resolve()
    return {
        "dataset_id": dataset_id,
        "raw_dataset_dir": str(resolved_path),
        "cache_root": str(get_data_cache_root(cache_root)),
    }


def _resolve_layout(raw_dataset_dir: Path) -> tuple[Path, Path]:
    candidates = [
        (raw_dataset_dir / "images", raw_dataset_dir / "annotations" / "yolo"),
        (raw_dataset_dir / "data" / "images", raw_dataset_dir / "data" / "annotations" / "yolo"),
    ]
    for images_root, labels_root in candidates:
        if images_root.exists() and labels_root.exists():
            return images_root.resolve(), labels_root.resolve()
    raise FileNotFoundError(
        "Could not find Kaggle YOLO layout. Expected `images/` and `annotations/yolo/` under the downloaded dataset."
    )


def _load_class_names(class_names_path: Path | None, class_count: int) -> list[str]:
    if class_names_path is None:
        return [f"class_{index:03d}" for index in range(class_count)]

    path = Path(class_names_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"class_names_path not found: {path}")

    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict) and "names" in loaded:
            loaded = loaded["names"]
        if isinstance(loaded, dict):
            names = [loaded[key] for key in sorted(loaded.keys(), key=lambda item: int(item))]
        else:
            names = list(loaded)
    else:
        names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    if len(names) < class_count:
        names.extend(f"class_{index:03d}" for index in range(len(names), class_count))
    return [str(name) for name in names[:class_count]]


def _link_or_copy(src: Path, dst: Path) -> str:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
        return "symlink"
    except Exception:
        shutil.copy2(src, dst)
        return "copy"


def _flatten_name(rel_path: Path, existing_names: set[str]) -> str:
    stem = rel_path.stem
    if stem not in existing_names:
        existing_names.add(stem)
        return stem

    parent_bits = [bit for bit in rel_path.parent.parts if bit not in {".", ""}]
    prefix = "_".join(parent_bits) if parent_bits else "file"
    candidate = f"{prefix}__{stem}"
    counter = 1
    while candidate in existing_names:
        counter += 1
        candidate = f"{prefix}__{stem}_{counter}"
    existing_names.add(candidate)
    return candidate


def _read_max_class_id(label_path: Path) -> int:
    max_class_id = -1
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            max_class_id = max(max_class_id, int(float(parts[0])))
        except Exception:
            continue
    return max_class_id


def _resolve_image_for_label(
    images_root: Path,
    label_rel_path: Path,
    stem_fallback_index: dict[str, Path] | None,
) -> Path | None:
    stem_only = label_rel_path.with_suffix("")
    for suffix in VALID_IMAGE_SUFFIXES:
        candidate = images_root / stem_only.with_suffix(suffix)
        if candidate.exists():
            return candidate.resolve()
        candidate_upper = images_root / stem_only.with_suffix(suffix.upper())
        if candidate_upper.exists():
            return candidate_upper.resolve()

    if stem_fallback_index is None:
        return None
    return stem_fallback_index.get(label_rel_path.stem)


def _build_stem_fallback_index(images_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for image_path in images_root.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in VALID_IMAGE_SUFFIXES:
            index.setdefault(image_path.stem, image_path.resolve())
    return index


def build_kaggle_yolo_dataset(config: KaggleYOLOBuildConfig) -> dict[str, Any]:
    seed_metadata = seed_everything(config.seed, deterministic=False)
    output_dir = Path(config.output_dir).expanduser().resolve() if config.output_dir else default_prepared_dataset_dir(
        config.dataset_id, config.cache_root
    )

    hf_probe = probe_kaggle_hf_dataset(config.dataset_id) if config.use_hf_probe and config.raw_dataset_dir is None else None
    download_info = (
        {"dataset_id": config.dataset_id, "raw_dataset_dir": str(Path(config.raw_dataset_dir).expanduser().resolve())}
        if config.raw_dataset_dir is not None
        else download_kaggle_dataset(config.dataset_id, cache_root=config.cache_root, force_download=config.force_download)
    )
    raw_dataset_dir = Path(download_info["raw_dataset_dir"]).resolve()
    images_root, labels_root = _resolve_layout(raw_dataset_dir)

    if output_dir.exists() and config.force_rebuild:
        shutil.rmtree(output_dir)

    for split in ("train", "val", "test"):
        ensure_dir(output_dir / split / "images")
        ensure_dir(output_dir / split / "labels")

    max_images_per_split = config.max_images_per_split
    selected_per_split = {"train": 0, "val": 0, "test": 0}
    seen_names = {"train": set(), "val": set(), "test": set()}
    split_counts = {"train": 0, "val": 0, "test": 0}
    label_counts = {"train": 0, "val": 0, "test": 0}
    method_counts = {"symlink": 0, "copy": 0}
    max_class_id = -1

    stem_fallback_index: dict[str, Path] | None = None

    for split in ("train", "val", "test"):
        split_label_dir = labels_root / split
        if not split_label_dir.exists():
            continue

        for label_path in sorted(split_label_dir.rglob("*.txt")):
            if max_images_per_split is not None and selected_per_split[split] >= max_images_per_split:
                break

            label_rel_path = label_path.relative_to(split_label_dir)
            image_path = _resolve_image_for_label(images_root, label_rel_path, stem_fallback_index)
            if image_path is None and stem_fallback_index is None:
                stem_fallback_index = _build_stem_fallback_index(images_root)
                image_path = _resolve_image_for_label(images_root, label_rel_path, stem_fallback_index)
            if image_path is None:
                continue

            image_name = _flatten_name(label_rel_path, seen_names[split])
            image_dst = output_dir / split / "images" / f"{image_name}{image_path.suffix.lower()}"
            label_dst = output_dir / split / "labels" / f"{image_name}.txt"

            method = _link_or_copy(image_path, image_dst)
            _link_or_copy(label_path, label_dst)
            method_counts[method] += 1

            max_class_id = max(max_class_id, _read_max_class_id(label_path))
            selected_per_split[split] += 1
            split_counts[split] += 1
            label_counts[split] += len([line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()])

    class_count = max_class_id + 1 if max_class_id >= 0 else 0
    class_names = _load_class_names(config.class_names_path, class_count)

    dataset_yaml: dict[str, Any] = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": class_count,
        "names": class_names,
    }
    if split_counts["test"] > 0:
        dataset_yaml["test"] = "test/images"

    write_yaml(output_dir / "dataset.yaml", dataset_yaml)

    resolved_config = asdict(config)
    resolved_config["output_dir"] = str(output_dir.resolve())
    resolved_config["cache_root"] = str(get_data_cache_root(config.cache_root))
    resolved_config["raw_dataset_dir"] = str(raw_dataset_dir.resolve())
    resolved_config["class_names_path"] = (
        str(Path(config.class_names_path).expanduser().resolve()) if config.class_names_path is not None else None
    )
    write_yaml(output_dir / "resolved_config.yaml", resolved_config)
    write_json(
        output_dir / "runtime.json",
        collect_runtime_metadata(
            extra={
                "task": "prepare_kaggle_dataset",
                "dataset_id": config.dataset_id,
                "raw_dataset_dir": str(raw_dataset_dir.resolve()),
                "output_dir": str(output_dir.resolve()),
            }
        ),
    )
    write_json(output_dir / "seed_plan.json", seed_metadata)

    build_summary = {
        "source": "kaggle",
        "dataset_id": config.dataset_id,
        "prepared_at": timestamp(),
        "raw_dataset_dir": str(raw_dataset_dir.resolve()),
        "images_root": str(images_root.resolve()),
        "labels_root": str(labels_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "dataset_yaml": str((output_dir / "dataset.yaml").resolve()),
        "class_count": class_count,
        "processed_images": int(sum(split_counts.values())),
        "split_image_count": split_counts,
        "split_label_row_count": label_counts,
        "link_methods": method_counts,
        "hf_probe": hf_probe,
        "download_info": download_info,
    }
    write_json(output_dir / "build_summary.json", build_summary)

    from mad.dataset_builder import validate_yolo_dataset

    validation_summary = validate_yolo_dataset(output_dir / "dataset.yaml") if config.validate else None
    if validation_summary is not None:
        write_json(output_dir / "validation_summary.json", validation_summary)

    return {
        "dataset_yaml": str((output_dir / "dataset.yaml").resolve()),
        "output_dir": str(output_dir.resolve()),
        "raw_dataset_dir": str(raw_dataset_dir.resolve()),
        "class_count": class_count,
        "stats": build_summary,
        "validation": validation_summary,
    }
