from __future__ import annotations

import argparse
import os
import shutil
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from mad.runtime import (
    collect_runtime_metadata,
    ensure_result_layout,
    get_workspace_root,
    resolve_workspace_path,
)
from mad.utils import (
    configure_ultralytics_env,
    ensure_dir,
    normalize_seeds,
    read_yaml,
    resolve_device,
    seed_everything,
    timestamp,
    write_json,
    write_markdown,
    write_yaml,
)


def _to_float_dict(metrics: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(metrics, dict):
        return {}
    cleaned: dict[str, float] = {}
    for key, value in metrics.items():
        try:
            cleaned[str(key)] = float(value)
        except Exception:
            continue
    return cleaned


def _to_str_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [v.strip() for v in values.split(",") if v.strip()]
    if isinstance(values, (list, tuple, set)):
        return [str(v) for v in values]
    return [str(values)]


def _init_wandb_run(
    cfg: dict[str, Any],
    run_name: str,
    model_id: str,
    weights: str,
    seed: int,
    device: str,
    train_args: dict[str, Any],
) -> tuple[Any | None, Any | None]:
    wandb_cfg = cfg.get("wandb", {}) or {}
    if not bool(wandb_cfg.get("enabled", False)):
        return None, None

    try:
        import wandb
    except ImportError:
        print("[WARN] wandb.enabled=true but `wandb` is not installed. Run `pip install wandb`.")
        return None, None

    mode = str(wandb_cfg.get("mode", "online")).lower()
    if mode not in {"online", "offline", "disabled"}:
        mode = "online"

    os.environ.setdefault("WANDB_SILENT", str(wandb_cfg.get("silent", "true")).lower())
    if wandb_cfg.get("api_key_env"):
        key = os.environ.get(str(wandb_cfg["api_key_env"]))
        if key:
            os.environ.setdefault("WANDB_API_KEY", key)

    run_config = {
        "study_name": cfg.get("study_name", "detector_benchmark"),
        "dataset_yaml": str(Path(cfg["dataset_yaml"]).resolve()),
        "model_id": model_id,
        "weights": weights,
        "seed": int(seed),
        "device": device,
        "epochs": int(train_args.get("epochs", 0)),
        "imgsz": int(train_args.get("imgsz", 0)),
        "batch": int(train_args.get("batch", 0)),
        "fraction": float(train_args.get("fraction", 1.0)),
    }

    init_kwargs: dict[str, Any] = {
        "project": str(wandb_cfg.get("project", "military-object-detection")),
        "name": run_name,
        "config": run_config,
        "reinit": True,
        "mode": mode,
    }
    if wandb_cfg.get("entity"):
        init_kwargs["entity"] = str(wandb_cfg["entity"])
    if wandb_cfg.get("group"):
        init_kwargs["group"] = str(wandb_cfg["group"])
    if wandb_cfg.get("job_type"):
        init_kwargs["job_type"] = str(wandb_cfg["job_type"])
    tags = _to_str_list(wandb_cfg.get("tags"))
    if tags:
        init_kwargs["tags"] = tags
    if wandb_cfg.get("notes"):
        init_kwargs["notes"] = str(wandb_cfg["notes"])
    if wandb_cfg.get("anonymous"):
        init_kwargs["anonymous"] = str(wandb_cfg["anonymous"])

    try:
        run = wandb.init(**init_kwargs)
    except Exception as exc:
        print(f"[WARN] W&B run init failed ({run_name}): {exc}")
        return None, None
    return wandb, run


def _attach_wandb_callbacks(model: Any, wandb: Any) -> None:
    def _on_train_epoch_end(trainer: Any) -> None:
        if not getattr(wandb, "run", None):
            return
        payload: dict[str, float] = {}
        try:
            payload.update(_to_float_dict(trainer.label_loss_items(trainer.tloss, prefix="train")))
        except Exception:
            pass
        try:
            payload.update(_to_float_dict(dict(trainer.lr)))
        except Exception:
            pass
        if payload:
            wandb.log(payload, step=trainer.epoch + 1, commit=False)

    def _on_fit_epoch_end(trainer: Any) -> None:
        if not getattr(wandb, "run", None):
            return
        metrics = _to_float_dict(getattr(trainer, "metrics", {}))
        if metrics:
            wandb.log(metrics, step=trainer.epoch + 1, commit=True)

    def _on_train_end(trainer: Any) -> None:
        if not getattr(wandb, "run", None):
            return
        try:
            wandb.run.summary["save_dir"] = str(Path(trainer.save_dir).resolve())
        except Exception:
            pass

    model.add_callback("on_train_epoch_end", _on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
    model.add_callback("on_train_end", _on_train_end)


def _extract_metrics(results: Any) -> dict[str, float | None]:
    box = getattr(results, "box", None)
    speed = getattr(results, "speed", {}) or {}

    def _safe(value: Any) -> float | None:
        try:
            return float(value)
        except Exception:
            return None

    return {
        "map50": _safe(getattr(box, "map50", None)),
        "map50_95": _safe(getattr(box, "map", None)),
        "precision": _safe(getattr(box, "mp", None)),
        "recall": _safe(getattr(box, "mr", None)),
        "speed_preprocess_ms": _safe(speed.get("preprocess")),
        "speed_inference_ms": _safe(speed.get("inference")),
        "speed_postprocess_ms": _safe(speed.get("postprocess")),
    }


def _has_test_split(dataset_yaml: Path) -> bool:
    cfg = read_yaml(dataset_yaml)
    return "test" in cfg and cfg["test"] is not None


def _evaluate_model(model: Any, dataset_yaml: Path, device: str, split: str) -> dict[str, float | None]:
    results = model.val(
        data=str(dataset_yaml),
        split=split,
        device=device,
        plots=False,
        verbose=False,
        save_json=False,
    )
    return _extract_metrics(results)


def _default_train_kwargs(cfg: dict[str, Any], run_dir: Path, run_name: str, device: str) -> dict[str, Any]:
    train_cfg = cfg.get("train", {})
    train_args = {
        "data": str(Path(cfg["dataset_yaml"]).resolve()),
        "epochs": int(train_cfg.get("epochs", 60)),
        "imgsz": int(train_cfg.get("imgsz", 640)),
        "batch": int(train_cfg.get("batch", 4)),
        "workers": int(train_cfg.get("workers", 2)),
        "optimizer": train_cfg.get("optimizer", "auto"),
        "lr0": float(train_cfg.get("lr0", 0.01)),
        "lrf": float(train_cfg.get("lrf", 0.01)),
        "cos_lr": bool(train_cfg.get("cos_lr", True)),
        "weight_decay": float(train_cfg.get("weight_decay", 0.0005)),
        "patience": int(train_cfg.get("patience", 20)),
        "close_mosaic": int(train_cfg.get("close_mosaic", 8)),
        "cache": bool(train_cfg.get("cache", False)),
        "amp": bool(train_cfg.get("amp", True)),
        "fraction": float(train_cfg.get("fraction", 1.0)),
        "device": device,
        "project": str(run_dir),
        "name": run_name,
        "exist_ok": True,
        "seed": int(train_cfg.get("seed", cfg.get("seed", 42))),
        "deterministic": bool(train_cfg.get("deterministic", True)),
        "save": True,
        "val": True,
        "verbose": True,
    }
    if "time" in train_cfg:
        train_args["time"] = float(train_cfg["time"])
    return train_args


def _write_leaderboard(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        out_path.write_text("No successful runs.\n", encoding="utf-8")
        return

    order_col = _primary_metric_column(df)
    ranked = df.sort_values(order_col, ascending=_metric_sort_ascending(order_col), na_position="last").reset_index(drop=True)

    lines = [
        "# Detector Benchmark Leaderboard",
        "",
        f"Sorted by `{order_col}`.",
        "",
        "| rank | model_id | seed | test_map50_95 | val_map50_95 | test_map50 | val_map50 | status |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for idx, row in ranked.iterrows():
        lines.append(
            "| {rank} | {model} | {seed} | {tmap95} | {vmap95} | {tmap50} | {vmap50} | {status} |".format(
                rank=idx + 1,
                model=row.get("model_id", ""),
                seed=int(row.get("seed", 0)),
                tmap95=_format_metric(row.get("test_map50_95")),
                vmap95=_format_metric(row.get("val_map50_95")),
                tmap50=_format_metric(row.get("test_map50")),
                vmap50=_format_metric(row.get("val_map50")),
                status=row.get("status", ""),
            )
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_latest_artifact(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def _apply_overrides(cfg: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    if not overrides:
        return cfg

    merged = dict(cfg)
    train_cfg = dict(merged.get("train", {}) or {})
    merged["train"] = train_cfg

    for key, value in overrides.items():
        if value is None:
            continue
        if key.startswith("train."):
            train_cfg[key.split(".", 1)[1]] = value
        elif key == "models":
            allowed = {str(item) for item in value}
            merged["models"] = [item for item in merged.get("models", []) if str(item.get("id")) in allowed]
        elif key == "wandb.enabled":
            wandb_cfg = dict(merged.get("wandb", {}) or {})
            wandb_cfg["enabled"] = bool(value)
            merged["wandb"] = wandb_cfg
        else:
            merged[key] = value

    return merged


def _resolve_benchmark_paths(cfg: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    workspace_root = get_workspace_root(cfg.get("workspace_root"))
    resolved_cfg = dict(cfg)
    resolved_cfg["workspace_root"] = str(workspace_root.resolve())
    resolved_cfg["dataset_yaml"] = str(resolve_workspace_path(resolved_cfg["dataset_yaml"], workspace_root))
    resolved_cfg["output_dir"] = str(
        resolve_workspace_path(resolved_cfg.get("output_dir", "experiments/detector_benchmark"), workspace_root)
    )
    return resolved_cfg, workspace_root


def _metric_sort_ascending(metric_name: str) -> bool:
    return any(token in metric_name for token in ("speed", "minute"))


def _primary_metric_column(df: pd.DataFrame) -> str:
    for column in ("test_map50_95", "val_map50_95", "test_map50", "val_map50"):
        if column in df.columns and df[column].notna().any():
            return column
    return "train_eval_minutes"


def _format_metric(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    return f"{numeric:.{digits}f}"


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"

    columns = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        values: list[str] = []
        for column in df.columns:
            value = row[column]
            if isinstance(value, float):
                values.append(_format_metric(value))
            elif value is None or pd.isna(value):
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _unique_columns(columns: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for column in columns:
        if column in seen:
            continue
        unique.append(column)
        seen.add(column)
    return unique


def build_benchmark_summary_frames(records: list[dict[str, Any]]) -> dict[str, Any]:
    all_df = pd.DataFrame(records)
    if "status" not in all_df.columns:
        all_df["status"] = pd.Series(dtype="object")
    success_df = all_df[all_df["status"] == "ok"].copy()
    primary_metric = _primary_metric_column(success_df if not success_df.empty else all_df)

    metric_cols = [
        col
        for col in [
            "val_map50",
            "val_map50_95",
            "test_map50",
            "test_map50_95",
            "train_eval_minutes",
            "best_model_size_mb",
        ]
        if col in success_df.columns
    ]

    model_summary = pd.DataFrame()
    best_runs = pd.DataFrame()

    if not success_df.empty:
        agg_kwargs: dict[str, Any] = {
            "successful_runs": ("model_id", "size"),
            "unique_seeds": ("seed", "nunique"),
        }
        for metric in metric_cols:
            agg_kwargs[f"{metric}_mean"] = (metric, "mean")
            agg_kwargs[f"{metric}_std"] = (metric, "std")

        model_summary = success_df.groupby("model_id", as_index=False).agg(**agg_kwargs)
        if primary_metric in success_df.columns:
            if _metric_sort_ascending(primary_metric):
                best_primary = success_df.groupby("model_id")[primary_metric].min().rename(f"best_{primary_metric}")
            else:
                best_primary = success_df.groupby("model_id")[primary_metric].max().rename(f"best_{primary_metric}")
            model_summary = model_summary.merge(best_primary, on="model_id", how="left")
            model_summary = model_summary.sort_values(
                f"{primary_metric}_mean" if f"{primary_metric}_mean" in model_summary.columns else f"best_{primary_metric}",
                ascending=_metric_sort_ascending(primary_metric),
                na_position="last",
            ).reset_index(drop=True)
        model_summary.insert(0, "rank", range(1, len(model_summary) + 1))

        best_runs = (
            success_df.sort_values(primary_metric, ascending=_metric_sort_ascending(primary_metric), na_position="last")
            .groupby("model_id", as_index=False)
            .head(1)
            .reset_index(drop=True)
        )
        best_runs.insert(0, "rank", range(1, len(best_runs) + 1))

    return {
        "all_runs": all_df,
        "successful_runs": success_df,
        "model_summary": model_summary,
        "best_runs": best_runs,
        "primary_metric": primary_metric,
    }


def write_benchmark_artifacts(
    *,
    records: list[dict[str, Any]],
    config_path: Path,
    dataset_yaml: Path,
    workspace_root: Path,
    output_dir: Path,
    study_layout: dict[str, str],
    latest_layout: dict[str, str],
    resolved_config: dict[str, Any],
    runtime_metadata: dict[str, Any],
    seed_plan: dict[str, Any],
) -> dict[str, Any]:
    study_root = Path(study_layout["root"])
    study_metadata_dir = Path(study_layout["metadata_dir"])
    study_summaries_dir = Path(study_layout["summaries_dir"])
    latest_root = Path(latest_layout["root"])
    latest_metadata_dir = Path(latest_layout["metadata_dir"])
    latest_summaries_dir = Path(latest_layout["summaries_dir"])

    summary_frames = build_benchmark_summary_frames(records)
    all_df = summary_frames["all_runs"]
    success_df = summary_frames["successful_runs"]
    model_summary = summary_frames["model_summary"]
    best_runs = summary_frames["best_runs"]
    primary_metric = summary_frames["primary_metric"]

    write_yaml(study_metadata_dir / "resolved_config.yaml", resolved_config)
    write_json(study_metadata_dir / "runtime.json", runtime_metadata)
    write_json(study_metadata_dir / "seed_plan.json", seed_plan)

    all_runs_csv = study_summaries_dir / "benchmark_all_runs.csv"
    all_runs_json = study_summaries_dir / "benchmark_all_runs.json"
    model_summary_csv = study_summaries_dir / "benchmark_model_summary.csv"
    best_runs_csv = study_summaries_dir / "benchmark_best_runs.csv"
    leaderboard_md = study_summaries_dir / "leaderboard.md"
    summary_md = study_summaries_dir / "benchmark_summary.md"

    all_df.to_csv(all_runs_csv, index=False)
    write_json(all_runs_json, records)
    model_summary.to_csv(model_summary_csv, index=False)
    best_runs.to_csv(best_runs_csv, index=False)
    _write_leaderboard(success_df, leaderboard_md)

    run_summary = {
        "config_path": str(config_path.resolve()),
        "dataset_yaml": str(dataset_yaml.resolve()),
        "workspace_root": str(workspace_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "study_dir": str(study_root.resolve()),
        "latest_dir": str(latest_root.resolve()),
        "primary_metric": primary_metric,
        "total_runs": int(len(all_df)),
        "successful_runs": int((all_df["status"] == "ok").sum()) if "status" in all_df.columns else 0,
        "failed_runs": int((all_df["status"] != "ok").sum()) if "status" in all_df.columns else 0,
    }
    write_json(study_metadata_dir / "run_summary.json", run_summary)

    summary_lines = [
        "# Benchmark Summary",
        "",
        f"- study_dir: `{study_root.resolve()}`",
        f"- latest_dir: `{latest_root.resolve()}`",
        f"- dataset_yaml: `{dataset_yaml.resolve()}`",
        f"- primary_metric: `{primary_metric}`",
        f"- total_runs: {run_summary['total_runs']}",
        f"- successful_runs: {run_summary['successful_runs']}",
        f"- failed_runs: {run_summary['failed_runs']}",
        f"- seeds: `{', '.join(str(seed) for seed in seed_plan.get('seeds', []))}`",
        "",
        "## Best Runs Per Model",
        "",
        _dataframe_to_markdown(
            best_runs[
                _unique_columns(
                    [
                        col
                        for col in ["rank", "model_id", "seed", primary_metric, "val_map50_95", "test_map50_95", "best_model"]
                        if col in best_runs.columns
                    ]
                )
            ]
        ),
        "",
        "## Aggregate Summary",
        "",
        _dataframe_to_markdown(model_summary),
        "",
    ]
    write_markdown(summary_md, "\n".join(summary_lines))

    copies = [
        (study_metadata_dir / "resolved_config.yaml", latest_metadata_dir / "resolved_config.yaml"),
        (study_metadata_dir / "runtime.json", latest_metadata_dir / "runtime.json"),
        (study_metadata_dir / "seed_plan.json", latest_metadata_dir / "seed_plan.json"),
        (study_metadata_dir / "run_summary.json", latest_metadata_dir / "run_summary.json"),
        (all_runs_csv, latest_summaries_dir / "benchmark_all_runs.csv"),
        (all_runs_json, latest_summaries_dir / "benchmark_all_runs.json"),
        (model_summary_csv, latest_summaries_dir / "benchmark_model_summary.csv"),
        (best_runs_csv, latest_summaries_dir / "benchmark_best_runs.csv"),
        (leaderboard_md, latest_summaries_dir / "leaderboard.md"),
        (summary_md, latest_summaries_dir / "benchmark_summary.md"),
        (all_runs_csv, output_dir / "benchmark_all_runs.csv"),
        (all_runs_json, output_dir / "benchmark_all_runs.json"),
        (model_summary_csv, output_dir / "benchmark_model_summary.csv"),
        (best_runs_csv, output_dir / "benchmark_best_runs.csv"),
        (leaderboard_md, output_dir / "leaderboard.md"),
        (summary_md, output_dir / "benchmark_summary.md"),
        (study_metadata_dir / "resolved_config.yaml", output_dir / "resolved_config.yaml"),
        (study_metadata_dir / "runtime.json", output_dir / "runtime.json"),
        (study_metadata_dir / "seed_plan.json", output_dir / "seed_plan.json"),
        (study_metadata_dir / "run_summary.json", output_dir / "run_summary.json"),
    ]
    for src, dst in copies:
        _copy_latest_artifact(src, dst)

    return run_summary


def run_benchmark(config_path: Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    configure_ultralytics_env()
    from ultralytics import YOLO

    cfg = _apply_overrides(read_yaml(config_path), overrides)
    cfg, workspace_root = _resolve_benchmark_paths(cfg)
    dataset_yaml = Path(cfg["dataset_yaml"]).resolve()
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"dataset_yaml not found: {dataset_yaml}")

    out_dir = ensure_dir(Path(cfg["output_dir"]))
    study_id = f"study_{timestamp()}"
    study_layout = ensure_result_layout(out_dir, study_id=study_id)
    latest_layout = ensure_result_layout(out_dir / "latest")
    run_root = Path(study_layout["runs_dir"])

    train_cfg = cfg.get("train", {}) or {}
    deterministic = bool(train_cfg.get("deterministic", True))
    seeds = normalize_seeds(cfg.get("seed", 42), cfg.get("seeds"))
    models = cfg.get("models", [])
    if not models:
        raise ValueError("No models defined in benchmark config.")

    device = resolve_device(cfg.get("device", "auto"))
    has_test = _has_test_split(dataset_yaml)
    runtime_metadata = collect_runtime_metadata(
        workspace_root=workspace_root,
        device=device,
        extra={
            "config_path": str(config_path.resolve()),
            "study_id": study_id,
            "study_name": cfg.get("study_name", "detector_benchmark"),
        },
    )
    seed_plan = {
        "seed": int(cfg.get("seed", seeds[0])),
        "seeds": seeds,
        "deterministic": deterministic,
        "study_id": study_id,
        "study_name": cfg.get("study_name", "detector_benchmark"),
        "model_ids": [str(item.get("id")) for item in models],
    }

    records: list[dict[str, Any]] = []

    for model_spec in models:
        model_id = str(model_spec["id"])
        weights = str(model_spec["weights"])

        for seed in seeds:
            seed_metadata = seed_everything(int(seed), deterministic=deterministic)
            run_name = f"{model_id}_seed{seed}_{timestamp()}"
            run_start = perf_counter()
            wandb_module = None
            wandb_run = None

            row: dict[str, Any] = {
                "study_id": study_id,
                "study_name": cfg.get("study_name", "detector_benchmark"),
                "model_id": model_id,
                "weights": weights,
                "seed": int(seed),
                "device": device,
                "deterministic": deterministic,
                "status": "failed",
            }
            row.update({f"seed_{key}": value for key, value in seed_metadata.items()})

            try:
                model = YOLO(weights)
                train_args = _default_train_kwargs(cfg, run_root, run_name, device)
                train_args["seed"] = int(seed)
                train_args.update(model_spec.get("train_overrides", {}))
                wandb_module, wandb_run = _init_wandb_run(
                    cfg=cfg,
                    run_name=run_name,
                    model_id=model_id,
                    weights=weights,
                    seed=int(seed),
                    device=device,
                    train_args=train_args,
                )
                if wandb_module is not None and wandb_run is not None:
                    _attach_wandb_callbacks(model, wandb_module)
                    row["wandb_run_id"] = getattr(wandb_run, "id", None)
                    row["wandb_run_name"] = getattr(wandb_run, "name", None)
                    row["wandb_run_url"] = getattr(wandb_run, "url", None)

                train_results = model.train(**train_args)
                save_dir = Path(train_results.save_dir)
                best_model = save_dir / "weights" / "best.pt"
                checkpoint_type = "best"
                if not best_model.exists():
                    best_model = save_dir / "weights" / "last.pt"
                    checkpoint_type = "last"
                    print(
                        f"[WARN] best.pt not found for {run_name}. "
                        "Using last.pt for evaluation — training may have ended early."
                    )

                eval_model = YOLO(str(best_model))
                val_metrics = _evaluate_model(eval_model, dataset_yaml, device=device, split="val")
                test_metrics = _evaluate_model(eval_model, dataset_yaml, device=device, split="test") if has_test else {}

                row.update(
                    {
                        "status": "ok",
                        "run_name": run_name,
                        "checkpoint_type": checkpoint_type,
                        "save_dir": str(save_dir.resolve()),
                        "best_model": str(best_model.resolve()),
                        "val_map50": val_metrics.get("map50"),
                        "val_map50_95": val_metrics.get("map50_95"),
                        "val_precision": val_metrics.get("precision"),
                        "val_recall": val_metrics.get("recall"),
                        "test_map50": test_metrics.get("map50"),
                        "test_map50_95": test_metrics.get("map50_95"),
                        "test_precision": test_metrics.get("precision"),
                        "test_recall": test_metrics.get("recall"),
                        "val_speed_inference_ms": val_metrics.get("speed_inference_ms"),
                        "test_speed_inference_ms": test_metrics.get("speed_inference_ms"),
                    }
                )

                if best_model.exists():
                    row["best_model_size_mb"] = round(best_model.stat().st_size / (1024 * 1024), 3)
                train_args_yaml = save_dir / "args.yaml"
                if train_args_yaml.exists():
                    row["train_args_yaml"] = str(train_args_yaml.resolve())
                if wandb_module is not None and wandb_run is not None:
                    final_metrics = {
                        "final/val_map50": row.get("val_map50"),
                        "final/val_map50_95": row.get("val_map50_95"),
                        "final/val_precision": row.get("val_precision"),
                        "final/val_recall": row.get("val_recall"),
                        "final/test_map50": row.get("test_map50"),
                        "final/test_map50_95": row.get("test_map50_95"),
                        "final/test_precision": row.get("test_precision"),
                        "final/test_recall": row.get("test_recall"),
                    }
                    final_metrics = {k: float(v) for k, v in final_metrics.items() if v is not None}
                    if final_metrics:
                        wandb_module.log(final_metrics)
                    wandb_run.summary["status"] = "ok"
                    wandb_run.summary["best_model"] = str(best_model.resolve())
                    wandb_run.summary["save_dir"] = str(save_dir.resolve())

            except Exception as exc:
                row["error"] = str(exc)
                row["traceback"] = traceback.format_exc(limit=4)
                if wandb_run is not None:
                    try:
                        wandb_run.summary["status"] = "failed"
                        wandb_run.summary["error"] = str(exc)
                    except Exception:
                        pass

            finally:
                row["train_eval_minutes"] = round((perf_counter() - run_start) / 60.0, 2)
                if wandb_module is not None:
                    try:
                        wandb_module.log({"final/train_eval_minutes": row["train_eval_minutes"]})
                    except Exception:
                        pass
                if wandb_run is not None:
                    try:
                        wandb_module.finish(exit_code=0 if row.get("status") == "ok" else 1)
                    except Exception:
                        pass
            records.append(row)
    return write_benchmark_artifacts(
        records=records,
        config_path=config_path,
        dataset_yaml=dataset_yaml,
        workspace_root=workspace_root,
        output_dir=out_dir,
        study_layout=study_layout,
        latest_layout=latest_layout,
        resolved_config=cfg,
        runtime_metadata=runtime_metadata,
        seed_plan=seed_plan,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detector benchmark from YAML config")
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark_baseline.yaml"))
    parser.add_argument("--workspace-root", type=Path, default=None, help="Resolve relative dataset/output paths against this workspace root")
    parser.add_argument("--dataset-yaml", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--models", nargs="+", default=None, help="Optional subset of model ids from the config")
    parser.add_argument("--wandb", choices=["inherit", "on", "off"], default="inherit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        "workspace_root": args.workspace_root,
        "dataset_yaml": args.dataset_yaml,
        "output_dir": args.output_dir,
        "device": args.device,
        "seed": args.seed,
        "seeds": [int(args.seed)] if args.seed is not None else None,
        "train.epochs": args.epochs,
        "train.imgsz": args.imgsz,
        "train.batch": args.batch,
        "train.workers": args.workers,
        "train.fraction": args.fraction,
        "models": args.models,
        "wandb.enabled": True if args.wandb == "on" else False if args.wandb == "off" else None,
    }
    summary = run_benchmark(args.config, overrides=overrides)
    print("Benchmark finished")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
