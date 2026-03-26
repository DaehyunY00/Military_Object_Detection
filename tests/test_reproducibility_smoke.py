from __future__ import annotations

import os
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from mad.benchmark import write_benchmark_artifacts
from mad.dataset_builder import validate_yolo_dataset
from mad.runtime import ensure_result_layout
from mad.utils import normalize_seeds, read_json, read_yaml, seed_everything


class ReproducibilitySmokeTests(unittest.TestCase):
    def test_seed_everything_is_repeatable(self) -> None:
        seeds = normalize_seeds(42, [7, 11, 7])
        self.assertEqual(seeds, [7, 11])

        first = seed_everything(123, deterministic=True)
        random_values_a = [random.random() for _ in range(3)]
        numpy_values_a = np.random.rand(3).tolist()

        second = seed_everything(123, deterministic=True)
        random_values_b = [random.random() for _ in range(3)]
        numpy_values_b = np.random.rand(3).tolist()

        self.assertEqual(random_values_a, random_values_b)
        self.assertEqual(numpy_values_a, numpy_values_b)
        self.assertEqual(os.environ.get("PYTHONHASHSEED"), "123")
        self.assertTrue(first["deterministic"])
        self.assertEqual(first["seed"], second["seed"])

    def test_validate_yolo_dataset_on_minimal_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for split in ("train", "val", "test"):
                image_dir = root / split / "images"
                label_dir = root / split / "labels"
                image_dir.mkdir(parents=True, exist_ok=True)
                label_dir.mkdir(parents=True, exist_ok=True)
                (image_dir / f"{split}_sample.jpg").write_bytes(b"jpg")
                (label_dir / f"{split}_sample.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")

            dataset_yaml = root / "dataset.yaml"
            dataset_yaml.write_text(
                "\n".join(
                    [
                        f"path: {root.resolve()}",
                        "train: train/images",
                        "val: val/images",
                        "test: test/images",
                        "nc: 1",
                        "names:",
                        "  - aircraft",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = validate_yolo_dataset(dataset_yaml)

            self.assertTrue(summary["valid"])
            self.assertEqual(summary["class_count"], 1)
            self.assertEqual(summary["split_summary"]["train"]["image_count"], 1)
            self.assertEqual(summary["split_summary"]["val"]["invalid_label_rows"], 0)

    def test_write_benchmark_artifacts_creates_standard_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "experiments" / "smoke"
            study_layout = ensure_result_layout(output_dir, study_id="study_smoke")
            latest_layout = ensure_result_layout(output_dir / "latest")
            dataset_yaml = root / "data" / "dataset.yaml"
            dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
            dataset_yaml.write_text("path: .\ntrain: train/images\nval: val/images\ntest: test/images\nnc: 1\nnames: ['aircraft']\n", encoding="utf-8")
            config_path = root / "configs" / "smoke.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text("study_name: smoke\n", encoding="utf-8")

            records = [
                {
                    "model_id": "yolov8n",
                    "seed": 42,
                    "status": "ok",
                    "val_map50": 0.42,
                    "val_map50_95": 0.28,
                    "test_map50": 0.4,
                    "test_map50_95": 0.26,
                    "train_eval_minutes": 1.2,
                    "best_model_size_mb": 6.1,
                    "best_model": str(root / "weights" / "a.pt"),
                },
                {
                    "model_id": "yolov8n",
                    "seed": 43,
                    "status": "ok",
                    "val_map50": 0.46,
                    "val_map50_95": 0.31,
                    "test_map50": 0.43,
                    "test_map50_95": 0.3,
                    "train_eval_minutes": 1.1,
                    "best_model_size_mb": 6.1,
                    "best_model": str(root / "weights" / "b.pt"),
                },
                {
                    "model_id": "yolo11n",
                    "seed": 42,
                    "status": "failed",
                    "error": "simulated failure",
                    "train_eval_minutes": 0.2,
                },
            ]

            summary = write_benchmark_artifacts(
                records=records,
                config_path=config_path,
                dataset_yaml=dataset_yaml,
                workspace_root=root,
                output_dir=output_dir,
                study_layout=study_layout,
                latest_layout=latest_layout,
                resolved_config={"study_name": "smoke", "seed": 42, "seeds": [42, 43]},
                runtime_metadata={"device": "cpu", "is_colab": False},
                seed_plan={"seed": 42, "seeds": [42, 43], "deterministic": True},
            )

            self.assertEqual(summary["successful_runs"], 2)
            self.assertEqual(summary["failed_runs"], 1)

            summary_csv = Path(study_layout["summaries_dir"]) / "benchmark_model_summary.csv"
            best_runs_csv = Path(study_layout["summaries_dir"]) / "benchmark_best_runs.csv"
            summary_md = Path(study_layout["summaries_dir"]) / "benchmark_summary.md"
            seed_plan_json = Path(study_layout["metadata_dir"]) / "seed_plan.json"
            latest_summary_csv = Path(latest_layout["summaries_dir"]) / "benchmark_model_summary.csv"

            self.assertTrue(summary_csv.exists())
            self.assertTrue(best_runs_csv.exists())
            self.assertTrue(summary_md.exists())
            self.assertTrue(seed_plan_json.exists())
            self.assertTrue(latest_summary_csv.exists())

            model_summary = pd.read_csv(summary_csv)
            best_runs = pd.read_csv(best_runs_csv)
            self.assertIn("val_map50_95_mean", model_summary.columns)
            self.assertEqual(int(model_summary.iloc[0]["unique_seeds"]), 2)
            self.assertEqual(len(best_runs), 1)

            run_summary = read_json(Path(study_layout["metadata_dir"]) / "run_summary.json")
            resolved_cfg = read_yaml(Path(study_layout["metadata_dir"]) / "resolved_config.yaml")
            self.assertEqual(run_summary["primary_metric"], "test_map50_95")
            self.assertEqual(resolved_cfg["study_name"], "smoke")


if __name__ == "__main__":
    unittest.main(verbosity=2)
