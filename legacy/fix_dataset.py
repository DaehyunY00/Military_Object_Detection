#!/usr/bin/env python3
"""Force rebuild dataset with a clean output directory."""

from pathlib import Path

from mad.dataset_builder import DatasetBuildConfig, build_yolo_dataset


if __name__ == "__main__":
    result = build_yolo_dataset(
        DatasetBuildConfig(
            annotations_csv=Path("data/labels_with_split.csv"),
            images_dir=Path("data/dataset 2"),
            output_dir=Path("data/processed/yolo_dataset"),
            force=True,
            symlink=False,
        )
    )
    print(result)
