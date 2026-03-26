# Military Object Detection Research Pipeline

Colab-first research codebase for controlled detector benchmarking, optional synthetic augmentation, and paper-friendly experiment outputs on the Military Aircraft Detection dataset.

## Research focus

- Compare baseline and stronger object detection models under matched settings
- Evaluate whether synthetic image augmentation improves detection quality
- Keep runs reproducible, inspectable, and easy to summarize for a paper
- Support notebook-first workflows without giving up local CLI usage

## First-pass architecture

This implementation pass keeps the existing core package flat and readable, but adds a Colab-first runtime layer, explicit configs, and notebook entry points:

```text
.
├── mad/
│   ├── runtime.py                  # Colab/workspace detection and path helpers
│   ├── dataset_builder.py          # CSV -> YOLO conversion + validation
│   ├── benchmark.py                # benchmark runner + study summaries
│   ├── inference.py                # evaluation and prediction helpers
│   ├── synthetic_augmentation.py   # optional synthetic augmentation
│   └── utils.py
├── scripts/
│   ├── prepare_dataset.py
│   ├── benchmark.py
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   ├── augment.py
│   └── run_*.py                    # backward-compatible wrappers
├── notebooks/
│   ├── 00_colab_setup.ipynb
│   ├── 01_prepare_dataset.ipynb
│   ├── 02_train_baseline.ipynb
│   ├── 03_run_benchmark.ipynb
│   ├── 04_synthetic_augmentation.ipynb
│   └── 05_evaluate_and_visualize.ipynb
├── configs/
│   ├── colab_quick.yaml
│   ├── colab_full.yaml
│   ├── local_debug.yaml
│   ├── benchmark_baseline.yaml
│   ├── benchmark_baseline_m4_quick.yaml
│   ├── benchmark_sota.yaml
│   ├── synthetic_quick.yaml
│   └── synthetic_diffusion.yaml
├── requirements.txt
├── requirements-colab.txt
└── requirements-augmentation.txt   # optional diffusion dependencies
```

## Colab-first workflow

Relative dataset and output paths now resolve against `MAD_WORKSPACE_ROOT`.

- Local default: repository root
- Colab recommendation: `/content/drive/MyDrive/Military_Object_Detection`

That means you can keep code under `/content/Military_Object_Detection` and store datasets, checkpoints, artifacts, and experiment summaries on Google Drive.

## Install

Local:

```bash
python -m pip install -r requirements.txt
```

Colab:

```bash
python -m pip install -r requirements-colab.txt
```

Optional diffusion augmentation dependencies:

```bash
python -m pip install -r requirements-augmentation.txt
```

Weights & Biases is optional. None of the default Colab configs require it.

## Quickstart: Colab

### Option A: notebook-first

Run the notebooks in order:

1. `notebooks/00_colab_setup.ipynb`
2. `notebooks/01_prepare_dataset.ipynb`
3. `notebooks/03_run_benchmark.ipynb`
4. `notebooks/05_evaluate_and_visualize.ipynb`

Optional:

5. `notebooks/04_synthetic_augmentation.ipynb`
6. rerun `notebooks/03_run_benchmark.ipynb` on the augmented dataset

### Option B: Colab shell cells

```bash
git clone https://github.com/DaehyunY00/Military_Object_Detection.git /content/Military_Object_Detection
cd /content/Military_Object_Detection
python -m pip install -r requirements-colab.txt
```

Then in Python:

```python
import os
os.environ["MAD_WORKSPACE_ROOT"] = "/content/drive/MyDrive/Military_Object_Detection"
```

Prepare dataset:

```bash
python scripts/prepare_dataset.py \
  --workspace-root "$MAD_WORKSPACE_ROOT" \
  --annotations-csv data/labels_with_split.csv \
  --images-dir "data/dataset 2" \
  --output-dir data/processed/yolo_dataset \
  --force
```

Quick benchmark:

```bash
python scripts/benchmark.py \
  --config configs/colab_quick.yaml \
  --workspace-root "$MAD_WORKSPACE_ROOT"
```

Evaluate the best model:

```bash
python scripts/evaluate.py \
  --workspace-root "$MAD_WORKSPACE_ROOT" \
  --model /path/to/best.pt \
  --dataset-yaml data/processed/yolo_dataset/dataset.yaml \
  --split test
```

## Quickstart: Local

Set a workspace root only if you want outputs outside the repository:

```bash
export MAD_WORKSPACE_ROOT="$(pwd)"
```

Smoke-test dataset prep:

```bash
python scripts/prepare_dataset.py \
  --workspace-root "$MAD_WORKSPACE_ROOT" \
  --annotations-csv data/labels_with_split.csv \
  --images-dir "data/dataset 2" \
  --output-dir data/processed/yolo_dataset_smoke \
  --force \
  --smoke
```

Local debug benchmark:

```bash
python scripts/benchmark.py \
  --config configs/local_debug.yaml \
  --workspace-root "$MAD_WORKSPACE_ROOT"
```

Evaluate a trained checkpoint:

```bash
python scripts/evaluate.py \
  --workspace-root "$MAD_WORKSPACE_ROOT" \
  --model /path/to/best.pt \
  --dataset-yaml data/processed/yolo_dataset/dataset.yaml \
  --split test
```

Run folder or single-image inference:

```bash
python scripts/infer.py \
  --workspace-root "$MAD_WORKSPACE_ROOT" \
  --model /path/to/best.pt \
  --source data/processed/yolo_dataset/test/images \
  --output-dir artifacts/inference
```

## Dataset preparation

`mad.dataset_builder` now supports:

- workspace-root-aware paths
- deterministic smoke-test subsets with `--smoke` or `--max-images-per-split`
- dataset validation after conversion
- `dataset.yaml`, `build_summary.json`, `validation_summary.json`
- `resolved_config.yaml`, `runtime.json`, and `seed_plan.json`

Example:

```bash
python scripts/prepare_dataset.py \
  --workspace-root "$MAD_WORKSPACE_ROOT" \
  --annotations-csv data/labels_with_split.csv \
  --images-dir "data/dataset 2" \
  --output-dir data/processed/yolo_dataset \
  --force \
  --smoke
```

## Benchmarking

Available presets:

- `configs/colab_quick.yaml`: short Colab sanity-check benchmark
- `configs/colab_full.yaml`: longer Colab baseline comparison
- `configs/local_debug.yaml`: minimal local smoke benchmark
- `configs/benchmark_baseline.yaml`: legacy baseline benchmark, now `device: auto`
- `configs/benchmark_sota.yaml`: legacy stronger-model benchmark

New benchmark behavior:

- relative dataset and output paths resolve against `MAD_WORKSPACE_ROOT`
- each benchmark run writes to `experiments/<name>/study_<timestamp>/`
- each study now uses a standard layout with `runs/`, `metadata/`, `summaries/`, and `artifacts/`
- a `latest/` directory keeps the newest metadata and summary files in the same layout
- latest summary files are also copied to the experiment root for convenience
- seeds are normalized centrally and saved in `seed_plan.json`
- `resolved_config.yaml` and `runtime.json` are saved with each study

Key outputs:

- `benchmark_all_runs.csv`
- `benchmark_all_runs.json`
- `benchmark_model_summary.csv`
- `benchmark_best_runs.csv`
- `benchmark_summary.md`
- `leaderboard.md`
- `run_summary.json`
- `resolved_config.yaml`
- `runtime.json`
- `seed_plan.json`

Example study layout:

```text
experiments/colab_quick/
├── benchmark_all_runs.csv
├── benchmark_model_summary.csv
├── benchmark_best_runs.csv
├── benchmark_summary.md
├── leaderboard.md
├── resolved_config.yaml
├── runtime.json
├── seed_plan.json
├── run_summary.json
├── latest/
│   ├── metadata/
│   └── summaries/
└── study_20260326_123456/
    ├── runs/
    ├── metadata/
    ├── summaries/
    └── artifacts/
```

## Evaluation and visualization

`mad.inference` now writes stable evaluation artifacts:

- `metrics.json`
- `metrics.md`
- Ultralytics evaluation plots under the evaluation save directory
- prediction overlays and `detections.csv` for inference

Notebook options:

- `notebooks/05_evaluate_and_visualize.ipynb`
- `inference.ipynb` as a backward-compatible notebook entry point

## Synthetic augmentation

Synthetic augmentation remains optional in this pass.

- Default notebook path uses the lightweight procedural preset in `configs/synthetic_quick.yaml`
- Diffusion dependencies are isolated in `requirements-augmentation.txt`
- `mode=auto` can fall back to procedural when diffusion is unavailable or impractical
- `--disable-diffusion` forces the procedural path
- `--strict-diffusion` fails instead of silently falling back
- Existing CLI stays available through `scripts/augment.py` and `scripts/run_synthetic_augmentation.py`
- Synthetic outputs also snapshot `resolved_config.yaml`, `runtime.json`, and `seed_plan.json`

Quick Colab-friendly fallback:

```bash
python scripts/augment.py \
  --config configs/synthetic_quick.yaml \
  --workspace-root "$MAD_WORKSPACE_ROOT"
```

Explicit diffusion with procedural fallback:

```bash
python scripts/augment.py \
  --config configs/synthetic_diffusion.yaml \
  --workspace-root "$MAD_WORKSPACE_ROOT"
```

Explicit diffusion with no fallback:

```bash
python scripts/augment.py \
  --config configs/synthetic_diffusion.yaml \
  --workspace-root "$MAD_WORKSPACE_ROOT" \
  --strict-diffusion
```

## Backward compatibility

The older entry points still exist:

- `scripts/run_benchmark.py`
- `scripts/run_inference.py`
- `scripts/run_synthetic_augmentation.py`
- `scripts/prepare_dataset.py`
- `train_background.py`
- `evaluate.py`
- `data_converter.py`

The new `scripts/*.py` names are the preferred interface going forward, but the older wrappers are preserved to avoid breaking existing notes and commands.

## Reproducibility notes

- Keep `MAD_WORKSPACE_ROOT` fixed for a study and avoid moving output folders mid-experiment
- Prefer YAML configs committed to the repo, then use CLI overrides only for temporary path or debug changes
- Benchmark seeds are normalized centrally, recorded in `seed_plan.json`, and passed through both the benchmark runner and Ultralytics training args
- Use the saved `resolved_config.yaml`, `runtime.json`, and `seed_plan.json` when reporting results
- Use `latest/metadata/` and `latest/summaries/` for the newest run, and cite the matching `study_<timestamp>/` folder for archival references
- Use multiple seeds for final comparisons, not just screening runs
- Keep augmentation off unless it is part of the experiment being evaluated

Recommended multi-seed benchmark command:

```bash
python scripts/benchmark.py \
  --config configs/colab_full.yaml \
  --workspace-root "$MAD_WORKSPACE_ROOT"
```

To make the run multi-seed, edit the config to set both:

```yaml
seed: 42
seeds: [41, 42, 43]
```

## Lightweight smoke checks

Run the lightweight reproducibility checks without the full dataset or a GPU:

```bash
python test_pipeline.py
```

or:

```bash
python -m unittest tests.test_reproducibility_smoke -v
```

These smoke checks cover:

- centralized seeding behavior
- minimal YOLO dataset validation
- benchmark summary artifact generation and latest-pointer output
