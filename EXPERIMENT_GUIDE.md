# Military Object Detection — 실험 가이드라인

> 최종 업데이트: 2026-03-26 (Colab 환경 최적화)
> **Google Colab 우선** 환경을 기준으로 작성되었습니다.

---

## 목차

1. [시작 전 체크리스트](#1-시작-전-체크리스트)
2. [Google Colab 완전 설치 가이드](#2-google-colab-완전-설치-가이드)
3. [데이터셋 준비](#3-데이터셋-준비)
4. [벤치마크 실험 (핵심)](#4-벤치마크-실험-핵심)
5. [합성 데이터 증강 실험](#5-합성-데이터-증강-실험)
6. [전체 비교 실험 (Full Study)](#6-전체-비교-실험-full-study)
7. [평가 및 추론](#7-평가-및-추론)
8. [실험 결과 해석](#8-실험-결과-해석)
9. [Config 선택 가이드](#9-config-선택-가이드)
10. [재현성 체크리스트](#10-재현성-체크리스트)
11. [자주 발생하는 문제 및 해결책](#11-자주-발생하는-문제-및-해결책)
12. [로컬 환경 실행 (참고)](#12-로컬-환경-실행-참고)

---

## 1. 시작 전 체크리스트

```
[ ] Google Drive에 Military_Object_Detection/ 폴더가 준비되어 있다
[ ] Drive/data/ 에 labels_with_split.csv 와 dataset 2/ 가 업로드되어 있다
[ ] Colab 런타임 유형이 GPU로 설정되어 있다
      메뉴 → 런타임 → 런타임 유형 변경 → T4 GPU 선택
[ ] 논문 실험에는 colab_quick.yaml (fraction=0.25) 을 사용하지 않는다
[ ] W&B를 사용하지 않는 실험의 모든 config에서 wandb.enabled: false 확인
```

---

## 2. Google Colab 완전 설치 가이드

### 2-1. 노트북 순서 (반드시 이 순서대로)

```
notebooks/
├── 00_colab_setup.ipynb          ← ① 최초 1회 (런타임 재시작 시도 재실행)
├── 01_prepare_dataset.ipynb      ← ② 데이터셋 준비 (최초 1회)
├── 02_train_baseline.ipynb       ← 단일 모델 빠른 훈련 (선택)
├── 03_run_benchmark.ipynb        ← ③ 멀티 모델 벤치마크 (핵심)
├── 04_synthetic_augmentation.ipynb ← ④ 합성 데이터 생성 (선택)
└── 05_evaluate_and_visualize.ipynb ← ⑤ 평가 및 시각화
```

### 2-2. `00_colab_setup.ipynb` 상세 설명

| 셀 | 작업 | 비고 |
|---|---|---|
| ① 파라미터 | `REPO_DIR`, `WORKSPACE_ROOT` 설정 | 기본값 그대로 사용 가능 |
| ② Drive 마운트 | `/content/drive` 에 마운트 | Google 계정 인증 필요 |
| ③ 저장소 클론 | `/content/Military_Object_Detection` 에 클론 | 이미 있으면 pull |
| ④ 의존성 설치 | `requirements-colab.txt` 설치 | `INSTALL_AUGMENTATION_DEPS=True` 시 diffusers도 설치 |
| ⑤ 환경 검증 | GPU 확인, 경로 확인 | 이상 없으면 ✅ 표시 |

### 2-3. 런타임 재시작 후 빠른 재설정

런타임이 재시작되면 `/content` 파일이 사라지지만 Drive 파일은 유지됩니다.

```python
# 임의 노트북의 첫 번째 셀에 아래 코드 붙여넣기
import sys, os, subprocess
from pathlib import Path

REPO_DIR       = Path('/content/Military_Object_Detection')
WORKSPACE_ROOT = Path('/content/drive/MyDrive/Military_Object_Detection')

# 저장소가 없으면 재클론
if not REPO_DIR.exists():
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/DaehyunY00/Military_Object_Detection.git',
                    str(REPO_DIR)], check=True)

sys.path.insert(0, str(REPO_DIR))
os.chdir(REPO_DIR)
from mad.colab_utils import setup_colab_env
setup_colab_env(REPO_DIR, WORKSPACE_ROOT)
```

---

## 3. 데이터셋 준비

### 3-1. Drive에 파일 업로드

Google Drive의 `Military_Object_Detection/data/` 폴더 아래에 업로드:
- `labels_with_split.csv` — 어노테이션 CSV
- `dataset 2/` — 이미지 디렉토리 (공백 포함 이름 유의)

### 3-2. 노트북 실행 (`01_prepare_dataset.ipynb`)

파라미터 셀에서 경로 확인 후 모든 셀 실행.

### 3-3. Colab 셀에서 직접 실행

```python
# 03_run_benchmark.ipynb 또는 임의 노트북에서 직접 호출 가능
from mad.dataset_builder import DatasetBuildConfig, build_yolo_dataset
from pathlib import Path

WORKSPACE_ROOT = Path('/content/drive/MyDrive/Military_Object_Detection')

result = build_yolo_dataset(
    DatasetBuildConfig(
        annotations_csv=WORKSPACE_ROOT / 'data' / 'labels_with_split.csv',
        images_dir=WORKSPACE_ROOT / 'data' / 'dataset 2',
        output_dir=WORKSPACE_ROOT / 'data' / 'processed' / 'yolo_dataset',
        workspace_root=WORKSPACE_ROOT,
        validate=True,
    )
)
print(result['dataset_yaml'])
```

### 3-4. `!` 셀 명령어 (Colab 터미널 방식)

```bash
!cd /content/Military_Object_Detection && \
  python -m mad.dataset_builder \
    --annotations-csv "$MAD_WORKSPACE_ROOT/data/labels_with_split.csv" \
    --images-dir "$MAD_WORKSPACE_ROOT/data/dataset 2" \
    --output-dir "$MAD_WORKSPACE_ROOT/data/processed/yolo_dataset" \
    --workspace-root "$MAD_WORKSPACE_ROOT"
```

---

## 4. 벤치마크 실험 (핵심)

### 4-1. Config 선택

| Config | 용도 | epochs | batch | 데이터 비율 | T4 예상 시간 |
|---|---|---|---|---|---|
| `colab_quick.yaml` | ⚠️ 파이프라인 검증용 | 5 | 16 | 25% | ~5분 |
| `colab_full.yaml` | **Colab 실제 실험** | 20 | 16 | 100% | ~60분 |
| `benchmark_baseline.yaml` | **논문 베이스라인** | 45 | 16 | 100% | ~2.5시간 |
| `benchmark_sota.yaml` | 고성능 탐색 | 120 | 16 | 100% | ~8시간+ |
| `local_debug.yaml` | 코드 디버깅 | 1 | 4 | 5% | ~2분 |

### 4-2. 노트북 실행 (`03_run_benchmark.ipynb`)

`CONFIG_PATH` 변수만 바꾸면 됩니다:

```python
# 빠른 검증 (5 epochs, 25% 데이터)
CONFIG_PATH = REPO_DIR / 'configs' / 'colab_quick.yaml'

# Colab 전체 실험 (20 epochs, 100% 데이터) ← 권장
CONFIG_PATH = REPO_DIR / 'configs' / 'colab_full.yaml'

# 논문 기준 (45 epochs, 100% 데이터)
CONFIG_PATH = REPO_DIR / 'configs' / 'benchmark_baseline.yaml'
```

### 4-3. Colab `!` 셀 명령어

```bash
# Colab 전체 실험 (권장)
!cd /content/Military_Object_Detection && \
  python scripts/run_benchmark.py \
    --config configs/colab_full.yaml \
    --dataset-yaml "$MAD_WORKSPACE_ROOT/data/processed/yolo_dataset/dataset.yaml" \
    --output-dir "$MAD_WORKSPACE_ROOT/experiments/colab_full" \
    --device cuda \
    --wandb off

# 특정 모델만 실행
!cd /content/Military_Object_Detection && \
  python scripts/run_benchmark.py \
    --config configs/colab_full.yaml \
    --dataset-yaml "$MAD_WORKSPACE_ROOT/data/processed/yolo_dataset/dataset.yaml" \
    --output-dir "$MAD_WORKSPACE_ROOT/experiments/colab_full" \
    --models yolov8n yolo11n

# 에폭 수 즉석 오버라이드
!cd /content/Military_Object_Detection && \
  python scripts/run_benchmark.py \
    --config configs/colab_full.yaml \
    --dataset-yaml "$MAD_WORKSPACE_ROOT/data/processed/yolo_dataset/dataset.yaml" \
    --epochs 30 --batch 8
```

### 4-4. Python API (노트북 셀)

```python
from mad.benchmark import run_benchmark
from pathlib import Path

REPO_DIR       = Path('/content/Military_Object_Detection')
WORKSPACE_ROOT = Path('/content/drive/MyDrive/Military_Object_Detection')
DATASET_YAML   = WORKSPACE_ROOT / 'data' / 'processed' / 'yolo_dataset' / 'dataset.yaml'

summary = run_benchmark(
    REPO_DIR / 'configs' / 'colab_full.yaml',
    overrides={
        'workspace_root': WORKSPACE_ROOT,
        'dataset_yaml':   DATASET_YAML,
        'output_dir':     WORKSPACE_ROOT / 'experiments' / 'colab_full',
        'wandb.enabled':  False,
        # 필요 시 추가 오버라이드:
        # 'train.epochs': 30,
        # 'train.batch':   8,
    }
)
print(summary)
```

### 4-5. W&B 연동 (선택)

```bash
# Colab 셀에서 로그인
!wandb login

# config 수정 또는 --wandb on 옵션 추가
!python scripts/run_benchmark.py \
    --config configs/colab_full.yaml \
    --dataset-yaml "$MAD_WORKSPACE_ROOT/data/processed/yolo_dataset/dataset.yaml" \
    --wandb on
```

---

## 5. 합성 데이터 증강 실험

### 5-1. 모드 선택

| 모드 | GPU 필요 | 속도 | 배경 품질 |
|---|---|---|---|
| `procedural` | ❌ | 빠름 (이미지당 ~0.1초) | 단순 합성 하늘 |
| `diffusion` | ✅ (강력 권장) | 느림 (이미지당 ~3~5초) | Stable Diffusion 2.1 |
| `auto` | — | 자동 선택 | CUDA 있으면 diffusion, 없으면 procedural |

### 5-2. 노트북 실행 (`04_synthetic_augmentation.ipynb`)

```python
MODE            = 'procedural'   # 빠른 검증
# MODE          = 'diffusion'    # 논문 품질 (GPU 필요)
SYNTHETIC_COUNT = 500            # 생성할 이미지 수
```

### 5-3. Colab `!` 셀 명령어

```bash
# Procedural 모드 (GPU 불필요)
!cd /content/Military_Object_Detection && \
  python scripts/run_synthetic_augmentation.py \
    --dataset-yaml "$MAD_WORKSPACE_ROOT/data/processed/yolo_dataset/dataset.yaml" \
    --output-dir "$MAD_WORKSPACE_ROOT/data/processed/augmented_procedural" \
    --mode procedural \
    --synthetic-count 500 \
    --seed 42

# Diffusion 모드 (GPU 필요, 최초 1회 ~5GB 모델 다운로드)
!pip install -q -r requirements-augmentation.txt  # 최초 1회
!cd /content/Military_Object_Detection && \
  python scripts/run_synthetic_augmentation.py \
    --dataset-yaml "$MAD_WORKSPACE_ROOT/data/processed/yolo_dataset/dataset.yaml" \
    --output-dir "$MAD_WORKSPACE_ROOT/data/processed/augmented_diffusion" \
    --config configs/synthetic_diffusion.yaml \
    --seed 42
```

### 5-4. Python API (노트북 셀)

```python
from mad.synthetic_augmentation import SyntheticConfig, generate_augmented_dataset
from pathlib import Path

WORKSPACE_ROOT = Path('/content/drive/MyDrive/Military_Object_Detection')
DATASET_YAML   = WORKSPACE_ROOT / 'data' / 'processed' / 'yolo_dataset' / 'dataset.yaml'
OUTPUT_DIR     = WORKSPACE_ROOT / 'data' / 'processed' / 'augmented_procedural'

meta = generate_augmented_dataset(
    SyntheticConfig(
        dataset_yaml=DATASET_YAML,
        output_dir=OUTPUT_DIR,
        synthetic_count=500,
        image_size=640,
        mode='procedural',     # 'diffusion' 으로 변경 가능
        seed=42,
        workspace_root=WORKSPACE_ROOT,
    )
)

print(f"생성 완료: {meta['synthetic_count_created']}장")
print(f"dataset_yaml: {meta['output_dataset_yaml']}")
```

---

## 6. 전체 비교 실험 (Full Study)

베이스라인과 합성 증강 효과를 **한 번에** 비교합니다.

```bash
# Colab 셀에서 실행
!cd /content/Military_Object_Detection && \
  python scripts/run_full_study.py \
    --dataset-yaml "$MAD_WORKSPACE_ROOT/data/processed/yolo_dataset/dataset.yaml" \
    --benchmark-config configs/colab_full.yaml \
    --synthetic-count 500 \
    --synthetic-mode procedural

# Diffusion 모드 (GPU T4 이상 필요)
!cd /content/Military_Object_Detection && \
  python scripts/run_full_study.py \
    --dataset-yaml "$MAD_WORKSPACE_ROOT/data/processed/yolo_dataset/dataset.yaml" \
    --benchmark-config configs/colab_full.yaml \
    --synthetic-count 2000 \
    --synthetic-mode diffusion
```

실행 순서: `[1/3]` 베이스라인 → `[2/3]` 합성 데이터 생성 → `[3/3]` 증강 후 재학습

---

## 7. 평가 및 추론

### 7-1. Test Split 평가

```bash
!cd /content/Military_Object_Detection && \
  python -m mad.inference eval \
    --model "$MAD_WORKSPACE_ROOT/experiments/colab_full/latest/runs/<RUN_NAME>/weights/best.pt" \
    --dataset-yaml "$MAD_WORKSPACE_ROOT/data/processed/yolo_dataset/dataset.yaml" \
    --split test \
    --output-dir "$MAD_WORKSPACE_ROOT/artifacts/evaluation" \
    --workspace-root "$MAD_WORKSPACE_ROOT"
```

```python
# Python API
from mad.inference import evaluate_model
from pathlib import Path

metrics = evaluate_model(
    model_path=Path('path/to/best.pt'),
    dataset_yaml=DATASET_YAML,
    split='test',
    device='auto',
    output_dir=WORKSPACE_ROOT / 'artifacts' / 'evaluation',
)
print(metrics)
```

### 7-2. 이미지 추론

```bash
!cd /content/Military_Object_Detection && \
  python -m mad.inference predict \
    --model "$MAD_WORKSPACE_ROOT/experiments/colab_full/latest/runs/<RUN_NAME>/weights/best.pt" \
    --source "$MAD_WORKSPACE_ROOT/data/processed/yolo_dataset/test/images" \
    --conf 0.25 \
    --output-dir "$MAD_WORKSPACE_ROOT/artifacts/inference" \
    --workspace-root "$MAD_WORKSPACE_ROOT"
```

### 7-3. 노트북 시각화 (`05_evaluate_and_visualize.ipynb`)

`RESULTS_ROOT`를 실험 폴더로 지정하면 최고 성능 모델을 자동 탐색하여 시각화합니다.

```python
RESULTS_ROOT = WORKSPACE_ROOT / 'experiments' / 'colab_full'
# 또는 직접 모델 지정
MODEL_PATH = WORKSPACE_ROOT / 'experiments' / 'colab_full' / 'latest' / 'runs' / \
             '<RUN_NAME>' / 'weights' / 'best.pt'
```

---

## 8. 실험 결과 해석

### 8-1. 주요 메트릭

| 메트릭 | 설명 | 논문 주요 지표 |
|---|---|---|
| `test_map50_95` | IoU 0.5~0.95 mAP (COCO 표준) | ✅ 1순위 |
| `test_map50` | IoU 0.5 mAP | ✅ 2순위 |
| `val_map50_95` | Validation mAP | 훈련 중 모니터링 |
| `checkpoint_type` | `best` 또는 `last` | `last`이면 조기 종료 의심 |

### 8-2. 결과 파일 위치

```
$MAD_WORKSPACE_ROOT/experiments/colab_full/
├── study_<timestamp>/
│   ├── summaries/
│   │   ├── leaderboard.md              ← 모델 랭킹
│   │   ├── benchmark_model_summary.csv ← mean ± std
│   │   ├── benchmark_best_runs.csv     ← 모델별 최고 run
│   │   └── benchmark_all_runs.csv      ← 전체 raw 결과
│   └── metadata/
│       ├── resolved_config.yaml        ← 실제 사용된 config
│       ├── seed_plan.json              ← seed 정보
│       └── runtime.json                ← 실행 환경 정보
└── latest/                             ← 가장 최근 study 복사본
```

### 8-3. 결과 읽기 (Python)

```python
import pandas as pd
from pathlib import Path

WORKSPACE_ROOT = Path('/content/drive/MyDrive/Military_Object_Detection')
results_dir = WORKSPACE_ROOT / 'experiments' / 'colab_full' / 'latest' / 'summaries'

# 모델별 요약 (mean ± std)
summary = pd.read_csv(results_dir / 'benchmark_model_summary.csv')
display(summary)

# 리더보드
leaderboard = (results_dir / 'leaderboard.md').read_text(encoding='utf-8')
print(leaderboard)
```

---

## 9. Config 선택 가이드

```
실험 목적에 따른 config 선택:

파이프라인 검증 (5분)
  └─→ colab_quick.yaml

Colab 실제 실험 (1시간)
  └─→ colab_full.yaml  ★ 기본 권장

논문 베이스라인 (2~3시간)
  └─→ benchmark_baseline.yaml

논문 최고 성능 탐색 (8시간+, A100 권장)
  └─→ benchmark_sota.yaml

합성 증강 생성 (GPU 불필요)
  └─→ synthetic_quick.yaml

합성 증강 생성 (diffusion, GPU 필요)
  └─→ synthetic_diffusion.yaml

코드 디버깅 (2분)
  └─→ local_debug.yaml
```

### 새 모델 추가 방법

`configs/colab_full.yaml` 의 `models` 섹션에 추가:

```yaml
models:
  - id: yolov8n
    weights: yolov8n.pt
  # 추가 예시:
  - id: yolov9c
    weights: yolov9c.pt
    train_overrides:
      batch: 4       # 메모리 부족 시 배치 낮춤
```

---

## 10. 재현성 체크리스트

| 항목 | 확인 방법 |
|---|---|
| seed 확인 | `metadata/seed_plan.json` |
| 실제 사용 config | `metadata/resolved_config.yaml` |
| 실행 환경 (Python/CUDA 버전) | `metadata/runtime.json` |
| checkpoint 종류 (best/last) | `benchmark_all_runs.csv` → `checkpoint_type` 컬럼 |
| 재현 실행 | 아래 명령어 참조 |

```bash
# 이전 실험과 동일한 config로 재실행
!python scripts/run_benchmark.py \
  --config "$MAD_WORKSPACE_ROOT/experiments/colab_full/latest/metadata/resolved_config.yaml" \
  --seed 42
```

---

## 11. 자주 발생하는 문제 및 해결책

### ❶ GPU를 찾을 수 없음 / CPU로 실행됨

**해결:**
```
Colab 메뉴 → 런타임 → 런타임 유형 변경 → T4 GPU 선택
```
config에서 `amp: true` 확인 (T4 Mixed Precision 필수).

---

### ❷ CUDA out of memory

```yaml
# configs/colab_full.yaml 에서 배치 낮추기
train:
  batch: 8      # 기본 16 → 8 (OOM 발생 시)
  imgsz: 640    # 유지
  amp: true     # 반드시 true (Mixed Precision 필수)
```
또는 `03_run_benchmark.ipynb` 에서:
```python
OVERRIDE_BATCH = 8
```

> yolov8s, yolov8m 같은 큰 모델은 `train_overrides: {batch: 8}` 로 개별 설정 가능합니다 (`configs/benchmark_baseline.yaml` 참고).

---

### ❸ W&B 인증 오류로 실험 중단

```python
# 노트북에서 끄기
WANDB_MODE = 'off'

# 또는 config에서
# wandb:
#   enabled: false
```

---

### ❹ Drive 마운트 후 경로 오류

```python
# 런타임 재시작 후 빠른 재설정 (이 코드를 첫 셀에 붙여넣기)
import sys, os
REPO_DIR       = '/content/Military_Object_Detection'
WORKSPACE_ROOT = '/content/drive/MyDrive/Military_Object_Detection'
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)
from mad.colab_utils import setup_colab_env
setup_colab_env(REPO_DIR, WORKSPACE_ROOT)
```

---

### ❺ dataset.yaml 을 찾을 수 없음

`01_prepare_dataset.ipynb` 를 먼저 실행했는지 확인:
```python
from mad.colab_utils import check_dataset
DATASET_YAML = '/content/drive/MyDrive/Military_Object_Detection/data/processed/yolo_dataset/dataset.yaml'
check_dataset(DATASET_YAML)
```

---

### ❻ best.pt 없이 last.pt 로 평가됨

```
[WARN] best.pt not found ... Using last.pt for evaluation
```
훈련이 `patience` 내에 개선되지 않아 조기 종료된 것입니다.
```yaml
train:
  epochs: 30     # 늘리기
  patience: 15   # 늘리기
```

---

### ❼ `data/dataset 2/` 경로 공백 오류

```bash
# 반드시 큰따옴표 사용
--images-dir "$MAD_WORKSPACE_ROOT/data/dataset 2"

# 또는 심볼릭 링크로 공백 제거
!ln -sf "$MAD_WORKSPACE_ROOT/data/dataset 2" "$MAD_WORKSPACE_ROOT/data/dataset_v2"
# 이후 --images-dir "$MAD_WORKSPACE_ROOT/data/dataset_v2" 사용
```

---

### ❽ Colab 12시간 런타임 만료 대비

긴 실험(benchmark_sota.yaml 등)은 중간 체크포인트를 Drive에 저장합니다.
`output_dir`을 `WORKSPACE_ROOT/experiments/...`로 설정하면 자동 보존됩니다.

```python
# 완료된 run은 자동으로 Drive에 저장됨
# 재시작 후 남은 모델만 실행하려면 --models 옵션 사용
!python scripts/run_benchmark.py \
  --config configs/benchmark_sota.yaml \
  --models yolo11x rtdetr_l    # 아직 완료되지 않은 모델만
```

---

## 12. 로컬 환경 실행 (참고)

Colab이 아닌 로컬에서 실행할 경우:

```bash
# 의존성 설치
pip install -r requirements.txt

# 데이터셋 빌드
python -m mad.dataset_builder \
  --annotations-csv data/labels_with_split.csv \
  --images-dir "data/dataset 2" \
  --output-dir data/processed/yolo_dataset

# 벤치마크 실행
python scripts/run_benchmark.py \
  --config configs/benchmark_baseline.yaml \
  --device cuda

# 합성 증강
python scripts/run_synthetic_augmentation.py \
  --config configs/synthetic_quick.yaml

# 평가
python -m mad.inference eval \
  --model experiments/baseline_comparison/latest/runs/*/weights/best.pt \
  --dataset-yaml data/processed/yolo_dataset/dataset.yaml \
  --split test
```

> **로컬 메모리가 부족한 경우**: `configs/local_debug.yaml` 또는 `benchmark_baseline.yaml`에서 `batch: 4`로 낮추세요.

---

## 부록: 파일 구조

```
Military_Object_Detection/
├── mad/                          # 핵심 패키지
│   ├── benchmark.py              # 훈련 + 평가 + 리더보드
│   ├── dataset_builder.py        # CSV → YOLO 변환
│   ├── synthetic_augmentation.py # 합성 이미지 생성
│   ├── inference.py              # 평가 / 추론
│   ├── colab_utils.py            # ★ Colab 초기화 헬퍼 (신규)
│   ├── runtime.py                # 경로 해석, Colab 감지
│   └── utils.py                  # seed, IO, device
│
├── configs/                      # 실험 config
│   ├── colab_full.yaml           # ★ Colab 실제 실험 (권장)
│   ├── colab_quick.yaml          # ⚠️ 검증 전용 (fraction=0.25)
│   ├── benchmark_baseline.yaml   # 논문 베이스라인
│   ├── benchmark_sota.yaml       # 고성능 모델 탐색
│   ├── synthetic_diffusion.yaml  # diffusion 증강
│   ├── synthetic_quick.yaml      # procedural 증강
│   └── local_debug.yaml          # 코드 디버깅
│
├── notebooks/                    # Colab 실행 노트북
│   ├── 00_colab_setup.ipynb      # ① 환경 설정
│   ├── 01_prepare_dataset.ipynb  # ② 데이터셋 준비
│   ├── 02_train_baseline.ipynb   # 단일 모델 훈련
│   ├── 03_run_benchmark.ipynb    # ③ 벤치마크 실험
│   ├── 04_synthetic_augmentation.ipynb # ④ 합성 증강
│   └── 05_evaluate_and_visualize.ipynb # ⑤ 평가·시각화
│
├── scripts/                      # CLI 진입점
│   ├── run_benchmark.py
│   ├── run_synthetic_augmentation.py
│   ├── run_full_study.py
│   └── prepare_dataset.py
│
├── train_background.py           # 단일 모델 훈련 진입점
├── test_pipeline.py              # 재현성 스모크 테스트
├── EXPERIMENT_GUIDE.md           # ← 이 파일
└── legacy/                       # 구형 스크립트 (사용 금지)
```
