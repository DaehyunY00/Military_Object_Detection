# Guideline: Military Aircraft Detection SOTA Study

## 1. 연구 질문

- RQ1: `Military Aircraft Detection Dataset`에서 어떤 탐지 모델이 최고 성능(mAP50-95, mAP50)을 보이는가?
- RQ2: Diffusion 기반 synthetic augmentation이 성능을 유의미하게 향상시키는가?
- RQ3: 성능 향상은 특정 클래스(희소 클래스)에서 더 큰가?

## 2. 실험 설계 원칙

- 동일 비교 조건: 입력 해상도, epoch, optimizer family, 평가 split 고정
- 1차 스크리닝: 단일 seed로 모델 후보 축소
- 2차 확증: 상위 모델에 대해 다중 seed(예: 3 seeds)
- 데이터 누수 방지: synthetic 데이터는 train에만 추가, val/test는 원본 유지
- 보고 지표: mAP50-95(primary), mAP50, Precision, Recall, 추론시간(ms/image)

## 3. 추천 실험 매트릭스

1. Baseline 실험
- `configs/benchmark_baseline.yaml`
- 목적: 기본 성능 기준선 확보

2. SOTA 후보 실험
- `configs/benchmark_sota.yaml`
- 목적: 최고 성능 후보 모델 선정

3. 생성 증강 실험
- Base dataset vs Augmented dataset (`data/processed/augmented_diffusion/dataset.yaml`)
- 동일 모델/동일 하이퍼파라미터로 A/B 비교

4. Ablation 실험
- synthetic 개수: 500 / 1000 / 2000 / 4000
- Diffusion model 변경: SD2.1-base vs SDXL 계열
- object-per-image: 1~2 vs 3~4

## 4. 실행 프로토콜

1. 데이터 변환
```bash
python scripts/prepare_dataset.py --force
```

2. 베이스라인 벤치마크
```bash
python scripts/run_benchmark.py --config configs/benchmark_baseline.yaml
```

M4 기본 옵션에서 빠른 1차 스크리닝이 필요하면:
```bash
python scripts/run_benchmark.py --config configs/benchmark_baseline_m4_quick.yaml
```

실시간 모니터링:
- W&B를 사용하려면 사전에 `wandb login` 실행
- run URL은 결과 CSV(`benchmark_all_runs.csv`)의 `wandb_run_url` 컬럼으로 확인

3. SOTA 후보 벤치마크
```bash
python scripts/run_benchmark.py --config configs/benchmark_sota.yaml
```

4. Diffusion 증강 생성
```bash
python scripts/run_synthetic_augmentation.py \
  --dataset-yaml data/processed/yolo_dataset/dataset.yaml \
  --output-dir data/processed/augmented_diffusion \
  --synthetic-count 2000 \
  --mode diffusion
```

5. 증강 데이터 재평가
- 벤치마크 config의 `dataset_yaml`을 증강 YAML로 교체 후 재실행

## 5. 통계 검증(논문 필수)

- 상위 2~3개 모델에 대해 seeds = [42, 3407, 2025] 반복
- 각 모델별 평균±표준편차 보고
- paired t-test 또는 Wilcoxon signed-rank test로 증강 전후 비교
- 효과 크기(Cohen's d) 병행 보고

## 6. 에러 분석 권장 항목

- 클래스별 AP 하위 10개 클래스 분석
- 작은 객체/복잡 배경/밀집 객체 시나리오별 성능 분해
- False Positive 유형 분류
  - bird/cloud/contrail 오인식
  - 유사 기종 혼동(F16/F18 등)

## 7. 논문 작성 템플릿(권장)

1. Introduction
- 군용기 탐지의 필요성
- 클래스 수/롱테일 문제/생성증강 필요성

2. Related Work
- 현대 객체탐지(one-stage, transformer-based)
- diffusion 기반 데이터 증강

3. Method
- 데이터 전처리 파이프라인
- 모델 비교 프로토콜
- diffusion+copy-paste synthetic 생성 방식

4. Experiments
- 데이터셋/지표/환경
- baseline/sota/ablation 결과
- 통계 검증

5. Analysis
- 클래스별 성능, 실패 사례, 계산비용 대비 성능

6. Conclusion
- 최고 모델 제시
- 생성증강의 실효성 및 한계

## 8. 필수 산출물 체크리스트

- `benchmark_all_runs.csv`
- `benchmark_model_summary.csv`
- `leaderboard.md`
- 증강 메타데이터(`metadata.json`, `synthetic_records.json`)
- 논문용 그림
  - 모델별 mAP bar chart
  - 클래스별 AP heatmap
  - 정성적 추론 결과(inference.ipynb 캡처)

## 9. 재현성 체크리스트

- 코드 커밋 해시 또는 버전 기록
- 사용 모델 가중치 이름/버전 명시
- seed 및 하이퍼파라미터 공개
- train/val/test split 고정값 유지
