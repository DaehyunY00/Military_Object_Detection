# legacy/

이 디렉토리는 `mad/` 패키지 도입 이전의 구형 스크립트를 보관합니다.
현재 파이프라인에서는 사용하지 않습니다.

| 파일 | 대체 모듈 |
|---|---|
| `validate_annotations.py` | `mad/dataset_builder.py` → `validate_yolo_dataset()` |
| `monitor_training.py` | `mad/benchmark.py` 아티팩트 자동 기록으로 대체 |
| `fix_dataset.py` | `scripts/prepare_dataset.py` 또는 `python -m mad.dataset_builder --force` |
| `data_converter.py` | `scripts/prepare_dataset.py` |
| `evaluate.py` | `scripts/evaluate.py` 또는 `python -m mad.inference eval` |

> 이 파일들은 절대 경로 하드코딩 등의 이슈가 있으므로 직접 실행하지 마세요.
