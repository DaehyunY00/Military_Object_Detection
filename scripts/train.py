#!/usr/bin/env python3
"""단일 모델 훈련 진입점 — mad.benchmark 엔진을 직접 사용한다."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_background import main


if __name__ == "__main__":
    main()
