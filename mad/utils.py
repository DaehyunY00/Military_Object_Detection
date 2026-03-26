from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_ultralytics_env() -> Path:
    """Force Ultralytics to write settings inside workspace (sandbox-safe).

    Colab에서 MAD_WORKSPACE_ROOT가 설정되어 있으면 Drive 내부에 캐시를 보존합니다.
    그렇지 않으면 PROJECT_ROOT 하위에 캐시를 유지합니다.
    """
    ws_root = os.environ.get("MAD_WORKSPACE_ROOT")
    base_dir = Path(ws_root) if ws_root else PROJECT_ROOT
    config_dir = ensure_dir(base_dir / ".ultralytics")
    mpl_dir = ensure_dir(base_dir / ".matplotlib")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", str(config_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    return config_dir


def normalize_seeds(seed: int | None = None, seeds: list[int] | tuple[int, ...] | None = None) -> list[int]:
    """Return a stable, de-duplicated seed list for a study."""
    if seeds:
        normalized: list[int] = []
        seen: set[int] = set()
        for value in seeds:
            current = int(value)
            if current in seen:
                continue
            normalized.append(current)
            seen.add(current)
        if normalized:
            return normalized
    return [int(42 if seed is None else seed)]


def seed_everything(seed: int, deterministic: bool = True) -> dict[str, Any]:
    """Seed Python, NumPy, and torch with lightweight determinism metadata."""
    seed = int(seed)
    deterministic = bool(deterministic)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    metadata: dict[str, Any] = {
        "seed": seed,
        "deterministic": deterministic,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "torch_available": torch is not None,
        "cuda_available": bool(torch is not None and torch.cuda.is_available()),
    }

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if getattr(torch.backends, "cudnn", None) is not None:
            torch.backends.cudnn.deterministic = deterministic
            torch.backends.cudnn.benchmark = not deterministic
            metadata["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
            metadata["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)

    return metadata


def resolve_device(device: str | None = None) -> str:
    if device and device != "auto":
        return device

    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def write_json(path: Path, data: dict[str, Any] | list[Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)


def read_json(path: Path) -> dict[str, Any] | list[Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_markdown(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def sorted_name_values(names: dict) -> list[str]:
    """YOLO names dict (int 또는 str 키)를 정수 순서로 정렬된 값 리스트로 변환한다."""
    return [names[key] for key in sorted(names.keys(), key=lambda item: int(item) if str(item).isdigit() else str(item))]
