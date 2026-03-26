"""
mad/colab_utils.py
------------------
Google Colab 환경에서 실험을 시작하기 위한 단일 초기화 함수 모음.
모든 노트북의 setup 셀은 이 모듈의 setup_colab_env()를 호출하는 것으로 통일된다.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_REPO_DIR = Path("/content/Military_Object_Detection")
DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive")
DEFAULT_WORKSPACE_NAME = "Military_Object_Detection"


def setup_colab_env(
    repo_dir: str | Path | None = None,
    workspace_root: str | Path | None = None,
    *,
    verbose: bool = True,
) -> dict[str, Path]:
    """
    Colab 실험 환경을 초기화한다. 노트북의 모든 setup 셀에서 이 함수를 호출하라.

    수행 작업:
    1. repo_dir를 sys.path에 추가하고 cwd로 설정
    2. MAD_WORKSPACE_ROOT 환경변수 설정
    3. Ultralytics/Matplotlib 캐시 디렉토리를 workspace 내부로 고정
    4. 현재 환경 정보 출력 (verbose=True)

    Parameters
    ----------
    repo_dir : 저장소 루트 디렉토리 (기본값: /content/Military_Object_Detection)
    workspace_root : Drive 백업 워크스페이스 (기본값: /content/drive/MyDrive/Military_Object_Detection)
    verbose : 환경 정보 출력 여부

    Returns
    -------
    dict with keys: repo_dir, workspace_root
    """
    repo_dir = Path(repo_dir) if repo_dir else DEFAULT_REPO_DIR
    workspace_root = (
        Path(workspace_root) if workspace_root
        else DEFAULT_DRIVE_ROOT / DEFAULT_WORKSPACE_NAME
    )

    # ── 1. sys.path / cwd ──────────────────────────────────────────────────
    if not repo_dir.exists():
        raise FileNotFoundError(
            f"[setup_colab_env] Repository not found: {repo_dir}\n"
            "Run notebooks/00_colab_setup.ipynb first."
        )

    os.chdir(repo_dir)
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    # ── 2. MAD_WORKSPACE_ROOT ─────────────────────────────────────────────
    os.environ["MAD_WORKSPACE_ROOT"] = str(workspace_root)

    # ── 3. Ultralytics / Matplotlib 캐시 고정 ─────────────────────────────
    _ultr_dir = repo_dir / ".ultralytics"
    _mpl_dir = repo_dir / ".matplotlib"
    _ultr_dir.mkdir(parents=True, exist_ok=True)
    _mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(_ultr_dir))
    os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", str(_ultr_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

    if verbose:
        _print_env_summary(repo_dir, workspace_root)

    return {"repo_dir": repo_dir, "workspace_root": workspace_root}


def check_gpu(require: bool = False) -> dict[str, Any]:
    """
    GPU 가용성을 확인하고 요약 정보를 반환한다.

    Parameters
    ----------
    require : True이면 GPU가 없을 때 RuntimeError 발생
    """
    info: dict[str, Any] = {"cuda": False, "mps": False, "device": "cpu"}
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda"] = True
            info["device"] = "cuda"
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            print(f"✅ GPU 사용 가능: {info['gpu_name']} ({info['gpu_mem_gb']} GB)")
        elif getattr(getattr(torch, "backends", None), "mps", None) and torch.backends.mps.is_available():
            info["mps"] = True
            info["device"] = "mps"
            print("✅ MPS (Apple Silicon) 사용 가능")
        else:
            msg = "⚠️  GPU를 찾을 수 없습니다. CPU로 실행됩니다 (매우 느림)."
            if require:
                raise RuntimeError(msg + "\nColab 메뉴 → 런타임 → 런타임 유형 변경 → GPU 선택")
            print(msg)
    except ImportError:
        print("⚠️  torch가 설치되지 않았습니다.")
    return info


def check_dataset(dataset_yaml: str | Path) -> bool:
    """
    dataset.yaml과 split 경로가 존재하는지 빠르게 확인한다.
    """
    dataset_yaml = Path(dataset_yaml)
    if not dataset_yaml.exists():
        print(f"❌ dataset.yaml을 찾을 수 없습니다: {dataset_yaml}")
        print("   → notebooks/01_prepare_dataset.ipynb를 먼저 실행하세요.")
        return False

    try:
        import yaml
        cfg = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
        root = Path(cfg.get("path", dataset_yaml.parent))
        if not root.is_absolute():
            root = (dataset_yaml.parent / root).resolve()
        for split in ("train", "val"):
            split_val = cfg.get(split)
            if split_val is None:
                print(f"⚠️  dataset.yaml에 '{split}' split이 없습니다.")
                continue
            split_path = Path(split_val)
            if not split_path.is_absolute():
                split_path = (root / split_path).resolve()
            if not split_path.exists():
                print(f"❌ {split} 경로가 존재하지 않습니다: {split_path}")
                return False
    except Exception as exc:
        print(f"⚠️  dataset.yaml 검사 중 오류: {exc}")
        return False

    print(f"✅ dataset.yaml 확인 완료: {dataset_yaml}")
    return True


def _print_env_summary(repo_dir: Path, workspace_root: Path) -> None:
    try:
        import torch
        torch_ver = torch.__version__
        cuda_info = f"CUDA {torch.version.cuda}" if torch.cuda.is_available() else "CPU only"
    except ImportError:
        torch_ver = "not installed"
        cuda_info = "N/A"

    try:
        import ultralytics
        ultr_ver = ultralytics.__version__
    except ImportError:
        ultr_ver = "not installed"

    print("=" * 55)
    print("  MAD Colab 환경 초기화 완료")
    print("=" * 55)
    print(f"  repo_dir       : {repo_dir}")
    print(f"  workspace_root : {workspace_root}")
    print(f"  cwd            : {os.getcwd()}")
    print(f"  torch          : {torch_ver} ({cuda_info})")
    print(f"  ultralytics    : {ultr_ver}")
    print("=" * 55)
