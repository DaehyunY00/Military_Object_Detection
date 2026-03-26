from __future__ import annotations

import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COLAB_WORKSPACE_NAME = "Military_Object_Detection"
DEFAULT_COLAB_DRIVE_ROOT = Path("/content/drive/MyDrive")


def is_colab_runtime() -> bool:
    try:
        import google.colab  # noqa: F401
    except Exception:
        return False
    return True


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    return PROJECT_ROOT


def get_workspace_root(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()

    env_value = os.environ.get("MAD_WORKSPACE_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()

    if is_colab_runtime():
        return (DEFAULT_COLAB_DRIVE_ROOT / DEFAULT_COLAB_WORKSPACE_NAME).resolve()

    return PROJECT_ROOT


def configure_workspace_env(explicit: str | Path | None = None) -> Path:
    workspace_root = get_workspace_root(explicit)
    os.environ["MAD_WORKSPACE_ROOT"] = str(workspace_root)
    return workspace_root


def resolve_workspace_path(value: str | Path, workspace_root: str | Path | None = None) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (get_workspace_root(workspace_root) / candidate).resolve()


def maybe_resolve_workspace_path(
    value: str | Path | None,
    workspace_root: str | Path | None = None,
) -> Path | None:
    if value is None:
        return None
    return resolve_workspace_path(value, workspace_root=workspace_root)


def ensure_workspace_layout(explicit: str | Path | None = None) -> dict[str, str]:
    workspace_root = configure_workspace_env(explicit)
    layout = {
        "workspace_root": _ensure_dir(workspace_root),
        "data_root": _ensure_dir(workspace_root / "data"),
        "artifacts_root": _ensure_dir(workspace_root / "artifacts"),
        "experiments_root": _ensure_dir(workspace_root / "experiments"),
        "checkpoints_root": _ensure_dir(workspace_root / "checkpoints"),
        "logs_root": _ensure_dir(workspace_root / "logs"),
    }
    return {key: str(value.resolve()) for key, value in layout.items()}


def ensure_result_layout(base_dir: str | Path, study_id: str | None = None) -> dict[str, str]:
    """Create a standard result directory layout for studies and latest pointers."""
    root = Path(base_dir).expanduser().resolve()
    if study_id:
        root = root / study_id

    layout = {
        "root": _ensure_dir(root),
        "runs_dir": _ensure_dir(root / "runs"),
        "metadata_dir": _ensure_dir(root / "metadata"),
        "summaries_dir": _ensure_dir(root / "summaries"),
        "artifacts_dir": _ensure_dir(root / "artifacts"),
    }
    return {key: str(value.resolve()) for key, value in layout.items()}


def mount_google_drive(mount_point: str | Path = "/content/drive", force_remount: bool = False) -> Path:
    mount_path = Path(mount_point)
    if not is_colab_runtime():
        return mount_path

    from google.colab import drive  # type: ignore

    drive.mount(str(mount_path), force_remount=force_remount)
    return mount_path


def collect_runtime_metadata(
    workspace_root: str | Path | None = None,
    *,
    device: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(get_project_root().resolve()),
        "workspace_root": str(get_workspace_root(workspace_root)),
        "cwd": str(Path.cwd().resolve()),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "is_colab": is_colab_runtime(),
        "device": device,
    }

    if torch is not None:
        metadata.update(
            {
                "torch_version": getattr(torch, "__version__", None),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
                "mps_available": bool(
                    getattr(getattr(torch, "backends", None), "mps", None)
                    and torch.backends.mps.is_available()
                ),
            }
        )
        if torch.cuda.is_available():
            try:
                metadata["cuda_device_name"] = torch.cuda.get_device_name(0)
            except Exception:
                metadata["cuda_device_name"] = None
    else:
        metadata["torch_version"] = None

    if extra:
        metadata.update(extra)
    return metadata
