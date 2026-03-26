from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from mad.runtime import collect_runtime_metadata, get_workspace_root, resolve_workspace_path
from mad.utils import ensure_dir, read_yaml, resolve_device, seed_everything, sorted_name_values, timestamp, write_json, write_yaml

VALID_SYNTHETIC_MODES = ("auto", "diffusion", "procedural")
VALID_FALLBACK_MODES = ("procedural",)
DIFFUSION_INSTALL_HINT = "Install optional packages with `python -m pip install -r requirements-augmentation.txt`."


def _resolve_path(base: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def _normalize_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized in {"off", "none", "disabled", "disable"}:
        return "procedural"
    if normalized not in VALID_SYNTHETIC_MODES:
        raise ValueError(f"Unsupported synthetic mode: {mode}. Expected one of {VALID_SYNTHETIC_MODES}.")
    return normalized


def _normalize_fallback_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_FALLBACK_MODES:
        raise ValueError(f"Unsupported fallback mode: {mode}. Expected one of {VALID_FALLBACK_MODES}.")
    return normalized


def _diffusion_dependency_status() -> dict[str, Any]:
    try:
        import torch  # noqa: F401
        from diffusers import DiffusionPipeline  # noqa: F401
    except Exception as exc:  # pragma: no cover
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "install_hint": DIFFUSION_INSTALL_HINT,
        }
    return {
        "available": True,
        "error": None,
        "install_hint": None,
    }


def _read_label_file(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return rows
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, xc, yc, w, h = parts
        rows.append((int(float(cls)), float(xc), float(yc), float(w), float(h)))
    return rows


def _yolo_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x1 = int((xc - w / 2.0) * img_w)
    y1 = int((yc - h / 2.0) * img_h)
    x2 = int((xc + w / 2.0) * img_w)
    y2 = int((yc + h / 2.0) * img_h)
    x1 = max(0, min(x1, img_w - 2))
    y1 = max(0, min(y1, img_h - 2))
    x2 = max(x1 + 1, min(x2, img_w - 1))
    y2 = max(y1 + 1, min(y2, img_h - 1))
    return x1, y1, x2, y2


def _xyxy_to_yolo(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    xc = ((x1 + x2) / 2.0) / img_w
    yc = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return xc, yc, w, h


def _bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _procedural_sky(width: int, height: int) -> np.ndarray:
    top = np.array([random.randint(90, 160), random.randint(120, 200), random.randint(160, 255)], dtype=np.float32)
    bottom = np.array([random.randint(150, 220), random.randint(170, 240), random.randint(190, 255)], dtype=np.float32)
    alpha = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    sky = (top * (1.0 - alpha) + bottom * alpha).astype(np.uint8)
    sky = np.repeat(sky[:, None, :], width, axis=1)

    cloud_count = random.randint(8, 18)
    for _ in range(cloud_count):
        cx = random.randint(0, width - 1)
        cy = random.randint(0, height - 1)
        rx = random.randint(width // 20, width // 8)
        ry = random.randint(height // 30, height // 10)
        color = random.randint(220, 255)
        overlay = sky.copy()
        cv2.ellipse(overlay, (cx, cy), (rx, ry), random.randint(0, 180), 0, 360, (color, color, color), -1)
        alpha_mix = random.uniform(0.03, 0.12)
        sky = cv2.addWeighted(overlay, alpha_mix, sky, 1.0 - alpha_mix, 0)

    return sky


class DiffusionBackgroundGenerator:
    def __init__(
        self,
        model_id: str,
        device: str,
        inference_steps: int,
        guidance_scale: float,
        negative_prompt: str,
    ) -> None:
        try:
            import torch
            from diffusers import DiffusionPipeline
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "diffusers and torch are required for diffusion backgrounds. "
                f"{DIFFUSION_INSTALL_HINT}"
            ) from exc

        dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
        self.pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
        self.device = device
        self.steps = inference_steps
        self.guidance_scale = guidance_scale
        self.negative_prompt = negative_prompt

        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()

        if device == "cuda":
            self.pipe.to("cuda")
        elif device == "mps":
            self.pipe.to("mps")
        else:
            self.pipe.to("cpu")

        self.pipe.set_progress_bar_config(disable=True)

    def generate(self, prompt: str, width: int, height: int, seed: int | None = None) -> np.ndarray:
        import torch

        generator = None
        if seed is not None:
            generator_device = self.device if self.device in {"cuda", "mps"} else "cpu"
            generator = torch.Generator(device=generator_device).manual_seed(seed)

        image = self.pipe(
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance_scale,
            width=width,
            height=height,
            generator=generator,
        ).images[0]

        arr = np.array(image.convert("RGB"), dtype=np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


@dataclass
class SyntheticConfig:
    dataset_yaml: Path
    output_dir: Path
    synthetic_count: int = 2000
    image_size: int = 960
    min_objects_per_image: int = 1
    max_objects_per_image: int = 4
    max_crops_per_class: int = 30
    min_crop_size: int = 20
    seed: int = 42
    mode: str = "auto"  # auto | diffusion | procedural
    diffusion_model_id: str = "stabilityai/stable-diffusion-2-1-base"
    prompt_template: str = "high-altitude aerial photography of clouded sky, realistic, high detail"
    negative_prompt: str = "aircraft, plane, jet, warplane, helicopter, text, watermark, logo"
    guidance_scale: float = 6.5
    diffusion_steps: int = 30
    device: str = "auto"
    workspace_root: Path | None = None
    allow_fallback: bool = True
    fallback_mode: str = "procedural"


def _build_crop_bank(
    train_images_dir: Path,
    train_labels_dir: Path,
    class_names: list[str],
    bank_dir: Path,
    max_crops_per_class: int,
    min_crop_size: int,
) -> list[dict[str, Any]]:
    ensure_dir(bank_dir)
    crops_per_class = {idx: 0 for idx in range(len(class_names))}
    records: list[dict[str, Any]] = []

    label_files = list(train_labels_dir.glob("*.txt"))
    random.shuffle(label_files)

    for label_path in tqdm(label_files, desc="Building crop bank"):
        if all(v >= max_crops_per_class for v in crops_per_class.values()):
            break

        stem = label_path.stem
        image_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            candidate = train_images_dir / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break
            candidate = train_images_dir / f"{stem}{ext.upper()}"
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        img_h, img_w = image.shape[:2]
        annotations = _read_label_file(label_path)
        for ann_idx, (class_id, xc, yc, w, h) in enumerate(annotations):
            if class_id not in crops_per_class:
                continue
            if crops_per_class[class_id] >= max_crops_per_class:
                continue

            x1, y1, x2, y2 = _yolo_to_xyxy(xc, yc, w, h, img_w, img_h)
            crop = image[y1:y2, x1:x2]
            ch, cw = crop.shape[:2]
            if ch < min_crop_size or cw < min_crop_size:
                continue

            class_dir = ensure_dir(bank_dir / f"{class_id:03d}_{class_names[class_id]}")
            crop_name = f"{stem}_{ann_idx}.png"
            crop_path = class_dir / crop_name
            cv2.imwrite(str(crop_path), crop)

            crops_per_class[class_id] += 1
            records.append(
                {
                    "path": str(crop_path.resolve()),
                    "class_id": class_id,
                    "class_name": class_names[class_id],
                    "width": int(cw),
                    "height": int(ch),
                }
            )

    return records


def _pick_position(
    bg_w: int,
    bg_h: int,
    obj_w: int,
    obj_h: int,
    existing_boxes: list[tuple[int, int, int, int]],
    max_iou: float = 0.25,
) -> tuple[int, int, int, int] | None:
    if obj_w >= bg_w or obj_h >= bg_h:
        return None

    for _ in range(40):
        x1 = random.randint(0, bg_w - obj_w - 1)
        y1 = random.randint(0, bg_h - obj_h - 1)
        box = (x1, y1, x1 + obj_w, y1 + obj_h)
        if all(_bbox_iou(box, existing) <= max_iou for existing in existing_boxes):
            return box
    return None


def _paste_object(background: np.ndarray, crop: np.ndarray, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    target_h = y2 - y1
    target_w = x2 - x1
    resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    mask = np.ones((target_h, target_w), dtype=np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2.0)
    alpha = np.expand_dims(np.clip(mask, 0.0, 1.0), axis=-1)

    roi = background[y1:y2, x1:x2].astype(np.float32)
    fg = resized.astype(np.float32)
    mixed = alpha * fg + (1.0 - alpha) * roi
    background[y1:y2, x1:x2] = mixed.astype(np.uint8)


def _build_background_generator(
    config: SyntheticConfig,
    device: str,
) -> tuple[DiffusionBackgroundGenerator | None, str, str | None, dict[str, Any]]:
    requested_mode = _normalize_mode(config.mode)
    fallback_mode = _normalize_fallback_mode(config.fallback_mode)
    dependency_status = _diffusion_dependency_status()

    if requested_mode == "procedural":
        return None, "procedural", None, dependency_status

    if requested_mode == "auto" and device != "cuda":
        return (
            None,
            fallback_mode,
            f"Auto mode selected `{fallback_mode}` on non-CUDA device `{device}` to keep augmentation lightweight.",
            dependency_status,
        )

    if not dependency_status["available"]:
        message = (
            "Diffusion dependencies are unavailable. "
            f"{dependency_status['error']}. {dependency_status['install_hint']}"
        )
        if requested_mode == "diffusion" and not config.allow_fallback:
            raise RuntimeError(message)
        return None, fallback_mode, message, dependency_status

    try:
        generator = DiffusionBackgroundGenerator(
            model_id=config.diffusion_model_id,
            device=device,
            inference_steps=config.diffusion_steps,
            guidance_scale=config.guidance_scale,
            negative_prompt=config.negative_prompt,
        )
        return generator, "diffusion", None, dependency_status
    except Exception as exc:
        message = f"Diffusion background initialization failed: {type(exc).__name__}: {exc}"
        if requested_mode == "diffusion" and not config.allow_fallback:
            raise RuntimeError(f"{message}. {DIFFUSION_INSTALL_HINT}") from exc
        return None, fallback_mode, f"{message}. Falling back to `{fallback_mode}`.", dependency_status


def generate_augmented_dataset(config: SyntheticConfig) -> dict[str, Any]:
    seed_metadata = seed_everything(config.seed)

    workspace_root = get_workspace_root(config.workspace_root)
    dataset_yaml_path = resolve_workspace_path(config.dataset_yaml, workspace_root)
    output_dir = resolve_workspace_path(config.output_dir, workspace_root)

    source_yaml = read_yaml(dataset_yaml_path)
    yaml_parent = dataset_yaml_path.parent
    source_root = _resolve_path(yaml_parent, str(source_yaml.get("path", ".")))

    train_images_dir = _resolve_path(source_root, str(source_yaml["train"]))
    if train_images_dir.suffix.lower() == ".txt":
        raise ValueError("Expected train split as directory, got txt list. Use base dataset.yaml from prepare step.")
    train_labels_dir = train_images_dir.parent / "labels"

    val_images_dir = _resolve_path(source_root, str(source_yaml["val"]))

    _test_key_missing = "test" not in source_yaml or source_yaml["test"] is None
    if _test_key_missing:
        print(
            "[WARN] synthetic_augmentation: 'test' split not found in source dataset.yaml. "
            "Falling back to 'val' for the test split of the augmented dataset. "
            "This means val == test in the augmented dataset.yaml — be careful about data leakage."
        )
    test_images_dir = _resolve_path(source_root, str(source_yaml.get("test", source_yaml["val"])))

    class_names = source_yaml["names"]
    if isinstance(class_names, dict):
        class_names = sorted_name_values(class_names)

    output_dir = ensure_dir(output_dir)
    resolved_config = asdict(config)
    resolved_config["dataset_yaml"] = str(dataset_yaml_path.resolve())
    resolved_config["output_dir"] = str(output_dir.resolve())
    resolved_config["workspace_root"] = str(workspace_root.resolve())
    write_yaml(output_dir / "resolved_config.yaml", resolved_config)
    write_json(
        output_dir / "runtime.json",
        collect_runtime_metadata(
            workspace_root=workspace_root,
            extra={
                "task": "synthetic_augmentation",
                "dataset_yaml": str(dataset_yaml_path.resolve()),
                "output_dir": str(output_dir.resolve()),
            },
        ),
    )
    write_json(output_dir / "seed_plan.json", seed_metadata)
    synth_images_dir = ensure_dir(output_dir / "synthetic_train" / "images")
    synth_labels_dir = ensure_dir(output_dir / "synthetic_train" / "labels")
    crop_bank_dir = ensure_dir(output_dir / "crop_bank")

    crop_bank = _build_crop_bank(
        train_images_dir=train_images_dir,
        train_labels_dir=train_labels_dir,
        class_names=class_names,
        bank_dir=crop_bank_dir,
        max_crops_per_class=config.max_crops_per_class,
        min_crop_size=config.min_crop_size,
    )
    if not crop_bank:
        raise RuntimeError("Failed to build crop bank; no valid crops were extracted.")

    device = resolve_device(config.device)
    bg_generator, effective_mode, fallback_reason, dependency_status = _build_background_generator(config, device)
    synth_records: list[dict[str, Any]] = []

    for index in tqdm(range(config.synthetic_count), desc=f"Generating {effective_mode} synthetic images"):
        image_id = f"synthetic_{index:06d}"
        image_path = synth_images_dir / f"{image_id}.jpg"
        label_path = synth_labels_dir / f"{image_id}.txt"

        if bg_generator is not None:
            background = bg_generator.generate(
                prompt=config.prompt_template,
                width=config.image_size,
                height=config.image_size,
                seed=config.seed + index,
            )
        else:
            background = _procedural_sky(width=config.image_size, height=config.image_size)

        bboxes: list[tuple[int, int, int, int]] = []
        label_lines: list[str] = []
        instance_count = random.randint(config.min_objects_per_image, config.max_objects_per_image)

        for _ in range(instance_count):
            crop_meta = random.choice(crop_bank)
            crop = cv2.imread(crop_meta["path"])
            if crop is None:
                continue

            ch, cw = crop.shape[:2]
            scale = random.uniform(0.45, 1.25)
            target_w = max(18, int(cw * scale))
            target_h = max(18, int(ch * scale))
            target_w = min(target_w, config.image_size // 2)
            target_h = min(target_h, config.image_size // 2)

            placed = _pick_position(
                bg_w=config.image_size,
                bg_h=config.image_size,
                obj_w=target_w,
                obj_h=target_h,
                existing_boxes=bboxes,
                max_iou=0.2,
            )
            if placed is None:
                continue

            _paste_object(background, crop, placed)
            bboxes.append(placed)
            xc, yc, w, h = _xyxy_to_yolo(*placed, img_w=config.image_size, img_h=config.image_size)
            label_lines.append(f"{crop_meta['class_id']} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        if not label_lines:
            continue

        cv2.imwrite(str(image_path), background)
        label_path.write_text("\n".join(label_lines), encoding="utf-8")
        synth_records.append(
            {
                "image": str(image_path.resolve()),
                "label": str(label_path.resolve()),
                "instances": len(label_lines),
                "background_mode": effective_mode,
            }
        )

    train_txt = output_dir / "train.txt"
    original_train_images = sorted(train_images_dir.glob("*"))
    with train_txt.open("w", encoding="utf-8") as f:
        for image in original_train_images:
            if image.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
                f.write(str(image.resolve()) + "\n")
        for record in synth_records:
            f.write(record["image"] + "\n")

    # val/test 경로는 source_root 기준 상대 경로로 저장한다.
    # 절대 경로로 저장하면 Colab 재시작·Drive 경로 변경 시 dataset.yaml이 파손된다.
    def _to_relative_or_abs(target: Path, base: Path) -> str:
        try:
            return str(target.relative_to(base))
        except ValueError:
            return str(target.resolve())

    augmented_yaml = {
        "path": str(source_root.resolve()),
        "train": str(train_txt.resolve()),  # train은 .txt 파일 (합성+원본 목록), 절대 경로 유지
        "val": _to_relative_or_abs(val_images_dir, source_root),
        "test": _to_relative_or_abs(test_images_dir, source_root),
        "nc": len(class_names),
        "names": class_names,
    }
    write_yaml(output_dir / "dataset.yaml", augmented_yaml)

    metadata = {
        "workspace_root": str(workspace_root.resolve()),
        "source_dataset_yaml": str(dataset_yaml_path.resolve()),
        "generated_at": timestamp(),
        "requested_mode": _normalize_mode(config.mode),
        "effective_mode": effective_mode,
        "allow_fallback": bool(config.allow_fallback),
        "fallback_mode": _normalize_fallback_mode(config.fallback_mode),
        "fallback_reason": fallback_reason,
        "device": device,
        "synthetic_count_requested": config.synthetic_count,
        "synthetic_count_created": len(synth_records),
        "image_size": config.image_size,
        "diffusion_model_id": config.diffusion_model_id if effective_mode == "diffusion" else None,
        "diffusion_dependency_available": dependency_status["available"],
        "diffusion_dependency_error": dependency_status["error"],
        "diffusion_install_hint": dependency_status["install_hint"],
        "crop_bank_size": len(crop_bank),
        "output_dataset_yaml": str((output_dir / "dataset.yaml").resolve()),
    }
    write_json(output_dir / "metadata.json", metadata)
    write_json(output_dir / "synthetic_records.json", synth_records)

    return metadata


def _load_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    return read_yaml(config_path) or {}


def _pick_value(cli_value: Any, config: dict[str, Any], key: str, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    return config.get(key, default)


def _default_output_dir(mode: str) -> Path:
    return Path("data/processed") / f"augmented_{_normalize_mode(mode)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic data for detection with optional diffusion fallback")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config for synthetic augmentation")
    parser.add_argument("--workspace-root", type=Path, default=None, help="Resolve relative dataset/output paths against this workspace root")
    parser.add_argument("--dataset-yaml", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--synthetic-count", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--min-objects-per-image", type=int, default=None)
    parser.add_argument("--max-objects-per-image", type=int, default=None)
    parser.add_argument("--max-crops-per-class", type=int, default=None)
    parser.add_argument("--min-crop-size", type=int, default=None)
    parser.add_argument("--mode", type=str, choices=VALID_SYNTHETIC_MODES, default=None)
    parser.add_argument("--disable-diffusion", action="store_true", help="Force procedural mode even if diffusion dependencies are installed")
    parser.add_argument("--strict-diffusion", action="store_true", help="Fail instead of falling back when diffusion cannot be used")
    parser.add_argument("--fallback-mode", type=str, choices=VALID_FALLBACK_MODES, default=None)
    parser.add_argument("--diffusion-model-id", type=str, default=None)
    parser.add_argument("--prompt-template", type=str, default=None)
    parser.add_argument("--negative-prompt", type=str, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--diffusion-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    file_cfg = _load_config(args.config)

    requested_mode = _pick_value(args.mode, file_cfg, "mode", "auto")
    if args.disable_diffusion:
        requested_mode = "procedural"

    allow_fallback = bool(_pick_value(None, file_cfg, "allow_fallback", True))
    if args.strict_diffusion:
        allow_fallback = False

    fallback_mode = _pick_value(args.fallback_mode, file_cfg, "fallback_mode", "procedural")
    output_dir = _pick_value(args.output_dir, file_cfg, "output_dir", _default_output_dir(requested_mode))

    meta = generate_augmented_dataset(
        SyntheticConfig(
            dataset_yaml=Path(_pick_value(args.dataset_yaml, file_cfg, "dataset_yaml", Path("data/processed/yolo_dataset/dataset.yaml"))),
            output_dir=Path(output_dir),
            synthetic_count=int(_pick_value(args.synthetic_count, file_cfg, "synthetic_count", 2000)),
            image_size=int(_pick_value(args.image_size, file_cfg, "image_size", 960)),
            min_objects_per_image=int(_pick_value(args.min_objects_per_image, file_cfg, "min_objects_per_image", 1)),
            max_objects_per_image=int(_pick_value(args.max_objects_per_image, file_cfg, "max_objects_per_image", 4)),
            max_crops_per_class=int(_pick_value(args.max_crops_per_class, file_cfg, "max_crops_per_class", 30)),
            min_crop_size=int(_pick_value(args.min_crop_size, file_cfg, "min_crop_size", 20)),
            seed=int(_pick_value(args.seed, file_cfg, "seed", 42)),
            mode=requested_mode,
            diffusion_model_id=str(_pick_value(args.diffusion_model_id, file_cfg, "diffusion_model_id", "stabilityai/stable-diffusion-2-1-base")),
            prompt_template=str(_pick_value(args.prompt_template, file_cfg, "prompt_template", "high-altitude aerial photography of clouded sky, realistic, high detail")),
            negative_prompt=str(_pick_value(args.negative_prompt, file_cfg, "negative_prompt", "aircraft, plane, jet, warplane, helicopter, text, watermark, logo")),
            guidance_scale=float(_pick_value(args.guidance_scale, file_cfg, "guidance_scale", 6.5)),
            diffusion_steps=int(_pick_value(args.diffusion_steps, file_cfg, "diffusion_steps", 30)),
            device=str(_pick_value(args.device, file_cfg, "device", "auto")),
            workspace_root=args.workspace_root,
            allow_fallback=allow_fallback,
            fallback_mode=fallback_mode,
        )
    )
    print("Synthetic dataset ready")
    for key, value in meta.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
