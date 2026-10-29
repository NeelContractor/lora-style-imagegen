from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from string import Formatter
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
IMAGE_FORMATS = ("png", "jpg")
DEFAULT_TRIGGER_TOKEN = "sks style"
DEFAULT_CAPTION_TEMPLATE = "{trigger}"
DEFAULT_IMAGE_SIZE = 512
DEFAULT_MIN_IMAGES = 15
DEFAULT_MAX_IMAGES = 30
IMAGES_DIRNAME = "images"
METADATA_FILENAME = "metadata.jsonl"
TEMPLATE_FILENAME = "caption_template.txt"
INFO_FILENAME = "dataset_info.json"


class DatasetError(Exception):
    pass


class DatasetValidationError(DatasetError):
    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("\n".join(self.errors))


@dataclass(frozen=True)
class PrepConfig:
    raw_dir: Path
    output_dir: Path
    trigger_token: str = DEFAULT_TRIGGER_TOKEN
    caption_template: str | None = None
    image_size: int = DEFAULT_IMAGE_SIZE
    min_images: int = DEFAULT_MIN_IMAGES
    max_images: int = DEFAULT_MAX_IMAGES
    image_format: str = "png"
    force: bool = False


@dataclass(frozen=True)
class PrepResult:
    output_dir: Path
    image_count: int
    trigger_token: str
    metadata_path: Path
    template_path: Path


@dataclass(frozen=True)
class ValidationReport:
    dataset_dir: Path
    image_count: int
    trigger_token: str
    image_size: int


def discover_images(raw_dir: Path) -> list[Path]:
    if not raw_dir.is_dir():
        raise DatasetError(f"Raw image directory does not exist: {raw_dir}")

    images = [
        path
        for path in raw_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not any(part.startswith(".") for part in path.relative_to(raw_dir).parts)
    ]
    return sorted(images, key=lambda path: path.relative_to(raw_dir).as_posix().casefold())


def prepare_dataset(config: PrepConfig) -> PrepResult:
    _validate_config(config)

    raw_images = discover_images(config.raw_dir)
    _validate_image_count(len(raw_images), config.raw_dir, config.min_images, config.max_images)

    raw_resolved = config.raw_dir.resolve()
    output_resolved = config.output_dir.resolve()
    if raw_resolved == output_resolved:
        raise DatasetError("Raw and output directories must be different")

    if config.output_dir.exists() and any(config.output_dir.iterdir()) and not config.force:
        raise DatasetError(
            f"Output directory is not empty: {config.output_dir}. "
            "Use --force to rebuild the processed images."
        )

    images_dir = config.output_dir / IMAGES_DIRNAME
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.force and images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    template = _load_or_create_template(config)
    caption = render_caption(template, config.trigger_token)
    image_format = config.image_format.lower()
    extension = "jpg" if image_format == "jpg" else "png"

    records: list[dict[str, str]] = []
    for index, source_path in enumerate(raw_images, start=1):
        output_name = f"image_{index:04d}.{extension}"
        output_path = images_dir / output_name
        _prepare_image(source_path, output_path, config.image_size, image_format)
        records.append(
            {
                "file_name": f"{IMAGES_DIRNAME}/{output_name}",
                "text": caption,
            }
        )

    metadata_path = config.output_dir / METADATA_FILENAME
    _write_jsonl(metadata_path, records)

    info = {
        "trigger_token": config.trigger_token,
        "caption_template": template,
        "image_count": len(records),
        "image_size": config.image_size,
        "image_format": image_format,
    }
    _write_text(
        config.output_dir / INFO_FILENAME,
        json.dumps(info, indent=2, ensure_ascii=False) + "\n",
    )

    validate_dataset(
        config.output_dir,
        trigger_token=config.trigger_token,
        image_size=config.image_size,
        min_images=config.min_images,
        max_images=config.max_images,
    )

    return PrepResult(
        output_dir=config.output_dir,
        image_count=len(records),
        trigger_token=config.trigger_token,
        metadata_path=metadata_path,
        template_path=config.output_dir / TEMPLATE_FILENAME,
    )


def render_caption(template: str, trigger_token: str) -> str:
    try:
        fields = [field for _, field, _, _ in Formatter().parse(template) if field is not None]
    except ValueError as exc:
        raise DatasetError(f"Invalid caption template: {exc}") from exc

    unsupported = sorted({field for field in fields if field != "trigger"})
    if unsupported:
        names = ", ".join(repr(field) for field in unsupported)
        raise DatasetError(
            f"Caption template uses unsupported fields: {names}. Only {{trigger}} is supported."
        )

    caption = " ".join(template.format(trigger=trigger_token).split())
    if not caption:
        raise DatasetError("Caption template produced an empty caption")
    if trigger_token not in caption:
        raise DatasetError(
            "Caption template must include {trigger}; otherwise the trigger token is missing"
        )
    return caption


def validate_dataset(
    dataset_dir: Path,
    *,
    trigger_token: str | None = None,
    image_size: int | None = None,
    min_images: int = DEFAULT_MIN_IMAGES,
    max_images: int = DEFAULT_MAX_IMAGES,
) -> ValidationReport:
    if not dataset_dir.is_dir():
        raise DatasetValidationError([f"Dataset directory does not exist: {dataset_dir}"])
    if min_images < 1 or max_images < min_images:
        raise DatasetValidationError(["Invalid validation image-count bounds"])

    errors: list[str] = []
    info = _read_info(dataset_dir, errors)
    resolved_trigger = trigger_token or _optional_string(info.get("trigger_token"))
    resolved_size = image_size or _optional_positive_int(info.get("image_size"))

    if resolved_trigger is None:
        errors.append(
            "Trigger token is unknown; pass --trigger-token or regenerate dataset_info.json"
        )
    if resolved_size is None:
        errors.append("Image size is unknown; pass --size or regenerate dataset_info.json")

    info_trigger = _optional_string(info.get("trigger_token"))
    if trigger_token and info_trigger and trigger_token != info_trigger:
        errors.append(
            f"Trigger token mismatch: requested {trigger_token!r}, "
            f"dataset_info.json has {info_trigger!r}"
        )
    info_size = _optional_positive_int(info.get("image_size"))
    if image_size and info_size and image_size != info_size:
        errors.append(
            f"Image size mismatch: requested {image_size}, dataset_info.json has {info_size}"
        )

    metadata_path = dataset_dir / METADATA_FILENAME
    records = _read_metadata(metadata_path, errors)

    if not (min_images <= len(records) <= max_images):
        errors.append(
            f"Expected between {min_images} and {max_images} metadata entries, found {len(records)}"
        )

    info_count = _optional_positive_int(info.get("image_count"))
    if info_count is not None and info_count != len(records):
        errors.append(f"dataset_info.json reports {info_count} images, metadata has {len(records)}")

    seen_names: set[str] = set()
    referenced: set[str] = set()
    dataset_resolved = dataset_dir.resolve()
    for line_number, record in records:
        file_name = record.get("file_name")
        text = record.get("text")

        if not isinstance(file_name, str) or not file_name:
            errors.append(f"{METADATA_FILENAME}:{line_number}: missing file_name")
            continue
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{METADATA_FILENAME}:{line_number}: missing text")
            continue
        if resolved_trigger is not None and resolved_trigger not in text:
            errors.append(
                f"{METADATA_FILENAME}:{line_number}: caption does not contain trigger "
                f"{resolved_trigger!r}"
            )

        relative_path = PurePosixPath(file_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            errors.append(f"{METADATA_FILENAME}:{line_number}: unsafe file_name {file_name!r}")
            continue
        if file_name in seen_names:
            errors.append(f"{METADATA_FILENAME}:{line_number}: duplicate file_name {file_name!r}")
            continue
        seen_names.add(file_name)
        referenced.add(file_name)

        image_path = dataset_dir / Path(*relative_path.parts)
        if not image_path.resolve().is_relative_to(dataset_resolved):
            errors.append(f"{METADATA_FILENAME}:{line_number}: file_name escapes dataset directory")
            continue
        if not image_path.is_file():
            errors.append(f"{METADATA_FILENAME}:{line_number}: missing image {file_name!r}")
            continue

        try:
            with Image.open(image_path) as image:
                image.load()
                actual_size = image.size
                mode = image.mode
        except (OSError, UnidentifiedImageError) as exc:
            errors.append(f"{file_name}: cannot read image ({exc})")
            continue

        if resolved_size is not None and actual_size != (resolved_size, resolved_size):
            errors.append(
                f"{file_name}: expected {resolved_size}x{resolved_size}, "
                f"found {actual_size[0]}x{actual_size[1]}"
            )
        if mode != "RGB":
            errors.append(f"{file_name}: expected RGB mode, found {mode}")

    images_dir = dataset_dir / IMAGES_DIRNAME
    if images_dir.is_dir():
        disk_images = {
            path.relative_to(dataset_dir).as_posix()
            for path in images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        for extra in sorted(disk_images - referenced):
            errors.append(f"Unreferenced image in dataset: {extra}")

    if errors:
        raise DatasetValidationError(errors)

    assert resolved_trigger is not None
    assert resolved_size is not None
    return ValidationReport(
        dataset_dir=dataset_dir,
        image_count=len(records),
        trigger_token=resolved_trigger,
        image_size=resolved_size,
    )


def _validate_config(config: PrepConfig) -> None:
    if not config.trigger_token.strip():
        raise DatasetError("Trigger token must not be empty")
    if "{" in config.trigger_token or "}" in config.trigger_token:
        raise DatasetError("Trigger token must not contain braces")
    if config.image_size < 64:
        raise DatasetError("Image size must be at least 64 pixels")
    if config.min_images < 1 or config.max_images < config.min_images:
        raise DatasetError("Invalid image-count bounds")
    if config.image_format.lower() not in IMAGE_FORMATS:
        raise DatasetError(f"Image format must be one of: {', '.join(IMAGE_FORMATS)}")


def _validate_image_count(count: int, raw_dir: Path, minimum: int, maximum: int) -> None:
    if count < minimum:
        raise DatasetError(f"Found {count} images in {raw_dir}; at least {minimum} are required")
    if count > maximum:
        raise DatasetError(f"Found {count} images in {raw_dir}; at most {maximum} are allowed")


def _load_or_create_template(config: PrepConfig) -> str:
    template_path = config.output_dir / TEMPLATE_FILENAME

    if config.caption_template is not None:
        template = config.caption_template.strip()
        if not template:
            raise DatasetError("Caption template must not be empty")
        _write_text(template_path, template + "\n")
        return template

    if template_path.is_file():
        template = template_path.read_text(encoding="utf-8").strip()
        if not template:
            raise DatasetError(f"Caption template is empty: {template_path}")
        return template

    _write_text(template_path, DEFAULT_CAPTION_TEMPLATE + "\n")
    return DEFAULT_CAPTION_TEMPLATE


def _prepare_image(
    source_path: Path,
    output_path: Path,
    image_size: int,
    image_format: str,
) -> None:
    try:
        with Image.open(source_path) as source:
            image = ImageOps.exif_transpose(source)
            image.load()
            rgb_image = _to_rgb(image)
            resized = ImageOps.fit(
                rgb_image,
                (image_size, image_size),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )
            if image_format == "jpg":
                resized.save(output_path, format="JPEG", quality=95, optimize=True)
            else:
                resized.save(output_path, format="PNG", optimize=True)
    except (OSError, UnidentifiedImageError) as exc:
        raise DatasetError(f"Cannot process image {source_path}: {exc}") from exc


def _to_rgb(image: Image.Image) -> Image.Image:
    has_alpha = image.mode in {"LA", "RGBA"} or (image.mode == "P" and "transparency" in image.info)
    if not has_alpha:
        return image.convert("RGB")

    rgba = image.convert("RGBA")
    background = Image.new("RGB", rgba.size, "white")
    background.paste(rgba, mask=rgba.getchannel("A"))
    return background


def _read_info(dataset_dir: Path, errors: list[str]) -> dict[str, Any]:
    info_path = dataset_dir / INFO_FILENAME
    if not info_path.is_file():
        errors.append(f"Missing {INFO_FILENAME}")
        return {}

    try:
        data = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{INFO_FILENAME}: cannot read JSON ({exc})")
        return {}

    if not isinstance(data, dict):
        errors.append(f"{INFO_FILENAME}: expected a JSON object")
        return {}
    return data


def _read_metadata(
    metadata_path: Path,
    errors: list[str],
) -> list[tuple[int, dict[str, Any]]]:
    if not metadata_path.is_file():
        errors.append(f"Missing {METADATA_FILENAME}")
        return []

    try:
        lines = metadata_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"{METADATA_FILENAME}: cannot read file ({exc})")
        return []

    records: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            errors.append(f"{METADATA_FILENAME}:{line_number}: blank line")
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{METADATA_FILENAME}:{line_number}: invalid JSON ({exc})")
            continue
        if not isinstance(record, dict):
            errors.append(f"{METADATA_FILENAME}:{line_number}: expected a JSON object")
            continue
        records.append((line_number, record))
    return records


def _optional_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _optional_positive_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _write_jsonl(path: Path, records: Sequence[dict[str, str]]) -> None:
    content = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    _write_text(path, content)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resize and caption a small Stable Diffusion LoRA dataset."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--trigger-token", default=None)
    parser.add_argument(
        "--caption-template",
        default=None,
        help="Caption template using {trigger}; written to caption_template.txt",
    )
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--min-images", type=int, default=DEFAULT_MIN_IMAGES)
    parser.add_argument("--max-images", type=int, default=DEFAULT_MAX_IMAGES)
    parser.add_argument("--format", dest="image_format", choices=IMAGE_FORMATS, default="png")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild processed images when the output directory is not empty",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing processed dataset without rebuilding it",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        if args.validate_only:
            report = validate_dataset(
                args.output_dir,
                trigger_token=args.trigger_token,
                image_size=args.size,
                min_images=args.min_images,
                max_images=args.max_images,
            )
            print(
                f"Dataset OK: {report.image_count} images, "
                f"{report.image_size}x{report.image_size}, "
                f"trigger {report.trigger_token!r}"
            )
            return 0

        result = prepare_dataset(
            PrepConfig(
                raw_dir=args.raw_dir,
                output_dir=args.output_dir,
                trigger_token=args.trigger_token or DEFAULT_TRIGGER_TOKEN,
                caption_template=args.caption_template,
                image_size=args.size or DEFAULT_IMAGE_SIZE,
                min_images=args.min_images,
                max_images=args.max_images,
                image_format=args.image_format,
                force=args.force,
            )
        )
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Prepared {result.image_count} images at {result.output_dir}")
    print(f"Metadata: {result.metadata_path}")
    print(f"Edit captions in: {result.template_path}, then rerun with --force")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
