from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import torch
from PIL import Image, ImageOps


class DataError(Exception):
    pass


@dataclass(frozen=True)
class CaptionRecord:
    file_name: str
    text: str
    path: Path


def load_caption_records(dataset_dir: Path) -> list[CaptionRecord]:
    metadata_path = dataset_dir / "metadata.jsonl"
    if not metadata_path.is_file():
        raise DataError(f"Missing metadata file: {metadata_path}")

    records: list[CaptionRecord] = []
    try:
        lines = metadata_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DataError(f"Cannot read {metadata_path}: {exc}") from exc

    dataset_resolved = dataset_dir.resolve()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise DataError(f"{metadata_path}:{line_number}: blank line")
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataError(f"{metadata_path}:{line_number}: invalid JSON ({exc})") from exc
        if not isinstance(entry, dict):
            raise DataError(f"{metadata_path}:{line_number}: expected a JSON object")

        file_name = entry.get("file_name")
        text = entry.get("text")
        if not isinstance(file_name, str) or not file_name:
            raise DataError(f"{metadata_path}:{line_number}: missing file_name")
        if not isinstance(text, str) or not text.strip():
            raise DataError(f"{metadata_path}:{line_number}: missing text")

        relative_path = PurePosixPath(file_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise DataError(f"{metadata_path}:{line_number}: unsafe file_name {file_name!r}")

        image_path = dataset_dir / Path(*relative_path.parts)
        if not image_path.resolve().is_relative_to(dataset_resolved):
            raise DataError(f"{metadata_path}:{line_number}: file_name escapes dataset directory")
        if not image_path.is_file():
            raise DataError(f"{metadata_path}:{line_number}: missing image {file_name!r}")

        records.append(CaptionRecord(file_name=file_name, text=text.strip(), path=image_path))

    if not records:
        raise DataError(f"No caption records found in {metadata_path}")
    return records


def image_to_tensor(path: Path, resolution: int) -> torch.Tensor:
    try:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            fitted = ImageOps.fit(
                image,
                (resolution, resolution),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )
            buffer = bytearray(fitted.tobytes())
    except OSError as exc:
        raise DataError(f"Cannot load image {path}: {exc}") from exc

    tensor = torch.frombuffer(buffer, dtype=torch.uint8).reshape(resolution, resolution, 3)
    return tensor.permute(2, 0, 1).float().div(127.5).sub(1.0)
