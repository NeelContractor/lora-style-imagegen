from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from lora_style_imagegen.data import DataError, image_to_tensor, load_caption_records
from lora_style_imagegen.dataset_prep import PrepConfig, prepare_dataset


def _prepared_dataset(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()
    Image.new("RGB", (640, 480), "orange").save(raw_dir / "source.png")
    prepare_dataset(
        PrepConfig(
            raw_dir=raw_dir,
            output_dir=output_dir,
            min_images=1,
            max_images=1,
            caption_template="{trigger}, painted on canvas",
        )
    )
    return output_dir


def test_load_caption_records(tmp_path: Path) -> None:
    dataset_dir = _prepared_dataset(tmp_path)

    records = load_caption_records(dataset_dir)

    assert len(records) == 1
    assert records[0].file_name == "images/image_0001.png"
    assert records[0].text == "sks style, painted on canvas"
    assert records[0].path.is_file()


def test_image_to_tensor_has_expected_shape_and_range(tmp_path: Path) -> None:
    dataset_dir = _prepared_dataset(tmp_path)
    record = load_caption_records(dataset_dir)[0]

    tensor = image_to_tensor(record.path, 512)

    assert tensor.shape == (3, 512, 512)
    assert tensor.dtype.is_floating_point
    assert float(tensor.min()) >= -1.0
    assert float(tensor.max()) <= 1.0


def test_rejects_path_traversal(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "processed"
    dataset_dir.mkdir()
    (dataset_dir / "metadata.jsonl").write_text(
        '{"file_name": "../secret.png", "text": "sks style"}\n',
        encoding="utf-8",
    )

    with pytest.raises(DataError, match="unsafe file_name"):
        load_caption_records(dataset_dir)
