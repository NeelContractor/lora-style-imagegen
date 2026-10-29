from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from lora_style_imagegen.dataset_prep import (
    DatasetError,
    DatasetValidationError,
    PrepConfig,
    discover_images,
    main,
    prepare_dataset,
    validate_dataset,
)


def _seed_images(
    raw_dir: Path,
    count: int,
    *,
    size: tuple[int, int] = (800, 600),
    image_format: str = "JPEG",
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    extension = "jpg" if image_format == "JPEG" else "png"
    for index in range(count):
        color = ((index * 37) % 256, (index * 71) % 256, (index * 113) % 256)
        image_size = [(800, 600), (600, 800), (512, 512)][index % 3] if count > 2 else size
        Image.new("RGB", image_size, color).save(
            raw_dir / f"source_{index:03d}.{extension}",
            format=image_format,
        )


def _read_jsonl(path: Path) -> list[dict[str, str]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_discovers_avif_images(tmp_path: Path) -> None:
    source = tmp_path / "source.avif"
    source.write_bytes(b"avif")

    assert discover_images(tmp_path) == [source]


def _prepare(
    raw_dir: Path,
    output_dir: Path,
    *,
    count: int = 1,
    min_images: int = 1,
    max_images: int = 30,
    **kwargs: object,
):
    _seed_images(raw_dir, count)
    return prepare_dataset(
        PrepConfig(
            raw_dir=raw_dir,
            output_dir=output_dir,
            min_images=min_images,
            max_images=max_images,
            **kwargs,
        )
    )


def test_prepare_creates_square_rgb_images_and_metadata(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"

    result = _prepare(raw_dir, output_dir, count=15, min_images=15, max_images=30)

    assert result.image_count == 15
    assert result.trigger_token == "sks style"
    assert result.template_path.read_text(encoding="utf-8").strip() == "{trigger}"

    records = _read_jsonl(output_dir / "metadata.jsonl")
    assert len(records) == 15
    assert all(record["text"] == "sks style" for record in records)
    assert records[0]["file_name"] == "images/image_0001.png"
    assert records[-1]["file_name"] == "images/image_0015.png"

    for record in records:
        image_path = output_dir / record["file_name"]
        with Image.open(image_path) as image:
            assert image.size == (512, 512)
            assert image.mode == "RGB"
            assert image.format == "PNG"

    info = json.loads((output_dir / "dataset_info.json").read_text(encoding="utf-8"))
    assert info["trigger_token"] == "sks style"
    assert info["image_count"] == 15
    assert info["image_size"] == 512
    assert info["image_format"] == "png"


def test_prepare_uses_custom_template_and_reuses_edited_template(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"

    _prepare(
        raw_dir,
        output_dir,
        count=2,
        caption_template="{trigger}, thick ink on rough paper",
    )
    records = _read_jsonl(output_dir / "metadata.jsonl")
    assert all(record["text"] == "sks style, thick ink on rough paper" for record in records)

    template_path = output_dir / "caption_template.txt"
    template_path.write_text("{trigger} painted with a palette knife\n", encoding="utf-8")

    prepare_dataset(
        PrepConfig(
            raw_dir=raw_dir,
            output_dir=output_dir,
            min_images=1,
            max_images=30,
            force=True,
        )
    )

    records = _read_jsonl(output_dir / "metadata.jsonl")
    assert all(record["text"] == "sks style painted with a palette knife" for record in records)


@pytest.mark.parametrize(
    ("count", "message"),
    [(14, "at least 15"), (31, "at most 30")],
)
def test_prepare_rejects_image_counts_outside_default_bounds(
    tmp_path: Path,
    count: int,
    message: str,
) -> None:
    raw_dir = tmp_path / "raw"
    _seed_images(raw_dir, count)

    with pytest.raises(DatasetError, match=message):
        prepare_dataset(PrepConfig(raw_dir=raw_dir, output_dir=tmp_path / "processed"))


def test_prepare_requires_force_for_nonempty_output(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    _prepare(raw_dir, output_dir)

    with pytest.raises(DatasetError, match="not empty"):
        prepare_dataset(PrepConfig(raw_dir=raw_dir, output_dir=output_dir, min_images=1))


def test_prepare_composites_transparency_onto_white(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()
    Image.new("RGBA", (200, 120), (255, 0, 0, 0)).save(raw_dir / "alpha.png")

    prepare_dataset(PrepConfig(raw_dir=raw_dir, output_dir=output_dir, min_images=1))

    with Image.open(output_dir / "images/image_0001.png") as image:
        assert image.mode == "RGB"
        assert image.getpixel((0, 0)) == (255, 255, 255)


def test_prepare_rejects_template_without_trigger(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _seed_images(raw_dir, 1)

    with pytest.raises(DatasetError, match="must include"):
        prepare_dataset(
            PrepConfig(
                raw_dir=raw_dir,
                output_dir=tmp_path / "processed",
                min_images=1,
                caption_template="a nice photograph",
            )
        )


def test_prepare_can_emit_jpeg(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    _prepare(raw_dir, output_dir, image_format="jpg")

    image_path = output_dir / "images/image_0001.jpg"
    with Image.open(image_path) as image:
        assert image.format == "JPEG"
        assert image.size == (512, 512)
        assert image.mode == "RGB"


def test_validate_detects_missing_image(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    _prepare(raw_dir, output_dir)
    (output_dir / "images/image_0001.png").unlink()

    with pytest.raises(DatasetValidationError, match="missing image"):
        validate_dataset(output_dir, min_images=1)


def test_validate_detects_wrong_dimensions(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    _prepare(raw_dir, output_dir)
    Image.new("RGB", (64, 64), "blue").save(output_dir / "images/image_0001.png")

    with pytest.raises(DatasetValidationError, match="512x512"):
        validate_dataset(output_dir, min_images=1)


def test_validate_detects_missing_trigger(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    _prepare(raw_dir, output_dir)
    metadata_path = output_dir / "metadata.jsonl"
    metadata_path.write_text(
        json.dumps({"file_name": "images/image_0001.png", "text": "a plain photo"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetValidationError, match="does not contain trigger"):
        validate_dataset(output_dir, min_images=1)


def test_validate_detects_unreferenced_images(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    _prepare(raw_dir, output_dir)
    Image.new("RGB", (512, 512), "green").save(output_dir / "images/extra.png")

    with pytest.raises(DatasetValidationError, match="Unreferenced image"):
        validate_dataset(output_dir, min_images=1)


def test_cli_prepares_then_validates(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    _seed_images(raw_dir, 1)

    assert (
        main(
            [
                "--raw-dir",
                str(raw_dir),
                "--output-dir",
                str(output_dir),
                "--min-images",
                "1",
                "--max-images",
                "1",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "--output-dir",
                str(output_dir),
                "--validate-only",
                "--min-images",
                "1",
                "--max-images",
                "1",
            ]
        )
        == 0
    )
