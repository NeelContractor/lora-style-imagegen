from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from lora_style_imagegen.dataset_prep import PrepConfig, prepare_dataset
from lora_style_imagegen.train import LORA_WEIGHT_FILE, main

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dry_run_exports_safetensors_without_base_model(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    dataset_dir = tmp_path / "processed"
    output_dir = tmp_path / "output"
    raw_dir.mkdir()
    for index in range(15):
        Image.new(
            "RGB",
            (64, 64),
            ((index * 7) % 256, (index * 11) % 256, (index * 13) % 256),
        ).save(raw_dir / f"source_{index:02d}.png")
    prepare_dataset(
        PrepConfig(
            raw_dir=raw_dir,
            output_dir=dataset_dir,
            min_images=15,
            max_images=30,
        )
    )

    exit_code = main(
        [
            "--config",
            str(PROJECT_ROOT / "configs/train_lora.yaml"),
            "--dataset-dir",
            str(dataset_dir),
            "--output-dir",
            str(output_dir),
            "--dry-run",
            "--report-to",
            "none",
        ]
    )

    assert exit_code == 0
    artifact_dir = output_dir / "dry-run/lora"
    assert (artifact_dir / LORA_WEIGHT_FILE).stat().st_size > 0
    metadata = json.loads((artifact_dir / "lora_metadata.json").read_text(encoding="utf-8"))
    assert metadata["dry_run"] is True
    assert metadata["dataset_images"] == 15
    assert metadata["trigger_token"] == "sks style"
    assert metadata["rank"] == 16
    assert (output_dir / "dry-run/dry_run_report.json").is_file()


def test_cli_rejects_invalid_step_override(tmp_path: Path) -> None:
    exit_code = main(
        [
            "--config",
            str(PROJECT_ROOT / "configs/train_lora.yaml"),
            "--dataset-dir",
            str(tmp_path / "missing"),
            "--output-dir",
            str(tmp_path / "output"),
            "--max-train-steps",
            "0",
        ]
    )

    assert exit_code == 1
