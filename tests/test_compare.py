from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

import lora_style_imagegen.compare as compare
from lora_style_imagegen.compare import (
    DEFAULT_NEGATIVE_PROMPT,
    GRID_HEADER_HEIGHT,
    GRID_LABEL_WIDTH,
    GRID_THUMB_SIZE,
    ComparisonConfig,
    ComparisonError,
    build_comparison_prompts,
    create_comparison_grid,
    load_adapter_info,
    main,
    run_comparison,
)


def _write_adapter(adapter_dir: Path, **overrides: object) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "pytorch_lora_weights.safetensors").write_bytes(b"test")
    metadata: dict[str, object] = {
        "base_model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "base_model_revision": None,
        "trigger_token": "vbw style",
        "weights_file": "pytorch_lora_weights.safetensors",
        "validation_prompts": ["vbw style, a fern"],
        "validation_seeds": [1234, 5678],
        "validation_guidance_scale": 7.5,
        "validation_inference_steps": 25,
        "resolution": 512,
    }
    metadata.update(overrides)
    (adapter_dir / "lora_metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )


def test_build_comparison_prompts_adds_untriggered_variants() -> None:
    prompts = build_comparison_prompts(
        ("vbw style, a fern", "a mushroom", "vbw style, a fern"),
        "vbw style",
    )

    assert prompts == ("vbw style, a fern", "a fern", "a mushroom")


def test_load_adapter_info_uses_metadata_defaults(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    _write_adapter(adapter_dir)

    info = load_adapter_info(adapter_dir)

    assert info.base_model == "stable-diffusion-v1-5/stable-diffusion-v1-5"
    assert info.base_model_revision is None
    assert info.trigger_token == "vbw style"
    assert info.weights_file == "pytorch_lora_weights.safetensors"
    assert info.prompts == ("vbw style, a fern",)
    assert info.seeds == (1234, 5678)
    assert info.guidance_scale == 7.5
    assert info.num_inference_steps == 25
    assert info.resolution == 512
    assert info.negative_prompt == DEFAULT_NEGATIVE_PROMPT


def test_load_adapter_info_requires_safetensors(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    _write_adapter(adapter_dir, weights_file="pytorch_lora_weights.bin")

    with pytest.raises(ComparisonError, match="safetensors"):
        load_adapter_info(adapter_dir)


def test_load_adapter_info_requires_metadata(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    with pytest.raises(ComparisonError, match="metadata"):
        load_adapter_info(adapter_dir)


def test_create_comparison_grid_dimensions(tmp_path: Path) -> None:
    prompts = ("vbw style, a fern", "a mushroom")
    seeds = (11, 22)
    image_paths: dict[tuple[str, int, int], Path] = {}
    for variant in ("base", "lora"):
        for row in range(len(prompts)):
            for seed in seeds:
                path = tmp_path / f"{variant}_{row}_{seed}.png"
                Image.new("RGB", (512, 512), "white").save(path)
                image_paths[(variant, row, seed)] = path
    output_path = tmp_path / "grid.png"

    create_comparison_grid(
        prompts=prompts,
        seeds=seeds,
        image_paths=image_paths,
        output_path=output_path,
    )

    with Image.open(output_path) as grid:
        assert grid.size == (
            GRID_LABEL_WIDTH + len(seeds) * 2 * GRID_THUMB_SIZE,
            GRID_HEADER_HEIGHT + len(prompts) * GRID_THUMB_SIZE,
        )


class FakeVae:
    dtype = torch.float16
    config = SimpleNamespace(scaling_factor=0.18215)

    def decode(self, latents: torch.Tensor, return_dict: bool) -> tuple[torch.Tensor]:
        return (torch.zeros(1, 3, 8, 8),)


class FakeImageProcessor:
    def postprocess(self, image: torch.Tensor, output_type: str) -> list[Image.Image]:
        return [Image.new("RGB", (8, 8), "white")]


class FakePipe:
    def __init__(self) -> None:
        self.vae = FakeVae()
        self.image_processor = FakeImageProcessor()
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(images=torch.ones(1, 4, 8, 8))


def test_run_comparison_writes_grid_and_manifest(tmp_path: Path, monkeypatch) -> None:
    adapter_dir = tmp_path / "adapter"
    _write_adapter(adapter_dir)
    pipe = FakePipe()
    loaded_adapters: list[Path] = []

    monkeypatch.setattr(compare, "load_pipeline", lambda *args, **kwargs: pipe)

    def load_adapter(fake_pipe, adapter_dir, *, weights_file, adapter_name) -> None:
        loaded_adapters.append((adapter_dir, weights_file, adapter_name))

    monkeypatch.setattr(compare, "load_lora_adapter", load_adapter)

    config = ComparisonConfig(
        adapter_dir=adapter_dir,
        output_dir=tmp_path / "comparison",
        base_model="stable-diffusion-v1-5/stable-diffusion-v1-5",
        base_model_revision=None,
        trigger_token="vbw style",
        weights_file="pytorch_lora_weights.safetensors",
        adapter_name="style",
        prompts=("vbw style, a fern", "a mushroom"),
        seeds=(1234,),
        negative_prompt="blurry",
        guidance_scale=7.5,
        num_inference_steps=2,
        width=512,
        height=512,
    )

    result = run_comparison(config)

    assert result.image_count == 4
    assert pipe.calls == 4
    assert loaded_adapters == [(adapter_dir, "pytorch_lora_weights.safetensors", "style")]
    assert result.grid_path.is_file()
    assert result.metadata_path.is_file()
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert len(metadata["images"]) == 4
    assert metadata["grid"] == "comparison_grid.png"
    assert {image["variant"] for image in metadata["images"]} == {"base", "lora"}


def test_main_reports_missing_adapter(tmp_path: Path) -> None:
    assert main(["--adapter-dir", str(tmp_path / "missing")]) == 1
