from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

import lora_style_imagegen.demo as demo
from lora_style_imagegen.demo import (
    DemoConfig,
    DemoError,
    StyleDemo,
    build_interface,
    load_demo_config,
)


def _write_adapter(adapter_dir: Path) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "pytorch_lora_weights.safetensors").write_bytes(b"test")
    metadata = {
        "base_model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "base_model_revision": None,
        "trigger_token": "vbw style",
        "weights_file": "pytorch_lora_weights.safetensors",
        "validation_prompts": ["vbw style, a fern"],
        "validation_seeds": [1234],
        "validation_guidance_scale": 7.5,
        "validation_inference_steps": 25,
        "resolution": 512,
    }
    (adapter_dir / "lora_metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )


def _config(adapter_dir: Path) -> DemoConfig:
    return DemoConfig(
        adapter_dir=adapter_dir,
        base_model="stable-diffusion-v1-5/stable-diffusion-v1-5",
        base_model_revision=None,
        trigger_token="vbw style",
        weights_file="pytorch_lora_weights.safetensors",
        adapter_name="style",
        negative_prompt="blurry",
        guidance_scale=7.5,
        num_inference_steps=25,
        width=512,
        height=512,
        device="cpu",
    )


class FakePipe:
    def __init__(self) -> None:
        self.scales: list[tuple[list[str], list[float]]] = []

    def set_adapters(self, names: list[str], adapter_weights: list[float]) -> None:
        self.scales.append((names, adapter_weights))


def test_load_demo_config_reads_adapter_metadata(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    _write_adapter(adapter_dir)

    config = load_demo_config(adapter_dir)

    assert config.adapter_dir == adapter_dir
    assert config.base_model == "stable-diffusion-v1-5/stable-diffusion-v1-5"
    assert config.trigger_token == "vbw style"
    assert config.num_inference_steps == 25
    assert config.width == 512


def test_generate_pair_uses_same_seed_and_switches_lora_scale(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pipe = FakePipe()
    style = StyleDemo(_config(tmp_path / "adapter"), pipe=pipe, device=torch.device("cpu"))
    calls: list[dict[str, object]] = []

    def fake_generate(pipe_arg, **kwargs):
        calls.append(kwargs)
        color = "white" if len(calls) == 1 else "black"
        return Image.new("RGB", (8, 8), color)

    monkeypatch.setattr(demo, "generate_image", fake_generate)

    base, styled, seed = style.generate_pair(
        "vbw style, a fern",
        "blurry",
        42,
        20,
        7.5,
        0.8,
    )

    assert seed == 42
    assert base.getpixel((0, 0)) == (255, 255, 255)
    assert styled.getpixel((0, 0)) == (0, 0, 0)
    assert pipe.scales == [
        (["style"], [0.0]),
        (["style"], [0.8]),
    ]
    assert [call["seed"] for call in calls] == [42, 42]
    assert [call["prompt"] for call in calls] == [
        "vbw style, a fern",
        "vbw style, a fern",
    ]


def test_generate_pair_lazily_loads_pipeline(tmp_path: Path, monkeypatch) -> None:
    pipe = FakePipe()
    loaded: list[dict[str, object]] = []

    def fake_load_pipeline(base_model, **kwargs):
        loaded.append({"base_model": base_model, **kwargs})
        return pipe

    def fake_load_adapter(pipe_arg, adapter_dir, **kwargs):
        loaded.append({"adapter_dir": adapter_dir, **kwargs})

    monkeypatch.setattr(demo, "load_pipeline", fake_load_pipeline)
    monkeypatch.setattr(demo, "load_lora_adapter", fake_load_adapter)
    monkeypatch.setattr(
        demo,
        "generate_image",
        lambda pipe_arg, **kwargs: Image.new("RGB", (8, 8), "white"),
    )

    style = StyleDemo(_config(tmp_path / "adapter"))
    style.generate_pair("vbw style, a fern", "blurry", 42, 20, 7.5, 1.0)

    assert loaded == [
        {
            "base_model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "revision": None,
            "device": torch.device("cpu"),
            "dtype": torch.float32,
        },
        {
            "adapter_dir": tmp_path / "adapter",
            "weights_file": "pytorch_lora_weights.safetensors",
            "adapter_name": "style",
        },
    ]


def test_generate_pair_rejects_empty_prompt(tmp_path: Path) -> None:
    style = StyleDemo(_config(tmp_path / "adapter"), pipe=FakePipe(), device=torch.device("cpu"))

    with pytest.raises(DemoError, match="Prompt"):
        style.generate_pair("   ", "blurry", 42, 20, 7.5, 1.0)


def test_build_interface_creates_blocks(tmp_path: Path) -> None:
    pytest.importorskip("gradio")
    style = StyleDemo(_config(tmp_path / "adapter"))

    ui = build_interface(style)

    assert ui is not None
