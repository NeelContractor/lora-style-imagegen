from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import lora_style_imagegen.generation as generation
from lora_style_imagegen.generation import (
    GenerationError,
    generate_image,
    load_lora_adapter,
    load_pipeline,
    resolve_dtype,
)


class FakeVae:
    dtype = torch.float16
    config = SimpleNamespace(scaling_factor=0.18215)
    seen: torch.Tensor | None = None

    def decode(self, latents: torch.Tensor, return_dict: bool) -> tuple[torch.Tensor]:
        self.seen = latents
        return (torch.zeros(1, 3, 8, 8, requires_grad=True),)


class FakeImageProcessor:
    def postprocess(self, image: torch.Tensor, output_type: str) -> list[str]:
        assert image.requires_grad is False
        return ["decoded"]


class FakePipe:
    def __init__(self) -> None:
        self.vae = FakeVae()
        self.image_processor = FakeImageProcessor()
        self.kwargs: dict[str, object] | None = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(images=torch.ones(1, 4, 8, 8))


def test_generate_image_decodes_latents_with_reproducible_seed() -> None:
    pipe = FakePipe()

    image = generate_image(
        pipe,
        prompt="vbw style, a fern",
        negative_prompt="blurry",
        num_inference_steps=25,
        guidance_scale=7.5,
        width=512,
        height=512,
        seed=42,
        device=torch.device("cpu"),
    )

    assert image == "decoded"
    assert pipe.kwargs is not None
    assert pipe.kwargs["output_type"] == "latent"
    assert pipe.kwargs["prompt"] == "vbw style, a fern"
    assert pipe.kwargs["negative_prompt"] == "blurry"
    assert pipe.kwargs["generator"].initial_seed() == 42
    assert pipe.vae.seen is not None
    assert pipe.vae.seen.dtype == torch.float16


def test_resolve_dtype_rejects_fp16_on_cpu() -> None:
    with pytest.raises(GenerationError, match="fp16 requires CUDA"):
        resolve_dtype("fp16", torch.device("cpu"))


def test_load_pipeline_uses_safetensors_and_vae_slicing(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeScheduler:
        config = SimpleNamespace(sample_size=64)

        @staticmethod
        def from_config(config):
            return "scheduler"

    class FakeModule:
        def __init__(self) -> None:
            self.requires_grad = True

        def requires_grad_(self, requires_grad: bool) -> None:
            self.requires_grad = requires_grad

    class FakeVae(FakeModule):
        def __init__(self) -> None:
            super().__init__()
            self.sliced = False

        def enable_slicing(self) -> None:
            self.sliced = True

    class FakePipe:
        def __init__(self) -> None:
            self.scheduler = FakeScheduler()
            self.vae = FakeVae()
            self.text_encoder = FakeModule()
            self.unet = FakeModule()
            self.attention_sliced = False
            self.progress_disabled = False
            self.device = None

        def set_progress_bar_config(self, disable: bool) -> None:
            self.progress_disabled = disable

        def enable_attention_slicing(self) -> None:
            self.attention_sliced = True

        def to(self, device: torch.device) -> None:
            self.device = device

    pipe = FakePipe()

    class FakePipeline:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.update(kwargs)
            return pipe

    monkeypatch.setattr(generation, "StableDiffusionPipeline", FakePipeline)
    monkeypatch.setattr(generation, "DPMSolverMultistepScheduler", FakeScheduler)

    result = load_pipeline(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        revision=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert result is pipe
    assert calls["dtype"] is torch.float32
    assert calls["use_safetensors"] is True
    assert result.scheduler == "scheduler"
    assert result.vae.sliced is True
    assert result.vae.requires_grad is False
    assert result.text_encoder.requires_grad is False
    assert result.unet.requires_grad is False
    assert result.attention_sliced is True
    assert result.progress_disabled is True
    assert result.device == torch.device("cpu")


def test_load_lora_adapter_uses_unprefixed_safetensors() -> None:
    class FakeUnet:
        def __init__(self) -> None:
            self.loaded: dict[str, object] | None = None
            self.active: str | None = None

        def load_lora_adapter(self, path: str, **kwargs) -> None:
            self.loaded = {"path": path, **kwargs}

        def set_adapter(self, name: str) -> None:
            self.active = name

    unet = FakeUnet()
    pipe = SimpleNamespace(unet=unet)

    load_lora_adapter(
        pipe,
        Path("/tmp/adapter"),
        weights_file="pytorch_lora_weights.safetensors",
        adapter_name="style",
    )

    assert unet.loaded == {
        "path": "/tmp/adapter",
        "weight_name": "pytorch_lora_weights.safetensors",
        "prefix": None,
        "adapter_name": "style",
    }
    assert unet.active == "style"
