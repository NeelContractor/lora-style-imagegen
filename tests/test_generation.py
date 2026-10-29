from __future__ import annotations

from types import SimpleNamespace

import torch

from lora_style_imagegen.generation import generate_image


class FakeVae:
    dtype = torch.float16
    config = SimpleNamespace(scaling_factor=0.18215)
    seen: torch.Tensor | None = None

    def decode(self, latents: torch.Tensor, return_dict: bool) -> tuple[torch.Tensor]:
        self.seen = latents
        return (torch.zeros(1, 3, 8, 8),)


class FakeImageProcessor:
    def postprocess(self, image: torch.Tensor, output_type: str) -> list[str]:
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
