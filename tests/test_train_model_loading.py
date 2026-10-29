from __future__ import annotations

from pathlib import Path

import torch

import lora_style_imagegen.train as train
from lora_style_imagegen.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeModule:
    def requires_grad_(self, requires_grad: bool) -> FakeModule:
        return self

    def eval(self) -> FakeModule:
        return self

    def to(self, device: torch.device) -> FakeModule:
        return self


def test_component_loading_requires_safetensors(monkeypatch) -> None:
    calls: dict[str, dict[str, object]] = {}

    def make_loader(name: str, result: object):
        def load(*args, **kwargs):
            calls[name] = kwargs
            return result

        return load

    class FakeTokenizer:
        from_pretrained = staticmethod(make_loader("tokenizer", object()))

    class FakeTextEncoder:
        from_pretrained = staticmethod(make_loader("text_encoder", FakeModule()))

    class FakeVae:
        from_pretrained = staticmethod(make_loader("vae", FakeModule()))

    class FakeUnet:
        from_pretrained = staticmethod(make_loader("unet", FakeModule()))

    class FakeScheduler:
        from_pretrained = staticmethod(make_loader("scheduler", object()))

    monkeypatch.setattr(train, "CLIPTokenizer", FakeTokenizer)
    monkeypatch.setattr(train, "CLIPTextModel", FakeTextEncoder)
    monkeypatch.setattr(train, "AutoencoderKL", FakeVae)
    monkeypatch.setattr(train, "UNet2DConditionModel", FakeUnet)
    monkeypatch.setattr(train, "DDPMScheduler", FakeScheduler)

    config = load_config(PROJECT_ROOT / "configs/train_lora.yaml")
    train._load_components(config, torch.device("cpu"), torch.float32)

    for component in ("text_encoder", "vae", "unet"):
        assert calls[component]["use_safetensors"] is True
    assert "use_safetensors" not in calls["tokenizer"]
    assert "use_safetensors" not in calls["scheduler"]


def test_lora_parameters_are_float32_when_base_model_is_fp16() -> None:
    config = load_config(PROJECT_ROOT / "configs/train_lora.yaml")
    unet = train._build_tiny_unet().half()

    train._add_lora(unet, config)

    trainable = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    assert trainable
    assert {parameter.dtype for parameter in trainable} == {torch.float32}
    assert {parameter.dtype for parameter in unet.parameters() if not parameter.requires_grad} == {
        torch.float16
    }
