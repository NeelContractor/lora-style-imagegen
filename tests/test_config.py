from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lora_style_imagegen.config import ConfigError, format_prompts, load_config, parse_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/train_lora.yaml"


def _raw_config() -> dict[str, object]:
    return yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_loads_default_t4_config() -> None:
    config = load_config(DEFAULT_CONFIG)

    assert config.base_model == "stable-diffusion-v1-5/stable-diffusion-v1-5"
    assert config.rank == 16
    assert config.mixed_precision == "fp16"
    assert config.use_8bit_adam is True
    assert config.gradient_checkpointing is True
    assert config.report_to == "tensorboard"
    assert config.dataset_dir == Path("data/processed")
    assert config.max_train_steps == 800


def test_rejects_unknown_keys() -> None:
    raw = _raw_config()
    raw["learning_rates"] = 1e-3

    with pytest.raises(ConfigError, match="Unknown config keys"):
        parse_config(raw)


@pytest.mark.parametrize("rank", [4, 17])
def test_rejects_rank_outside_supported_range(rank: int) -> None:
    raw = _raw_config()
    raw["rank"] = rank

    with pytest.raises(ConfigError, match="between 8 and 16"):
        parse_config(raw)


def test_rejects_invalid_validation_prompt_placeholder() -> None:
    raw = _raw_config()
    raw["validation_prompts"] = ["{trigger}, {unknown}"]

    with pytest.raises(ConfigError, match="only \\{trigger\\}"):
        parse_config(raw)


def test_rejects_unknown_trigger_braces() -> None:
    raw = _raw_config()
    raw["trigger_token"] = "sks {style}"

    with pytest.raises(ConfigError, match="must not contain braces"):
        parse_config(raw)


def test_formats_trigger_in_validation_prompts() -> None:
    config = load_config(DEFAULT_CONFIG)

    prompts = format_prompts(config.validation_prompts, "sks style")

    assert prompts[0] == "sks style, portrait of a person, soft window light"
    assert all(prompt.startswith("sks style,") for prompt in prompts)
