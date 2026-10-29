from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any

import yaml

ALLOWED_KEYS = frozenset(
    {
        "base_model",
        "base_model_revision",
        "batch_size",
        "dataset_dir",
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "learning_rate",
        "logging_steps",
        "lora_alpha",
        "lora_dropout",
        "lr_scheduler",
        "lr_warmup_steps",
        "max_grad_norm",
        "max_train_steps",
        "mixed_precision",
        "output_dir",
        "rank",
        "report_to",
        "resolution",
        "run_name",
        "save_steps",
        "seed",
        "target_modules",
        "trigger_token",
        "use_8bit_adam",
        "validation_guidance_scale",
        "validation_inference_steps",
        "validation_negative_prompt",
        "validation_prompts",
        "validation_seeds",
        "validation_steps",
    }
)
LR_SCHEDULERS = frozenset(
    {
        "constant",
        "constant_with_warmup",
        "cosine",
        "cosine_with_restarts",
        "linear",
        "polynomial",
    }
)
MIXED_PRECISION_MODES = frozenset({"no", "fp16"})
REPORT_TO_OPTIONS = frozenset({"none", "tensorboard", "wandb"})


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class TrainConfig:
    base_model: str
    dataset_dir: Path
    output_dir: Path
    run_name: str
    base_model_revision: str | None = None
    trigger_token: str | None = None
    resolution: int = 512
    seed: int = 42
    rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = ("to_q", "to_k", "to_v", "to_out.0")
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_train_steps: int = 800
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 50
    gradient_checkpointing: bool = True
    use_8bit_adam: bool = True
    mixed_precision: str = "fp16"
    max_grad_norm: float = 1.0
    save_steps: int = 400
    validation_steps: int = 200
    validation_prompts: tuple[str, ...] = ()
    validation_seeds: tuple[int, ...] = (1234, 5678, 9012)
    validation_guidance_scale: float = 7.5
    validation_inference_steps: int = 25
    validation_negative_prompt: str = "low quality, blurry, deformed"
    report_to: str = "tensorboard"
    logging_steps: int = 10


def load_config(path: Path) -> TrainConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"Cannot read config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, Mapping):
        raise ConfigError(f"Config {path} must be a YAML mapping")
    return parse_config(raw)


def parse_config(raw: Mapping[str, Any]) -> TrainConfig:
    unknown = sorted(set(raw) - ALLOWED_KEYS)
    if unknown:
        raise ConfigError(f"Unknown config keys: {', '.join(unknown)}")

    required = sorted(ALLOWED_KEYS - {"base_model_revision", "trigger_token"})
    missing = [key for key in required if key not in raw]
    if missing:
        raise ConfigError(f"Missing config keys: {', '.join(missing)}")

    config = TrainConfig(
        base_model=_string(raw, "base_model"),
        base_model_revision=_optional_string(raw, "base_model_revision"),
        dataset_dir=Path(_string(raw, "dataset_dir")),
        output_dir=Path(_string(raw, "output_dir")),
        run_name=_string(raw, "run_name"),
        trigger_token=_optional_string(raw, "trigger_token"),
        resolution=_integer(raw, "resolution"),
        seed=_integer(raw, "seed"),
        rank=_integer(raw, "rank"),
        lora_alpha=_integer(raw, "lora_alpha"),
        lora_dropout=_number(raw, "lora_dropout"),
        target_modules=_string_tuple(raw, "target_modules"),
        batch_size=_integer(raw, "batch_size"),
        gradient_accumulation_steps=_integer(raw, "gradient_accumulation_steps"),
        max_train_steps=_integer(raw, "max_train_steps"),
        learning_rate=_number(raw, "learning_rate"),
        lr_scheduler=_string(raw, "lr_scheduler"),
        lr_warmup_steps=_integer(raw, "lr_warmup_steps"),
        gradient_checkpointing=_boolean(raw, "gradient_checkpointing"),
        use_8bit_adam=_boolean(raw, "use_8bit_adam"),
        mixed_precision=_string(raw, "mixed_precision"),
        max_grad_norm=_number(raw, "max_grad_norm"),
        save_steps=_integer(raw, "save_steps"),
        validation_steps=_integer(raw, "validation_steps"),
        validation_prompts=_string_tuple(raw, "validation_prompts"),
        validation_seeds=_integer_tuple(raw, "validation_seeds"),
        validation_guidance_scale=_number(raw, "validation_guidance_scale"),
        validation_inference_steps=_integer(raw, "validation_inference_steps"),
        validation_negative_prompt=_string(raw, "validation_negative_prompt"),
        report_to=_string(raw, "report_to"),
        logging_steps=_integer(raw, "logging_steps"),
    )
    _validate_config(config)
    return config


def validate_config(config: TrainConfig) -> None:
    _validate_config(config)


def format_prompts(prompts: tuple[str, ...], trigger_token: str) -> tuple[str, ...]:
    formatted: list[str] = []
    for prompt in prompts:
        try:
            text = " ".join(prompt.format(trigger=trigger_token).split())
        except (KeyError, IndexError, ValueError) as exc:
            raise ConfigError(
                f"Invalid validation prompt {prompt!r}; only {{trigger}} is supported"
            ) from exc
        if not text:
            raise ConfigError("Validation prompts must not produce empty text")
        formatted.append(text)
    return tuple(formatted)


def _validate_config(config: TrainConfig) -> None:
    if not 8 <= config.rank <= 16:
        raise ConfigError("rank must be between 8 and 16")
    if config.lora_alpha <= 0:
        raise ConfigError("lora_alpha must be positive")
    if not 0 <= config.lora_dropout <= 1:
        raise ConfigError("lora_dropout must be between 0 and 1")
    if not config.target_modules:
        raise ConfigError("target_modules must not be empty")
    if config.resolution < 256 or config.resolution % 8 != 0:
        raise ConfigError("resolution must be at least 256 and divisible by 8")
    if config.batch_size < 1:
        raise ConfigError("batch_size must be at least 1")
    if config.gradient_accumulation_steps < 1:
        raise ConfigError("gradient_accumulation_steps must be at least 1")
    if config.max_train_steps < 1:
        raise ConfigError("max_train_steps must be at least 1")
    if config.learning_rate <= 0:
        raise ConfigError("learning_rate must be positive")
    if config.lr_scheduler not in LR_SCHEDULERS:
        raise ConfigError(f"lr_scheduler must be one of: {', '.join(sorted(LR_SCHEDULERS))}")
    if not 0 <= config.lr_warmup_steps <= config.max_train_steps:
        raise ConfigError("lr_warmup_steps must be between 0 and max_train_steps")
    if config.max_grad_norm <= 0:
        raise ConfigError("max_grad_norm must be positive")
    if config.save_steps < 1:
        raise ConfigError("save_steps must be at least 1")
    if config.validation_steps < 1:
        raise ConfigError("validation_steps must be at least 1")
    if not config.validation_prompts:
        raise ConfigError("validation_prompts must not be empty")
    if not config.validation_seeds:
        raise ConfigError("validation_seeds must not be empty")
    if any(seed < 0 for seed in config.validation_seeds):
        raise ConfigError("validation_seeds must be non-negative")
    if config.validation_guidance_scale <= 0:
        raise ConfigError("validation_guidance_scale must be positive")
    if config.validation_inference_steps < 1:
        raise ConfigError("validation_inference_steps must be at least 1")
    if config.mixed_precision not in MIXED_PRECISION_MODES:
        raise ConfigError(
            f"mixed_precision must be one of: {', '.join(sorted(MIXED_PRECISION_MODES))}"
        )
    if config.report_to not in REPORT_TO_OPTIONS:
        raise ConfigError(f"report_to must be one of: {', '.join(sorted(REPORT_TO_OPTIONS))}")
    if config.logging_steps < 1:
        raise ConfigError("logging_steps must be at least 1")
    if config.trigger_token is not None:
        _validate_trigger(config.trigger_token, "trigger_token")

    for prompt in config.validation_prompts:
        _validate_template(prompt, "validation prompt")


def _validate_trigger(value: str, label: str) -> None:
    if not value.strip():
        raise ConfigError(f"{label} must not be empty")
    if "{" in value or "}" in value:
        raise ConfigError(f"{label} must not contain braces")


def _validate_template(value: str, label: str) -> None:
    try:
        fields = [field for _, field, _, _ in Formatter().parse(value) if field is not None]
    except ValueError as exc:
        raise ConfigError(f"Invalid {label} {value!r}: {exc}") from exc
    unsupported = sorted({field for field in fields if field != "trigger"})
    if unsupported:
        raise ConfigError(
            f"Invalid {label} {value!r}; only {{trigger}} is supported, "
            f"found: {', '.join(unsupported)}"
        )


def _string(raw: Mapping[str, Any], key: str) -> str:
    value = raw[key]
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{key} must be a non-empty string")
    return value.strip()


def _optional_string(raw: Mapping[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{key} must be null or a non-empty string")
    return value.strip()


def _integer(raw: Mapping[str, Any], key: str) -> int:
    value = raw[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{key} must be an integer")
    return value


def _integer_tuple(raw: Mapping[str, Any], key: str) -> tuple[int, ...]:
    value = raw[key]
    if not isinstance(value, list):
        raise ConfigError(f"{key} must be a list of integers")
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            raise ConfigError(f"{key} must be a list of integers")
    return tuple(value)


def _number(raw: Mapping[str, Any], key: str) -> float:
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{key} must be a number")
    return float(value)


def _boolean(raw: Mapping[str, Any], key: str) -> bool:
    value = raw[key]
    if not isinstance(value, bool):
        raise ConfigError(f"{key} must be true or false")
    return value


def _string_tuple(raw: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = raw[key]
    if not isinstance(value, list):
        raise ConfigError(f"{key} must be a list of strings")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ConfigError(f"{key} must be a list of non-empty strings")
    return tuple(item.strip() for item in value)
