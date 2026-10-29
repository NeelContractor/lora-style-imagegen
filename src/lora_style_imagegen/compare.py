from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from lora_style_imagegen.config import ConfigError, format_prompts
from lora_style_imagegen.generation import (
    GenerationError,
    generate_image,
    load_lora_adapter,
    load_pipeline,
    resolve_device,
    resolve_dtype,
)

DEFAULT_ADAPTER_DIR = Path("outputs/lora-style/lora")
DEFAULT_OUTPUT_DIR = Path("outputs/comparison")
DEFAULT_ADAPTER_NAME = "style"
DEFAULT_NEGATIVE_PROMPT = "low quality, blurry, deformed"
DEFAULT_RESOLUTION = 512
GRID_THUMB_SIZE = 256
GRID_LABEL_WIDTH = 300
GRID_HEADER_HEIGHT = 36


class ComparisonError(Exception):
    pass


@dataclass(frozen=True)
class AdapterInfo:
    base_model: str
    base_model_revision: str | None
    trigger_token: str
    weights_file: str
    prompts: tuple[str, ...]
    seeds: tuple[int, ...]
    guidance_scale: float
    num_inference_steps: int
    resolution: int
    negative_prompt: str


@dataclass(frozen=True)
class ComparisonConfig:
    adapter_dir: Path
    output_dir: Path
    base_model: str
    base_model_revision: str | None
    trigger_token: str
    weights_file: str
    adapter_name: str
    prompts: tuple[str, ...]
    seeds: tuple[int, ...]
    negative_prompt: str
    guidance_scale: float
    num_inference_steps: int
    width: int
    height: int
    device: str = "auto"
    dtype: str = "auto"


@dataclass(frozen=True)
class ComparisonResult:
    output_dir: Path
    grid_path: Path
    metadata_path: Path
    image_count: int
    prompt_count: int
    seed_count: int
    device: str
    dtype: str


def load_adapter_info(adapter_dir: Path) -> AdapterInfo:
    metadata_path = adapter_dir / "lora_metadata.json"
    if not metadata_path.is_file():
        raise ComparisonError(f"Missing adapter metadata: {metadata_path}")

    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"Could not read {metadata_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ComparisonError(f"{metadata_path} must contain a JSON object")

    base_model = _metadata_string(raw, "base_model")
    trigger_token = _metadata_string(raw, "trigger_token")
    weights_file = _metadata_string(raw, "weights_file")
    if not weights_file.endswith(".safetensors"):
        raise ComparisonError("Adapter weights must use the safetensors format")
    if Path(weights_file).name != weights_file:
        raise ComparisonError("Adapter weight filename must not contain directories")
    if not (adapter_dir / weights_file).is_file():
        raise ComparisonError(f"Missing adapter weights: {adapter_dir / weights_file}")

    revision = raw.get("base_model_revision")
    if revision is not None and not isinstance(revision, str):
        raise ComparisonError("base_model_revision must be a string or null")

    prompts = _metadata_string_list(raw, "validation_prompts")
    seeds = _metadata_int_list(raw, "validation_seeds")
    guidance_scale = _metadata_number(raw, "validation_guidance_scale")
    num_inference_steps = _metadata_int(raw, "validation_inference_steps")
    resolution = _metadata_int(raw, "resolution", default=DEFAULT_RESOLUTION)
    negative_prompt = raw.get("validation_negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    if not isinstance(negative_prompt, str):
        raise ComparisonError("validation_negative_prompt must be a string")

    return AdapterInfo(
        base_model=base_model,
        base_model_revision=revision,
        trigger_token=trigger_token,
        weights_file=weights_file,
        prompts=prompts,
        seeds=seeds,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        resolution=resolution,
        negative_prompt=negative_prompt,
    )


def build_comparison_prompts(
    prompts: Sequence[str],
    trigger_token: str,
) -> tuple[str, ...]:
    if not prompts:
        raise ComparisonError("At least one comparison prompt is required")

    result: list[str] = []
    seen: set[str] = set()
    for prompt in prompts:
        try:
            formatted = format_prompts((prompt,), trigger_token)[0]
        except ConfigError as exc:
            raise ComparisonError(str(exc)) from exc
        candidates = [formatted]
        if trigger_token in formatted:
            without_trigger = _strip_trigger(formatted, trigger_token)
            if without_trigger and without_trigger != formatted:
                candidates.append(without_trigger)
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                result.append(candidate)
    return tuple(result)


def run_comparison(config: ComparisonConfig) -> ComparisonResult:
    _validate_config(config)
    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype, device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    base_dir = config.output_dir / "base"
    lora_dir = config.output_dir / "lora"
    base_dir.mkdir(parents=True, exist_ok=True)
    lora_dir.mkdir(parents=True, exist_ok=True)

    image_paths: dict[tuple[str, int, int], Path] = {}
    records: list[dict[str, Any]] = []
    pipe = load_pipeline(
        config.base_model,
        revision=config.base_model_revision,
        device=device,
        dtype=dtype,
    )
    try:
        for variant in ("base", "lora"):
            if variant == "lora":
                print(f"Loading LoRA adapter from {config.adapter_dir}")
                load_lora_adapter(
                    pipe,
                    config.adapter_dir,
                    weights_file=config.weights_file,
                    adapter_name=config.adapter_name,
                )
            for prompt_index, prompt in enumerate(config.prompts):
                for seed in config.seeds:
                    print(
                        f"{variant} prompt {prompt_index + 1}/{len(config.prompts)} "
                        f"seed {seed}: {prompt}"
                    )
                    image = generate_image(
                        pipe,
                        prompt=prompt,
                        negative_prompt=config.negative_prompt,
                        num_inference_steps=config.num_inference_steps,
                        guidance_scale=config.guidance_scale,
                        width=config.width,
                        height=config.height,
                        seed=seed,
                        device=device,
                    )
                    path = (
                        base_dir if variant == "base" else lora_dir
                    ) / f"prompt_{prompt_index + 1:02d}_seed_{seed}.png"
                    image.save(path)
                    image_paths[(variant, prompt_index, seed)] = path
                    records.append(
                        {
                            "variant": variant,
                            "prompt_index": prompt_index,
                            "prompt": prompt,
                            "seed": seed,
                            "file": path.relative_to(config.output_dir).as_posix(),
                        }
                    )
    finally:
        del pipe
        _empty_cuda_cache()

    grid_path = config.output_dir / "comparison_grid.png"
    create_comparison_grid(
        prompts=config.prompts,
        seeds=config.seeds,
        image_paths=image_paths,
        output_path=grid_path,
    )
    metadata_path = config.output_dir / "comparison_metadata.json"
    _write_metadata(
        config=config,
        device=device,
        dtype=dtype,
        records=records,
        grid_path=grid_path,
        metadata_path=metadata_path,
    )
    return ComparisonResult(
        output_dir=config.output_dir,
        grid_path=grid_path,
        metadata_path=metadata_path,
        image_count=len(records),
        prompt_count=len(config.prompts),
        seed_count=len(config.seeds),
        device=str(device),
        dtype=_dtype_name(dtype),
    )


def create_comparison_grid(
    *,
    prompts: Sequence[str],
    seeds: Sequence[int],
    image_paths: Mapping[tuple[str, int, int], Path],
    output_path: Path,
) -> None:
    columns = len(seeds) * 2
    width = GRID_LABEL_WIDTH + columns * GRID_THUMB_SIZE
    height = GRID_HEADER_HEIGHT + len(prompts) * GRID_THUMB_SIZE
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=13)
    small_font = ImageFont.load_default(size=11)

    draw.rectangle([0, 0, width, GRID_HEADER_HEIGHT], fill="#eeeeee")
    draw.text((8, 10), "Prompt", fill="black", font=font)
    for seed_index, seed in enumerate(seeds):
        for variant_index, variant in enumerate(("base", "lora")):
            column = seed_index * 2 + variant_index
            x = GRID_LABEL_WIDTH + column * GRID_THUMB_SIZE
            label = f"{_variant_label(variant)} {seed}"
            draw.text((x + 6, 10), label, fill="black", font=font)

    for row, prompt in enumerate(prompts):
        y = GRID_HEADER_HEIGHT + row * GRID_THUMB_SIZE
        label = f"{row + 1:02d}  {prompt}"
        wrapped = "\n".join(textwrap.wrap(label, width=34))
        draw.multiline_text((8, y + 8), wrapped, fill="black", font=small_font, spacing=3)

    for seed_index, seed in enumerate(seeds):
        for variant_index, variant in enumerate(("base", "lora")):
            column = seed_index * 2 + variant_index
            x = GRID_LABEL_WIDTH + column * GRID_THUMB_SIZE
            for row in range(len(prompts)):
                y = GRID_HEADER_HEIGHT + row * GRID_THUMB_SIZE
                thumbnail = _make_thumbnail(image_paths[(variant, row, seed)])
                canvas.paste(thumbnail, (x, y))
                draw.rectangle(
                    [x, y, x + GRID_THUMB_SIZE - 1, y + GRID_THUMB_SIZE - 1],
                    outline="#cccccc",
                )

    draw.rectangle([0, 0, width - 1, height - 1], outline="#888888")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _make_thumbnail(path: Path) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
        thumbnail = ImageOps.contain(
            image,
            (GRID_THUMB_SIZE, GRID_THUMB_SIZE),
            Image.Resampling.LANCZOS,
        )
    cell = Image.new("RGB", (GRID_THUMB_SIZE, GRID_THUMB_SIZE), "white")
    cell.paste(
        thumbnail,
        (
            (GRID_THUMB_SIZE - thumbnail.width) // 2,
            (GRID_THUMB_SIZE - thumbnail.height) // 2,
        ),
    )
    return cell


def _write_metadata(
    *,
    config: ComparisonConfig,
    device: torch.device,
    dtype: torch.dtype,
    records: list[dict[str, Any]],
    grid_path: Path,
    metadata_path: Path,
) -> None:
    payload = {
        "base_model": config.base_model,
        "base_model_revision": config.base_model_revision,
        "adapter_dir": str(config.adapter_dir),
        "adapter_name": config.adapter_name,
        "adapter_weights": config.weights_file,
        "trigger_token": config.trigger_token,
        "prompts": list(config.prompts),
        "seeds": list(config.seeds),
        "negative_prompt": config.negative_prompt,
        "guidance_scale": config.guidance_scale,
        "num_inference_steps": config.num_inference_steps,
        "width": config.width,
        "height": config.height,
        "device": str(device),
        "dtype": _dtype_name(dtype),
        "scheduler": "DPMSolverMultistepScheduler",
        "grid": grid_path.name,
        "images": records,
    }
    metadata_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_config(config: ComparisonConfig) -> None:
    if not config.adapter_dir.is_dir():
        raise ComparisonError(f"Adapter directory not found: {config.adapter_dir}")
    weight_path = config.adapter_dir / config.weights_file
    if not weight_path.is_file():
        raise ComparisonError(f"Adapter weights not found: {weight_path}")
    if not config.base_model:
        raise ComparisonError("base_model must not be empty")
    if not config.trigger_token:
        raise ComparisonError("trigger_token must not be empty")
    if not config.prompts:
        raise ComparisonError("At least one comparison prompt is required")
    if not config.seeds:
        raise ComparisonError("At least one seed is required")
    if any(seed < 0 for seed in config.seeds):
        raise ComparisonError("Seeds must be non-negative integers")
    if config.num_inference_steps < 1:
        raise ComparisonError("num_inference_steps must be at least 1")
    if config.guidance_scale <= 0:
        raise ComparisonError("guidance_scale must be positive")
    for dimension, name in ((config.width, "width"), (config.height, "height")):
        if dimension < 256 or dimension % 8 != 0:
            raise ComparisonError(f"{name} must be at least 256 and divisible by 8")
    if config.device not in {"auto", "cuda", "cpu"}:
        raise ComparisonError("device must be auto, cuda, or cpu")
    if config.dtype not in {"auto", "fp16", "fp32"}:
        raise ComparisonError("dtype must be auto, fp16, or fp32")


def _strip_trigger(prompt: str, trigger_token: str) -> str:
    without_trigger = " ".join(prompt.replace(trigger_token, " ").split())
    return without_trigger.strip(" ,")


def _variant_label(variant: str) -> str:
    return "Base" if variant == "base" else "LoRA"


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _metadata_string(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ComparisonError(f"Adapter metadata field {key!r} must be a non-empty string")
    return value.strip()


def _metadata_string_list(raw: dict[str, Any], key: str) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, list) or not value:
        raise ComparisonError(f"Adapter metadata field {key!r} must be a non-empty list")
    items = tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    if len(items) != len(value):
        raise ComparisonError(f"Adapter metadata field {key!r} must contain strings")
    return items


def _metadata_int_list(raw: dict[str, Any], key: str) -> tuple[int, ...]:
    value = raw.get(key)
    if not isinstance(value, list) or not value:
        raise ComparisonError(f"Adapter metadata field {key!r} must be a non-empty list")
    items = tuple(item for item in value if isinstance(item, int) and not isinstance(item, bool))
    if len(items) != len(value) or any(item < 0 for item in items):
        raise ComparisonError(f"Adapter metadata field {key!r} must contain non-negative integers")
    return items


def _metadata_int(
    raw: dict[str, Any],
    key: str,
    *,
    default: int | None = None,
) -> int:
    value = raw.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ComparisonError(f"Adapter metadata field {key!r} must be an integer")
    return value


def _metadata_number(raw: dict[str, Any], key: str) -> float:
    value = raw.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ComparisonError(f"Adapter metadata field {key!r} must be a number")
    return float(value)


def _empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Stable Diffusion 1.5 base outputs with a trained LoRA adapter."
    )
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--trigger-token", default=None)
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Prompt template; use {trigger}. Repeat to add prompts. Each trigger prompt "
        "also gets a matching prompt without the trigger.",
    )
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "fp16", "fp32"), default="auto")
    parser.add_argument("--adapter-name", default=DEFAULT_ADAPTER_NAME)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        info = load_adapter_info(args.adapter_dir)
        source_prompts = tuple(args.prompt) if args.prompt else info.prompts
        trigger_token = args.trigger_token or info.trigger_token
        config = ComparisonConfig(
            adapter_dir=args.adapter_dir,
            output_dir=args.output_dir,
            base_model=args.base_model or info.base_model,
            base_model_revision=info.base_model_revision,
            trigger_token=trigger_token,
            weights_file=info.weights_file,
            adapter_name=args.adapter_name,
            prompts=build_comparison_prompts(source_prompts, trigger_token),
            seeds=tuple(args.seed) if args.seed else info.seeds,
            negative_prompt=args.negative_prompt
            if args.negative_prompt is not None
            else info.negative_prompt,
            guidance_scale=args.guidance_scale
            if args.guidance_scale is not None
            else info.guidance_scale,
            num_inference_steps=args.steps if args.steps is not None else info.num_inference_steps,
            width=args.width if args.width is not None else info.resolution,
            height=args.height if args.height is not None else info.resolution,
            device=args.device,
            dtype=args.dtype,
        )
        result = run_comparison(config)
    except (ComparisonError, GenerationError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Comparison complete: {result.image_count} images "
        f"({result.prompt_count} prompts x {result.seed_count} seeds x 2 variants)"
    )
    print(f"Grid: {result.grid_path}")
    print(f"Metadata: {result.metadata_path}")
    print(f"Device: {result.device} ({result.dtype})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
