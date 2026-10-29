from __future__ import annotations

import argparse
import os
import secrets
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

from lora_style_imagegen.compare import ComparisonError, load_adapter_info
from lora_style_imagegen.generation import (
    GenerationError,
    generate_image,
    load_lora_adapter,
    load_pipeline,
    resolve_device,
    resolve_dtype,
)

DEFAULT_ADAPTER_DIR = Path("outputs/lora-style/lora")
DEFAULT_ADAPTER_NAME = "style"
DEFAULT_LORA_SCALE = 1.0
MAX_SEED = 2**31 - 1


class DemoError(Exception):
    pass


@dataclass(frozen=True)
class DemoConfig:
    adapter_dir: Path
    base_model: str
    base_model_revision: str | None
    trigger_token: str
    weights_file: str
    adapter_name: str
    negative_prompt: str
    guidance_scale: float
    num_inference_steps: int
    width: int
    height: int
    device: str = "auto"
    dtype: str = "auto"


def load_demo_config(
    adapter_dir: Path,
    *,
    base_model: str | None = None,
    device: str = "auto",
    dtype: str = "auto",
) -> DemoConfig:
    try:
        info = load_adapter_info(adapter_dir)
    except ComparisonError as exc:
        raise DemoError(str(exc)) from exc
    if info.resolution < 256 or info.resolution % 8 != 0:
        raise DemoError("Adapter resolution must be at least 256 and divisible by 8")
    return DemoConfig(
        adapter_dir=adapter_dir,
        base_model=base_model or info.base_model,
        base_model_revision=info.base_model_revision,
        trigger_token=info.trigger_token,
        weights_file=info.weights_file,
        adapter_name=DEFAULT_ADAPTER_NAME,
        negative_prompt=info.negative_prompt,
        guidance_scale=info.guidance_scale,
        num_inference_steps=info.num_inference_steps,
        width=info.resolution,
        height=info.resolution,
        device=device,
        dtype=dtype,
    )


class StyleDemo:
    def __init__(
        self,
        config: DemoConfig,
        *,
        pipe: Any | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self._pipe: StableDiffusionPipeline | None = pipe
        self._device = device
        self._lock = threading.Lock()

    def generate_pair(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        steps: int,
        guidance_scale: float,
        lora_scale: float,
    ) -> tuple[Image.Image, Image.Image, int]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise DemoError("Prompt must not be empty")
        prompt = prompt.strip()
        negative_prompt = "" if negative_prompt is None else str(negative_prompt)
        try:
            steps = int(steps)
        except (TypeError, ValueError) as exc:
            raise DemoError("Steps must be an integer") from exc
        if not 1 <= steps <= 100:
            raise DemoError("Steps must be between 1 and 100")
        try:
            guidance_scale = float(guidance_scale)
        except (TypeError, ValueError) as exc:
            raise DemoError("Guidance must be a number") from exc
        if not 0 <= guidance_scale <= 30:
            raise DemoError("Guidance must be between 0 and 30")
        try:
            lora_scale = float(lora_scale)
        except (TypeError, ValueError) as exc:
            raise DemoError("LoRA strength must be a number") from exc
        if not 0 <= lora_scale <= 2:
            raise DemoError("LoRA strength must be between 0 and 2")
        try:
            seed = int(seed)
        except (TypeError, ValueError) as exc:
            raise DemoError("Seed must be an integer") from exc
        if seed < 0:
            seed = secrets.randbelow(MAX_SEED)

        with self._lock:
            self._ensure_pipeline()
            if self._pipe is None or self._device is None:
                raise DemoError("Pipeline is not available")

            self._pipe.set_adapters(
                [self.config.adapter_name],
                adapter_weights=[0.0],
            )
            base = generate_image(
                self._pipe,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=self.config.width,
                height=self.config.height,
                seed=seed,
                device=self._device,
            )

            self._pipe.set_adapters(
                [self.config.adapter_name],
                adapter_weights=[lora_scale],
            )
            styled = generate_image(
                self._pipe,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=self.config.width,
                height=self.config.height,
                seed=seed,
                device=self._device,
            )

        return base, styled, seed

    def _ensure_pipeline(self) -> None:
        if self._pipe is not None:
            return
        try:
            device = resolve_device(self.config.device)
            dtype = resolve_dtype(self.config.dtype, device)
            pipe = load_pipeline(
                self.config.base_model,
                revision=self.config.base_model_revision,
                device=device,
                dtype=dtype,
            )
            load_lora_adapter(
                pipe,
                self.config.adapter_dir,
                weights_file=self.config.weights_file,
                adapter_name=self.config.adapter_name,
            )
        except (GenerationError, OSError, RuntimeError, ValueError) as exc:
            raise DemoError(f"Could not load the style model: {exc}") from exc
        self._pipe = pipe
        self._device = device


def build_interface(style: StyleDemo):
    try:
        import gradio as gr
    except ImportError as exc:
        raise DemoError("Gradio is not installed; run uv sync --extra train --extra app") from exc

    config = style.config
    default_prompt = f"{config.trigger_token}, a botanical study of ferns"

    with gr.Blocks(title="Botanical Watercolor LoRA") as demo:
        gr.Markdown(
            "# Botanical Watercolor LoRA\n"
            "Stable Diffusion 1.5 with a LoRA trained for a vintage botanical "
            "watercolor style.\n\n"
            f"Prompt with the trigger token `{config.trigger_token}`. Base and LoRA "
            "images use the same seed; set LoRA strength to 0 to match the base "
            "output. The first request downloads and loads the base model."
        )
        with gr.Row():
            with gr.Column(scale=1, min_width=340):
                prompt = gr.Textbox(
                    label="Prompt",
                    value=default_prompt,
                    lines=3,
                )
                negative_prompt = gr.Textbox(
                    label="Negative prompt",
                    value=config.negative_prompt,
                    lines=2,
                )
                with gr.Row():
                    seed = gr.Number(
                        label="Seed",
                        value=1234,
                        precision=0,
                        info="-1 for random",
                    )
                    steps = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=config.num_inference_steps,
                        step=1,
                        label="Steps",
                    )
                with gr.Row():
                    guidance_scale = gr.Slider(
                        minimum=0,
                        maximum=15,
                        value=config.guidance_scale,
                        step=0.5,
                        label="Guidance",
                    )
                    lora_scale = gr.Slider(
                        minimum=0,
                        maximum=1.5,
                        value=DEFAULT_LORA_SCALE,
                        step=0.05,
                        label="LoRA strength",
                    )
                generate = gr.Button("Generate", variant="primary")
            with gr.Column(scale=2):
                with gr.Row():
                    base_image = gr.Image(label="Base", type="pil", height=512)
                    lora_image = gr.Image(label="LoRA", type="pil", height=512)
                used_seed = gr.Number(label="Seed used", precision=0, interactive=False)

        def run(prompt_text, negative_text, seed_value, step_count, cfg, strength):
            try:
                return style.generate_pair(
                    prompt_text,
                    negative_text,
                    seed_value,
                    step_count,
                    cfg,
                    strength,
                )
            except DemoError as exc:
                raise gr.Error(str(exc)) from exc

        inputs = [prompt, negative_prompt, seed, steps, guidance_scale, lora_scale]
        outputs = [base_image, lora_image, used_seed]
        generate.click(run, inputs=inputs, outputs=outputs, api_name="generate")
        prompt.submit(run, inputs=inputs, outputs=outputs)

    return demo


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Gradio demo for the trained Stable Diffusion 1.5 LoRA."
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path(os.environ.get("LORA_ADAPTER_DIR", DEFAULT_ADAPTER_DIR)),
    )
    parser.add_argument("--base-model", default=os.environ.get("LORA_BASE_MODEL"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "fp16", "fp32"), default="auto")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        config = load_demo_config(
            args.adapter_dir,
            base_model=args.base_model,
            device=args.device,
            dtype=args.dtype,
        )
        style = StyleDemo(config)
        demo = build_interface(style)
    except DemoError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        inbrowser=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
