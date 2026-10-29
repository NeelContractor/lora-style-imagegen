from __future__ import annotations

from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from PIL import Image


class GenerationError(Exception):
    pass


def resolve_device(choice: str) -> torch.device:
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise GenerationError("CUDA was requested but is not available")
        return torch.device("cuda")
    if choice == "cpu":
        return torch.device("cpu")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def resolve_dtype(choice: str, device: torch.device) -> torch.dtype:
    if choice == "fp32":
        return torch.float32
    if choice == "fp16":
        if device.type != "cuda":
            raise GenerationError("fp16 requires CUDA; use fp32 on CPU")
        return torch.float16
    return torch.float16 if device.type == "cuda" else torch.float32


def load_pipeline(
    base_model: str,
    *,
    revision: str | None,
    device: torch.device,
    dtype: torch.dtype,
) -> StableDiffusionPipeline:
    revision_options = {"revision": revision} if revision else {}
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        dtype=dtype,
        use_safetensors=True,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        **revision_options,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.to(device)
    return pipe


def load_lora_adapter(
    pipe: StableDiffusionPipeline,
    adapter_dir: Path,
    *,
    weights_file: str,
    adapter_name: str,
) -> None:
    pipe.unet.load_lora_adapter(
        str(adapter_dir),
        weight_name=weights_file,
        prefix=None,
        adapter_name=adapter_name,
    )
    pipe.unet.set_adapter(adapter_name)


def decode_latents(pipe: StableDiffusionPipeline, latents: torch.Tensor) -> Image.Image:
    with torch.no_grad():
        decoded = pipe.vae.decode(
            (latents / pipe.vae.config.scaling_factor).to(pipe.vae.dtype),
            return_dict=False,
        )[0]
    return pipe.image_processor.postprocess(decoded.detach(), output_type="pil")[0]


def generate_image(
    pipe: StableDiffusionPipeline,
    *,
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    seed: int,
    device: torch.device,
) -> Image.Image:
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
        output_type="latent",
    ).images
    return decode_latents(pipe, latents)
