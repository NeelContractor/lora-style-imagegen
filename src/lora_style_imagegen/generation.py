from __future__ import annotations

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


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
