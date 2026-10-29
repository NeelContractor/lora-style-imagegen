from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from peft import LoraConfig
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from transformers import CLIPTextModel, CLIPTokenizer, get_scheduler

from lora_style_imagegen.config import (
    ConfigError,
    TrainConfig,
    format_prompts,
    load_config,
    validate_config,
)
from lora_style_imagegen.data import (
    CaptionRecord,
    DataError,
    image_to_tensor,
    load_caption_records,
)
from lora_style_imagegen.dataset_prep import DatasetError, validate_dataset

LORA_WEIGHT_FILE = "pytorch_lora_weights.safetensors"
DRY_RUN_STEPS = 3
DRY_RUN_BATCH = 4
DRY_RUN_SAMPLES = 8
DRY_RUN_CROSS_ATTENTION_DIM = 32
DRY_RUN_SAMPLE_SIZE = 8


class TrainingError(Exception):
    pass


@dataclass(frozen=True)
class TrainingResult:
    output_dir: Path
    artifact_dir: Path
    steps: int
    trainable_parameters: int
    dry_run: bool


@dataclass(frozen=True)
class PreparedDataset:
    latents: torch.Tensor
    prompt_embeds: torch.Tensor


def run_training(config: TrainConfig) -> TrainingResult:
    if not torch.cuda.is_available():
        raise TrainingError(
            "Real training requires a CUDA GPU. Use --dry-run locally, or run this on "
            "Colab/Kaggle with a T4 GPU."
        )

    set_seed(config.seed)
    trigger_token = _resolve_trigger(config)
    prompts = format_prompts(config.validation_prompts, trigger_token)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = _make_accelerator(config, config.output_dir, config.mixed_precision)
    dtype = torch.float16 if config.mixed_precision == "fp16" else torch.float32

    try:
        components = _load_components(config, accelerator.device, dtype)
        prepared = _precompute_dataset(
            config=config,
            records=load_caption_records(config.dataset_dir),
            tokenizer=components["tokenizer"],
            text_encoder=components["text_encoder"],
            vae=components["vae"],
            device=accelerator.device,
        )
        components["vae"].to("cpu")
        components["text_encoder"].to("cpu")
        _empty_cuda_cache()

        unet = components["unet"]
        noise_scheduler = components["scheduler"]
        trainable_parameters = _add_lora(unet, config)
        optimizer = _create_optimizer(unet, config, use_8bit=config.use_8bit_adam)
        lr_scheduler = get_scheduler(
            config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=config.max_train_steps,
        )

        validation_pipe = _build_validation_pipeline(
            components=components,
            unet=unet,
            noise_scheduler=noise_scheduler,
        )

        dataloader = DataLoader(
            TensorDataset(prepared.latents, prepared.prompt_embeds),
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=False,
        )
        unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

        validation_callback = _make_validation_callback(
            accelerator=accelerator,
            config=config,
            pipe=validation_pipe,
            unet=unet,
            prompts=prompts,
        )

        metadata = _artifact_metadata(
            config=config,
            trigger_token=trigger_token,
            prompts=prompts,
            trainable_parameters=trainable_parameters,
            steps=config.max_train_steps,
        )

        def save_callback(step: int) -> None:
            checkpoint_dir = config.output_dir / f"checkpoint-{step:06d}"
            _save_lora_artifact(
                accelerator=accelerator,
                unet=unet,
                directory=checkpoint_dir,
                metadata={**metadata, "step": step},
            )

        steps = _train_loop(
            accelerator=accelerator,
            unet=unet,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            dataloader=dataloader,
            noise_scheduler=noise_scheduler,
            config=config,
            device=accelerator.device,
            dtype=dtype,
            validation_callback=validation_callback,
            save_callback=save_callback,
            validation_steps=config.validation_steps,
            save_steps=config.save_steps,
            logging_steps=config.logging_steps,
        )

        artifact_dir = config.output_dir / "lora"
        _save_lora_artifact(
            accelerator=accelerator,
            unet=unet,
            directory=artifact_dir,
            metadata={**metadata, "step": steps},
        )
        accelerator.wait_for_everyone()
        return TrainingResult(
            output_dir=config.output_dir,
            artifact_dir=artifact_dir,
            steps=steps,
            trainable_parameters=trainable_parameters,
            dry_run=False,
        )
    finally:
        accelerator.end_training()


def run_dry_run(config: TrainConfig) -> TrainingResult:
    set_seed(config.seed)
    trigger_token = _resolve_trigger(config)
    prompts = format_prompts(config.validation_prompts, trigger_token)
    output_dir = config.output_dir / "dry-run"
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = _make_accelerator(config, output_dir, "no", force_cpu=True)
    try:
        unet = _build_tiny_unet()
        trainable_parameters = _add_lora(unet, config)
        optimizer = _create_optimizer(unet, config, use_8bit=False)
        lr_scheduler = get_scheduler(
            config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=min(config.lr_warmup_steps, DRY_RUN_STEPS),
            num_training_steps=DRY_RUN_STEPS,
        )
        noise_scheduler = _build_dry_run_scheduler()

        generator = torch.Generator().manual_seed(config.seed)
        latents = torch.randn(
            DRY_RUN_SAMPLES,
            4,
            DRY_RUN_SAMPLE_SIZE,
            DRY_RUN_SAMPLE_SIZE,
            generator=generator,
        )
        prompt_embeds = torch.randn(
            DRY_RUN_SAMPLES,
            77,
            DRY_RUN_CROSS_ATTENTION_DIM,
            generator=generator,
        )
        dataloader = DataLoader(
            TensorDataset(latents, prompt_embeds),
            batch_size=min(config.batch_size, DRY_RUN_BATCH),
            shuffle=True,
            drop_last=False,
        )
        unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

        dry_config = replace(
            config,
            max_train_steps=DRY_RUN_STEPS,
            lr_warmup_steps=min(config.lr_warmup_steps, DRY_RUN_STEPS),
            logging_steps=1,
            save_steps=DRY_RUN_STEPS + 1,
            validation_steps=DRY_RUN_STEPS + 1,
        )
        steps = _train_loop(
            accelerator=accelerator,
            unet=unet,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            dataloader=dataloader,
            noise_scheduler=noise_scheduler,
            config=dry_config,
            device=accelerator.device,
            dtype=torch.float32,
            validation_callback=None,
            save_callback=None,
            validation_steps=dry_config.validation_steps,
            save_steps=dry_config.save_steps,
            logging_steps=dry_config.logging_steps,
        )

        artifact_dir = output_dir / "lora"
        metadata = _artifact_metadata(
            config=config,
            trigger_token=trigger_token,
            prompts=prompts,
            trainable_parameters=trainable_parameters,
            steps=steps,
        )
        _save_lora_artifact(
            accelerator=accelerator,
            unet=unet,
            directory=artifact_dir,
            metadata={**metadata, "dry_run": True, "dataset_images": _dataset_count(config)},
        )
        (output_dir / "dry_run_report.json").write_text(
            json.dumps(
                {
                    "dataset_dir": str(config.dataset_dir),
                    "trigger_token": trigger_token,
                    "trainable_parameters": trainable_parameters,
                    "steps": steps,
                    "artifact": str(artifact_dir / LORA_WEIGHT_FILE),
                    "validation_images": "skipped; requires base SD 1.5 components",
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        accelerator.wait_for_everyone()
        return TrainingResult(
            output_dir=output_dir,
            artifact_dir=artifact_dir,
            steps=steps,
            trainable_parameters=trainable_parameters,
            dry_run=True,
        )
    finally:
        accelerator.end_training()


def _resolve_trigger(config: TrainConfig) -> str:
    try:
        report = validate_dataset(
            config.dataset_dir,
            trigger_token=config.trigger_token,
            image_size=config.resolution,
        )
    except DatasetError as exc:
        raise TrainingError(str(exc)) from exc
    return report.trigger_token


def _dataset_count(config: TrainConfig) -> int:
    return len(load_caption_records(config.dataset_dir))


def _make_accelerator(
    config: TrainConfig,
    output_dir: Path,
    mixed_precision: str,
    *,
    force_cpu: bool = False,
) -> Accelerator:
    log_with = None if config.report_to == "none" else config.report_to
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=log_with,
        project_dir=str(output_dir),
        cpu=force_cpu,
    )
    if log_with is not None:
        accelerator.init_trackers(config.run_name, config=_config_payload(config))
    return accelerator


def _config_payload(config: TrainConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key, value in payload.items():
        if value is None:
            payload[key] = "null"
        elif isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, tuple | list):
            payload[key] = ", ".join(str(item) for item in value)
    return payload


def _load_components(
    config: TrainConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    revision_options = (
        {"revision": config.base_model_revision} if config.base_model_revision else {}
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        config.base_model,
        subfolder="tokenizer",
        **revision_options,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        config.base_model,
        subfolder="text_encoder",
        dtype=dtype,
        use_safetensors=True,
        **revision_options,
    )
    vae = AutoencoderKL.from_pretrained(
        config.base_model,
        subfolder="vae",
        torch_dtype=dtype,
        use_safetensors=True,
        **revision_options,
    )
    unet = UNet2DConditionModel.from_pretrained(
        config.base_model,
        subfolder="unet",
        torch_dtype=dtype,
        use_safetensors=True,
        **revision_options,
    )
    scheduler = DDPMScheduler.from_pretrained(
        config.base_model,
        subfolder="scheduler",
        **revision_options,
    )

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.eval()
    vae.eval()
    text_encoder.to(device)
    vae.to(device)
    unet.to(device)

    return {
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "unet": unet,
        "scheduler": scheduler,
    }


def _precompute_dataset(
    *,
    config: TrainConfig,
    records: list[CaptionRecord],
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    vae: AutoencoderKL,
    device: torch.device,
) -> PreparedDataset:
    latents: list[torch.Tensor] = []
    prompt_embeds: list[torch.Tensor] = []
    encode_batch_size = 4

    for start in range(0, len(records), encode_batch_size):
        batch = records[start : start + encode_batch_size]
        images = torch.stack([image_to_tensor(record.path, config.resolution) for record in batch])
        captions = [record.text for record in batch]
        with torch.no_grad():
            pixel_values = images.to(device=device, dtype=vae.dtype)
            encoded = vae.encode(pixel_values).latent_dist.sample()
            encoded = encoded * vae.config.scaling_factor

            tokens = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            embeddings = text_encoder(tokens.input_ids.to(device)).last_hidden_state

        latents.extend(tensor.detach().cpu() for tensor in encoded)
        prompt_embeds.extend(tensor.detach().cpu() for tensor in embeddings)
        print(f"Encoded {min(start + len(batch), len(records))}/{len(records)} images")

    return PreparedDataset(
        latents=torch.stack(latents),
        prompt_embeds=torch.stack(prompt_embeds),
    )


def _add_lora(unet: UNet2DConditionModel, config: TrainConfig) -> int:
    unet.requires_grad_(False)
    unet.add_adapter(
        LoraConfig(
            r=config.rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.target_modules),
            init_lora_weights="gaussian",
        )
    )
    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    trainable_parameters = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise TrainingError("LoRA adapter added no trainable parameters")
    for parameter in trainable_parameters:
        parameter.data = parameter.data.float()
    return sum(parameter.numel() for parameter in trainable_parameters)


def _create_optimizer(
    unet: UNet2DConditionModel,
    config: TrainConfig,
    *,
    use_8bit: bool,
) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    if use_8bit and torch.cuda.is_available():
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise TrainingError(
                "8-bit Adam requires bitsandbytes; install the training extras on Linux"
            ) from exc
        return bnb.optim.AdamW8bit(
            parameters,
            lr=config.learning_rate,
            weight_decay=1e-2,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    return torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=1e-2,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def _train_loop(
    *,
    accelerator: Accelerator,
    unet: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    dataloader: DataLoader,
    noise_scheduler: DDPMScheduler,
    config: TrainConfig,
    device: torch.device,
    dtype: torch.dtype,
    validation_callback: Callable[[int], None] | None,
    save_callback: Callable[[int], None] | None,
    validation_steps: int,
    save_steps: int,
    logging_steps: int,
) -> int:
    global_step = 0
    last_loss = 0.0
    while global_step < config.max_train_steps:
        for latents, prompt_embeds in dataloader:
            with accelerator.accumulate(unet):
                latents = latents.to(device=device, dtype=dtype)
                prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
                loss = _training_loss(unet, noise_scheduler, latents, prompt_embeds)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [parameter for parameter in unet.parameters() if parameter.requires_grad],
                        config.max_grad_norm,
                    )
                optimizer.step()
                if accelerator.sync_gradients:
                    lr_scheduler.step()
                optimizer.zero_grad()

            last_loss = loss.detach().float().item()
            if not accelerator.sync_gradients:
                continue

            global_step += 1
            if global_step % logging_steps == 0 or global_step == 1:
                learning_rate = lr_scheduler.get_last_lr()[0]
                accelerator.log(
                    {"train/loss": last_loss, "train/learning_rate": learning_rate},
                    step=global_step,
                )
                print(f"step {global_step}/{config.max_train_steps} loss={last_loss:.4f}")

            if validation_callback and (
                global_step % validation_steps == 0 or global_step == config.max_train_steps
            ):
                validation_callback(global_step)

            if (
                save_callback
                and global_step % save_steps == 0
                and global_step != config.max_train_steps
            ):
                save_callback(global_step)

            if global_step >= config.max_train_steps:
                break

    return global_step


def _training_loss(
    unet: torch.nn.Module,
    noise_scheduler: DDPMScheduler,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
) -> torch.Tensor:
    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (latents.shape[0],),
        device=latents.device,
        dtype=torch.long,
    )
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    model_prediction = unet(noisy_latents, timesteps, encoder_hidden_states=prompt_embeds).sample

    if noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        target = noise
    return F.mse_loss(model_prediction.float(), target.float(), reduction="mean")


def _build_validation_pipeline(
    *,
    components: dict[str, Any],
    unet: UNet2DConditionModel,
    noise_scheduler: DDPMScheduler,
) -> StableDiffusionPipeline:
    validation_scheduler = DPMSolverMultistepScheduler.from_config(noise_scheduler.config)
    pipe = StableDiffusionPipeline(
        vae=components["vae"],
        text_encoder=components["text_encoder"],
        tokenizer=components["tokenizer"],
        unet=unet,
        scheduler=validation_scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    return pipe


def _make_validation_callback(
    *,
    accelerator: Accelerator,
    config: TrainConfig,
    pipe: StableDiffusionPipeline,
    unet: torch.nn.Module,
    prompts: tuple[str, ...],
) -> Callable[[int], None]:
    def validate(step: int) -> None:
        if not accelerator.is_main_process:
            return
        pipe.unet = accelerator.unwrap_model(unet)
        pipe.unet.eval()
        pipe.vae.to(accelerator.device)
        pipe.text_encoder.to(accelerator.device)

        output_dir = config.output_dir / "validation" / f"step-{step:06d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        images: dict[str, torch.Tensor] = {}
        try:
            for index, prompt in enumerate(prompts):
                seed = config.validation_seeds[index % len(config.validation_seeds)]
                generator = torch.Generator(device=accelerator.device).manual_seed(seed)
                image = pipe(
                    prompt=prompt,
                    negative_prompt=config.validation_negative_prompt,
                    num_inference_steps=config.validation_inference_steps,
                    guidance_scale=config.validation_guidance_scale,
                    generator=generator,
                ).images[0]
                image.save(output_dir / f"{index:02d}.png")
                images[f"validation/{index:02d}"] = _image_to_tensorboard(image).unsqueeze(0)
                print(f"validation step {step}: saved {output_dir / f'{index:02d}.png'}")
        finally:
            pipe.vae.to("cpu")
            pipe.text_encoder.to("cpu")
            pipe.unet.train()
            _empty_cuda_cache()

        if config.report_to == "tensorboard" and images:
            tracker = accelerator.get_tracker("tensorboard")
            if tracker is not None and hasattr(tracker, "log_images"):
                tracker.log_images(images, step=step)

    return validate


def _save_lora_artifact(
    *,
    accelerator: Accelerator,
    unet: torch.nn.Module,
    directory: Path,
    metadata: dict[str, Any],
) -> None:
    if not accelerator.is_main_process:
        return

    unwrapped_unet = accelerator.unwrap_model(unet)
    directory.mkdir(parents=True, exist_ok=True)
    unwrapped_unet.save_lora_adapter(
        directory,
        adapter_name="default",
        safe_serialization=True,
    )
    weight_path = directory / LORA_WEIGHT_FILE
    if not weight_path.is_file():
        raise TrainingError(f"LoRA export did not create {weight_path}")

    payload = {
        **metadata,
        "format": "diffusers-lora",
        "weights_file": LORA_WEIGHT_FILE,
    }
    (directory / "lora_metadata.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved LoRA weights to {weight_path}")


def _artifact_metadata(
    *,
    config: TrainConfig,
    trigger_token: str,
    prompts: tuple[str, ...],
    trainable_parameters: int,
    steps: int,
) -> dict[str, Any]:
    return {
        "base_model": config.base_model,
        "base_model_revision": config.base_model_revision,
        "trigger_token": trigger_token,
        "rank": config.rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "target_modules": list(config.target_modules),
        "resolution": config.resolution,
        "trainable_parameters": trainable_parameters,
        "steps": steps,
        "validation_prompts": list(prompts),
        "validation_seeds": list(config.validation_seeds),
        "validation_guidance_scale": config.validation_guidance_scale,
        "validation_inference_steps": config.validation_inference_steps,
    }


def _build_tiny_unet() -> UNet2DConditionModel:
    return UNet2DConditionModel(
        sample_size=DRY_RUN_SAMPLE_SIZE,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=DRY_RUN_CROSS_ATTENTION_DIM,
        attention_head_dim=8,
        norm_num_groups=32,
    )


def _build_dry_run_scheduler() -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
    )


def _image_to_tensorboard(image: Image.Image) -> torch.Tensor:
    rgb = image.convert("RGB")
    buffer = bytearray(rgb.tobytes())
    tensor = torch.frombuffer(buffer, dtype=torch.uint8).reshape(rgb.height, rgb.width, 3)
    return tensor.permute(2, 0, 1).float().div(255.0)


def _empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a Stable Diffusion 1.5 UNet LoRA on a prepared dataset."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/train_lora.yaml"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Exercise data validation, LoRA attach, training steps, and safetensors export "
        "with a tiny local UNet; no base model download or GPU.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--trigger-token", default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--report-to",
        choices=("none", "tensorboard", "wandb"),
        default=None,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        overrides: dict[str, Any] = {}
        if args.dataset_dir is not None:
            overrides["dataset_dir"] = args.dataset_dir
        if args.output_dir is not None:
            overrides["output_dir"] = args.output_dir
        if args.trigger_token is not None:
            overrides["trigger_token"] = args.trigger_token
        if args.max_train_steps is not None:
            overrides["max_train_steps"] = args.max_train_steps
        if args.seed is not None:
            overrides["seed"] = args.seed
        if args.report_to is not None:
            overrides["report_to"] = args.report_to
        if overrides:
            config = replace(config, **overrides)
            validate_config(config)

        result = run_dry_run(config) if args.dry_run else run_training(config)
    except (ConfigError, DataError, DatasetError, TrainingError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    mode = "Dry run" if result.dry_run else "Training"
    print(f"{mode} complete: {result.steps} optimizer steps")
    print(f"LoRA: {result.artifact_dir / LORA_WEIGHT_FILE}")
    print(f"Trainable parameters: {result.trainable_parameters:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
