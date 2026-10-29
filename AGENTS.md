# AGENTS.md

## Project

Standalone Python project for fine-tuning Stable Diffusion 1.5 with LoRA on one visual
style or subject, then comparing and hosting the result.

- Python 3.11+, uv, `src/` layout.
- Local machine: MacBook Air M2, 8 GB RAM, no usable GPU. Use it for development, tests,
  and dry runs only.
- Real training targets a free Colab or Kaggle T4 with 16 GB VRAM.
- Base model: Stable Diffusion 1.5.
- Approved libraries: diffusers, peft, transformers, accelerate, bitsandbytes, gradio,
  huggingface_hub, Pillow, PyYAML, tensorboard. Ask before adding anything else.
- Never commit raw or processed images, model weights, generated outputs, tokens, or
  secrets.
- Status: Phases 1 and 2 are complete. Wait for approval before starting Phase 3.

## Commands

```sh
make install          # sync all dependency groups and training extras
make check            # ruff + pytest
make prepare          # rebuild data/processed from data/raw
make validate         # validate data/processed
make train-dry-run    # CPU-only pipeline check; no base model download
make train            # real training; intended for a T4 GPU
```

Run Python entry points through uv, for example:

```sh
uv run prepare-dataset --validate-only
uv run train-lora --config configs/train_lora.yaml --dry-run
```

## Layout

- `src/lora_style_imagegen/dataset_prep.py`: Phase 1 image prep and validation.
- `src/lora_style_imagegen/config.py`: strict YAML training config.
- `src/lora_style_imagegen/data.py`: caption record loading for training.
- `src/lora_style_imagegen/train.py`: training CLI, LoRA training, validation images,
  safetensors export, and CPU dry-run.
- `configs/train_lora.yaml`: default T4 training settings.
- `notebooks/train_colab.ipynb`: Colab wrapper that clones the repo and starts training.
- `tests/`: pytest coverage for config, data, dataset prep, and dry-run helpers.
- `app/`: reserved for Phase 4 Gradio hosting.

## Conventions

- Keep commits small and scoped.
- Add tests for behavior changes.
- Keep Ruff clean and format with Ruff.
- Do not add comments unless the user asks for them.
- Keep dry-run free of network access and GPU requirements.
- Do not run real base-model training locally.
- Stop after each requested phase and wait for approval before starting the next one.
