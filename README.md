# lora-style-imagegen

A small, reproducible Stable Diffusion 1.5 LoRA project for one personal visual style or subject.

Current scope: Phase 2, dataset preparation and LoRA training.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- A CUDA T4 or better for real training; an Apple Silicon CPU is enough for dry runs

## Dataset preparation

1. Put 15 to 30 source images in `data/raw/`. Subdirectories are allowed.
2. Install dependencies:

   ```sh
   make install
   ```

3. Build the processed dataset:

   ```sh
   make prepare
   ```

   The command creates:

   - `data/processed/images/image_0001.png` and so on, each 512x512 RGB
   - `data/processed/metadata.jsonl` with `file_name` and `text` fields
   - `data/processed/caption_template.txt`
   - `data/processed/dataset_info.json`

4. Edit `data/processed/caption_template.txt` to describe your style or subject. The
   template must contain `{trigger}`. Then rebuild and revalidate:

   ```sh
   uv run prepare-dataset --force
   make validate
   ```

Use a different trigger token or output location when needed:

```sh
uv run prepare-dataset \
  --raw-dir data/raw \
  --output-dir data/processed \
  --trigger-token "sks style" \
  --caption-template "{trigger}, a hand-painted illustration"
```

Images are sorted by relative path, converted to RGB, resized to cover 512x512, and
center-cropped. Transparent pixels are composited onto white. The generated
`metadata.jsonl` can be edited directly, but rebuilding with `--force` overwrites it from
the template.

## LoRA training

Training targets a free Colab or Kaggle T4 with 16 GB VRAM. The default configuration in
`configs/train_lora.yaml` uses rank 16, fp16, gradient checkpointing, 8-bit Adam, batch
size 1 with gradient accumulation 4, 800 optimizer steps, and a learning rate of `1e-4`.

Check the complete pipeline locally without a GPU or base-model download:

```sh
make train-dry-run
```

The dry run validates `data/processed`, attaches LoRA to a tiny local UNet, runs three
optimizer steps, writes TensorBoard events, and exports a real safetensors adapter to
`outputs/lora-style/dry-run/lora/`. It does not generate sample images or exercise the
SD 1.5 download path.

Real training must run on a CUDA GPU:

```sh
make train
```

Useful overrides:

```sh
uv run train-lora \
  --config configs/train_lora.yaml \
  --dataset-dir data/processed \
  --output-dir outputs/lora-style \
  --max-train-steps 400
```

Outputs:

- `outputs/lora-style/lora/pytorch_lora_weights.safetensors`: final adapter
- `outputs/lora-style/lora/lora_metadata.json`: trigger, base model, prompts, and settings
- `outputs/lora-style/validation/step-XXXXXX/`: fixed-prompt, fixed-seed images
- `outputs/lora-style/checkpoint-XXXXXX/`: periodic safetensors adapters
- TensorBoard events under `outputs/lora-style/lora-style/`

Validation prompts use `{trigger}`; the trigger token itself is read from
`data/processed/dataset_info.json` unless `--trigger-token` is supplied. Set
`report_to: none` to disable logging, or `wandb` after installing `wandb` separately.

### Colab

Open `notebooks/train_colab.ipynb` in Colab, select a T4 GPU, set your repository URL, and
mount Drive. The notebook clones the repository, installs `uv`, syncs the training
extras, prepares the dataset if needed, runs training, and displays validation images.

## Development

```sh
make check
make pre-commit
```

`make check` runs Ruff and pytest. The pre-commit configuration also blocks files larger
than 500 KB, private keys, and common whitespace issues. Weights, processed images, raw
images, tokens, and secrets are ignored by git.
