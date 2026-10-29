# lora-style-imagegen

A small, reproducible Stable Diffusion 1.5 LoRA project for one personal visual style or subject.

Status: Functional version complete (v0.1.0). Optional follow-ups are listed in
[Backlog](#backlog).

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- A CUDA T4 or better for real training, comparison, and the demo; an Apple Silicon CPU is
  enough for dry runs and tests

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

## Base vs LoRA comparison

After training, compare the base model with the adapter using identical prompts, seeds,
and sampling settings:

```sh
make compare
```

This reads `outputs/lora-style/lora/lora_metadata.json`, generates the adapter's
validation prompts with the trigger token, and adds a matching version of each prompt
without the trigger. The default run produces two variants for every prompt and seed,
writes individual images under `outputs/comparison/base/` and
`outputs/comparison/lora/`, and builds:

- `outputs/comparison/comparison_grid.png`: rows are prompts, columns alternate base and
  LoRA for each seed
- `outputs/comparison/comparison_metadata.json`: model, prompt, seed, device, and file
  manifest

Override the defaults when needed:

```sh
uv run compare-lora \
  --adapter-dir outputs/lora-style/lora \
  --output-dir outputs/comparison \
  --prompt "{trigger}, a botanical study of ferns" \
  --prompt "a mushroom in a forest" \
  --seed 1234 \
  --steps 25
```

The comparison uses the safetensors adapter and the DPM-Solver scheduler used for
training validation. It uses CUDA fp16 automatically on a T4 and falls back to CPU fp32;
CPU generation is supported but slow and is not recommended on the 8 GB development
machine. No new dependencies are required.

On Colab, open `notebooks/compare_colab.ipynb`, set the repository URL, mount Drive, and
run the comparison cells. The adapter path defaults to
`MyDrive/lora-style-imagegen/outputs/lora-style/lora`.

## Gradio demo

Install the training and app extras:

```sh
make install
```

Start the demo with a local adapter:

```sh
make demo
```

Or provide the adapter path explicitly:

```sh
uv run style-demo \
  --adapter-dir outputs/lora-style/lora
```

Open the URL printed by Gradio. The app generates a base image and a LoRA image from the
same prompt and seed. The LoRA strength slider controls the adapter scale; setting it to
0 matches the base output. A seed of `-1` produces a random seed and reports the value
used.

The app loads the base model and adapter lazily on the first request, so the first
generation is slower. On Colab, open `notebooks/demo_colab.ipynb` and run its cells, or
run the commands directly with `--share` to create a public Gradio link:

```sh
uv sync --extra train --extra app --no-dev
uv run style-demo \
  --adapter-dir /content/drive/MyDrive/lora-style-imagegen/outputs/lora-style/lora \
  --share
```

The app reads `LORA_ADAPTER_DIR` and `LORA_BASE_MODEL` when the corresponding command-line
options are not supplied. `app/app.py` is the Gradio entry point for deployments.

Gradio 6.31 caps Pillow below 13, while Pillow 12.3 has known security advisories. The
project overrides the cap to install Pillow 13 and smoke-tests the Gradio UI against it.
`uv pip check` reports this intentional mismatch.

### Hosting notes

This demo is smoke-tested on a Colab T4. The `--share` URL is a temporary tunnel to the
Colab runtime, not permanent hosting; it stops when the runtime stops. There is no extra
Gradio fee, but the app uses the Colab GPU quota and anyone with the link can generate
images.

For permanent hosting, deploy `app/app.py` as a Hugging Face Space or another GPU host.
A free CPU Space is too slow for practical SD 1.5 inference; usable hosting requires GPU
hardware.

## Backlog

The functional project is complete. Optional follow-ups:

- **CI:** The GitHub Actions workflow was intentionally removed.
- **Permanent hosting:** Hugging Face Spaces is deferred. If the demo is made public,
  consider auth and rate limiting.
- **Release:** `v0.1.0` is not tagged yet; tag and push it when ready.
- **Model quality:** Prompt, LoRA strength, steps, or training hyperparameters can still
  be tuned.
- **Maintenance:** Revisit the Gradio 6.31/Pillow 13 override when Gradio supports Pillow
  13 natively. `uv pip check` currently reports this intentional mismatch.

## Development

```sh
make check
make build
make pre-commit
```

`make check` runs Ruff and pytest. `make build` writes the sdist and wheel to `dist/`,
which is ignored by git. The pre-commit configuration also blocks files larger than
500 KB, private keys, and common whitespace issues. Weights, processed images, raw images,
tokens, and secrets are ignored by git.
