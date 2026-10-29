# lora-style-imagegen

A small, reproducible Stable Diffusion 1.5 LoRA project for one personal visual style or subject.

Current scope: Phase 1, dataset preparation.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

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

## Development

```sh
make check
make pre-commit
```

`make check` runs Ruff and pytest. The pre-commit configuration also blocks files larger
than 500 KB, private keys, and common whitespace issues. Weights, processed images, raw
images, tokens, and secrets are ignored by git.
