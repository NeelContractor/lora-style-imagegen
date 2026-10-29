from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

main = importlib.import_module("lora_style_imagegen.demo").main


if __name__ == "__main__":
    raise SystemExit(main())
