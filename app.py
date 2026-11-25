"""
Entry point for the Patent Prior-Art Explorer.

HuggingFace Spaces runs this file (see app_file in README.md). It only
makes the `src/` package and the `apps/` folder importable and starts the
Gradio UI; all logic lives in src/patentsearch/ and apps/gradio_app.py.

Run locally with:   python app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from apps.gradio_app import CUSTOM_CSS, build_app  # noqa: E402

if __name__ == "__main__":
    app = build_app()
    app.launch(css=CUSTOM_CSS)
