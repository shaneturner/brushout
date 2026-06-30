"""Pre-download all models used by the bundled app."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from model import (
    _download, _simplify_lama,
    _LAMA_REPO, _LAMA_FILE,
    _MIGAN_REPO, _MIGAN_FILE,
)

lama_path = _download(_LAMA_REPO, _LAMA_FILE, "~200MB")
_simplify_lama(lama_path)
_download("opencv/inpainting_lama", "inpainting_lama_2025jan.onnx", "~100MB")
_download(_MIGAN_REPO, _MIGAN_FILE, "~26MB")
print("All models ready.")
