"""Configurable scorer assets, separate from the Hugging Face cache."""

import os
from pathlib import Path

MODELS_DIR = (
    Path(os.environ.get("SERF_ASSETS_DIR", Path.home() / ".cache/serf/assets"))
    .expanduser()
    .resolve()
)


def model_path(filename):
    return MODELS_DIR / filename
