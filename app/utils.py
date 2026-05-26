from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def unique_output_path(directory: str | Path, suffix: str) -> Path:
    return ensure_dir(directory) / f"{uuid4().hex}{suffix}"


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
