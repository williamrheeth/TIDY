"""Minimal RGB image input/output for TIDY inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor


IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})


def read_image(path: str | Path) -> Tensor:
    """Read any supported image as a normalized 1x3xHxW RGB tensor."""
    image_path = Path(path)
    try:
        with Image.open(image_path) as opened:
            rgb = ImageOps.exif_transpose(opened).convert("RGB")
            array = np.array(rgb, dtype=np.float32, copy=True) / 255.0
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not read image {image_path}: {error}") from error
    return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)


def write_image(image: Tensor, path: str | Path) -> None:
    """Clamp a 3xHxW RGB tensor to [0, 1] and write an 8-bit image."""
    output_path = Path(path)
    if output_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported output extension '{output_path.suffix}'. "
            f"Choose one of: {', '.join(sorted(IMAGE_EXTENSIONS))}."
        )
    array = (
        image.detach().squeeze(0).float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    )
    array = np.rint(array * 255.0).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_options = (
        {"quality": 95} if output_path.suffix.lower() in {".jpg", ".jpeg"} else {}
    )
    Image.fromarray(array, mode="RGB").save(output_path, **save_options)


def find_images(directory: str | Path, recursive: bool = False) -> list[Path]:
    """Return supported images in deterministic order."""
    root = Path(directory)
    candidates = root.rglob("*") if recursive else root.iterdir()
    return sorted(
        (
            path
            for path in candidates
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: str(path.relative_to(root)).lower(),
    )
