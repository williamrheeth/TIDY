"""Checkpoint loading with validation and legacy TIDY compatibility."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


LEGACY_WAVELET_PREFIXES = ("dwt.", "idwt.")
COMMON_STATE_KEYS = ("params", "params_ema", "state_dict", "model")
COMMON_PREFIXES = ("module.", "net_g.", "model.")


def _is_state_dict(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(isinstance(item, Tensor) for item in value.values())
    )


def _extract_state_dict(checkpoint: Any) -> Mapping[str, Tensor]:
    if _is_state_dict(checkpoint):
        return checkpoint
    if isinstance(checkpoint, Mapping):
        for key in COMMON_STATE_KEYS:
            candidate = checkpoint.get(key)
            if _is_state_dict(candidate):
                return candidate
    raise ValueError(
        "Checkpoint does not contain a tensor state dictionary. Expected a raw "
        "state dict or one under: " + ", ".join(COMMON_STATE_KEYS) + "."
    )


def _remove_common_prefix(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in COMMON_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
                break
    return key


def load_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> int:
    """Load and strictly validate learned TIDY parameters.

    Returns the number of loaded tensors. The fixed filter buffers embedded by
    the former wavelet package are safely ignored because TIDY now computes the
    same Haar transform directly.
    """
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    extracted = _extract_state_dict(checkpoint)
    state = {
        _remove_common_prefix(key): tensor
        for key, tensor in extracted.items()
        if not _remove_common_prefix(key).startswith(LEGACY_WAVELET_PREFIXES)
    }

    expected = model.state_dict()
    missing = sorted(expected.keys() - state.keys())
    unexpected = sorted(state.keys() - expected.keys())
    wrong_shapes = sorted(
        key
        for key in expected.keys() & state.keys()
        if expected[key].shape != state[key].shape
    )
    if missing or unexpected or wrong_shapes:
        details: list[str] = []
        if missing:
            details.append("missing keys: " + ", ".join(missing[:10]))
        if unexpected:
            details.append("unexpected keys: " + ", ".join(unexpected[:10]))
        if wrong_shapes:
            shape_details = ", ".join(
                f"{key} (expected {tuple(expected[key].shape)}, got {tuple(state[key].shape)})"
                for key in wrong_shapes[:10]
            )
            details.append("shape mismatches: " + shape_details)
        raise RuntimeError(
            "Checkpoint is incompatible with TIDYNet; " + "; ".join(details)
        )

    model.load_state_dict(state, strict=True)
    return len(state)
