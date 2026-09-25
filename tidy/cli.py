"""Command-line inference interface for TIDY."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import torch

from . import __version__
from .backends import (
    BACKEND_NAMES,
    OnnxRuntimeBackend,
    PyTorchBackend,
    TensorRTBackend,
    normalize_backend,
)
from .deployment import (
    DEFAULT_ONNX_MODEL,
    build_tensorrt_engine,
    export_onnx,
)
from .images import IMAGE_EXTENSIONS, find_images, read_image, write_image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "weights" / "tidy.pth"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tidy",
        description="Denoise one thermal infrared image or a directory of images.",
    )
    parser.add_argument(
        "-i", "--input", required=True, type=Path, help="Input image or directory."
    )
    parser.add_argument(
        "-o", "--output", required=True, type=Path, help="Output image or directory."
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Model checkpoint (default: {DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, cuda, or cuda:N (default: auto).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use half precision on CUDA to reduce memory use and improve throughput.",
    )
    parser.add_argument(
        "--backend",
        choices=BACKEND_NAMES,
        default="pytorch",
        help=(
            "Inference backend: pytorch (default), onnx, tensorrt, or both "
            "(ONNX Runtime + TensorRT)."
        ),
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=DEFAULT_ONNX_MODEL,
        help=f"ONNX artifact (default: {DEFAULT_ONNX_MODEL}).",
    )
    parser.add_argument(
        "--tensorrt-engine",
        type=Path,
        default=None,
        help="Native TensorRT engine (default: artifacts/tidy_fp16.engine or tidy_fp32.engine).",
    )
    parser.add_argument(
        "--rebuild-artifacts",
        action="store_true",
        help="Re-export ONNX and/or rebuild TensorRT before inference.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input subdirectories and preserve their relative paths.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Leave existing output images unchanged.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"TIDY {__version__} | PyTorch {torch.__version__}",
    )
    return parser


def resolve_device(requested: str) -> torch.device:
    normalized = requested.lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(normalized)
    except RuntimeError as error:
        raise ValueError(f"Invalid device '{requested}'.") from error
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("TIDY supports CPU and CUDA devices only.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable. Check the NVIDIA driver and "
                "install the CUDA 12.8 PyTorch build from environment.yml."
            )
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device {index} does not exist; {torch.cuda.device_count()} device(s) are visible."
            )
    return device


def build_jobs(
    input_path: Path, output_path: Path, recursive: bool
) -> list[tuple[Path, Path]]:
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported input image extension: {input_path.suffix}")
        output_is_directory = (
            output_path.is_dir() or output_path.suffix.lower() not in IMAGE_EXTENSIONS
        )
        destination = (
            output_path / input_path.name if output_is_directory else output_path
        )
        jobs = [(input_path, destination)]
    elif input_path.is_dir():
        if output_path.exists() and not output_path.is_dir():
            raise ValueError("Output must be a directory when input is a directory.")
        images = find_images(input_path, recursive=recursive)
        if not images:
            scope = " or its subdirectories" if recursive else ""
            raise ValueError(f"No supported images found in {input_path}{scope}.")
        jobs = [
            (source, output_path / source.relative_to(input_path)) for source in images
        ]
    else:
        raise ValueError(f"Input must be an image or directory: {input_path}")

    for source, destination in jobs:
        if source == destination:
            raise ValueError(
                f"Refusing to overwrite the input image in place: {source}"
            )
    return jobs


def run(args: argparse.Namespace) -> int:
    jobs = build_jobs(args.input, args.output, args.recursive)
    if args.skip_existing:
        jobs = [(source, target) for source, target in jobs if not target.exists()]
        if not jobs:
            print("All output images already exist; nothing to do.")
            return 0

    device = resolve_device(args.device)
    backend_name = normalize_backend(args.backend)
    if args.fp16 and device.type != "cuda":
        raise ValueError("--fp16 requires a CUDA device.")
    if backend_name in {"tensorrt", "onnx-tensorrt"} and device.type != "cuda":
        raise ValueError(f"The {args.backend} backend requires a CUDA device.")

    onnx_path = args.onnx_model.expanduser().resolve()
    if backend_name in {"onnx", "tensorrt", "onnx-tensorrt"}:
        export_onnx(
            args.checkpoint,
            onnx_path,
            force=args.rebuild_artifacts,
        )

    engine_path: Path | None = None
    if backend_name == "tensorrt":
        engine_path = (
            args.tensorrt_engine.expanduser().resolve()
            if args.tensorrt_engine is not None
            else PROJECT_ROOT
            / "artifacts"
            / ("tidy_fp16.engine" if args.fp16 else "tidy_fp32.engine")
        )
        build_tensorrt_engine(
            onnx_path,
            engine_path,
            fp16=args.fp16,
            force=args.rebuild_artifacts,
        )

    print(f"TIDY {__version__}")
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
        print(
            f"Device: cuda:{index} ({torch.cuda.get_device_name(index)}) | "
            f"PyTorch {torch.__version__} | CUDA {torch.version.cuda}"
        )
    else:
        print(f"Device: CPU | PyTorch {torch.__version__}")
    checkpoint = args.checkpoint.expanduser().resolve()
    print(f"Backend: {args.backend}")
    if backend_name == "pytorch":
        backend = PyTorchBackend(checkpoint, device, fp16=args.fp16)
        print(f"Checkpoint: {checkpoint}")
        print(f"Loaded {backend.loaded_tensors} learned tensors.")
    elif backend_name == "onnx":
        backend = OnnxRuntimeBackend(onnx_path, device, fp16=args.fp16)
        print(f"ONNX model: {onnx_path}")
    elif backend_name == "onnx-tensorrt":
        cache_directory = PROJECT_ROOT / "artifacts" / "ort-trt-cache"
        if args.rebuild_artifacts and cache_directory.exists():
            shutil.rmtree(cache_directory)
        backend = OnnxRuntimeBackend(
            onnx_path,
            device,
            tensorrt=True,
            fp16=args.fp16,
            cache_directory=cache_directory,
        )
        print(f"ONNX model: {onnx_path}")
        print(f"TensorRT cache: {cache_directory}")
    else:
        assert engine_path is not None
        backend = TensorRTBackend(engine_path, device, fp16=args.fp16)
        print(f"TensorRT engine: {engine_path}")
    print(f"Runtime: {backend.label}; processing {len(jobs)} image(s).")

    total_seconds = 0.0
    with torch.inference_mode():
        for number, (source, destination) in enumerate(jobs, start=1):
            image = read_image(source)
            backend.synchronize()
            started = time.perf_counter()
            result = backend.infer(image)
            backend.synchronize()
            elapsed = time.perf_counter() - started
            total_seconds += elapsed
            write_image(result, destination)
            print(f"[{number}/{len(jobs)}] {source} -> {destination} ({elapsed:.3f} s)")

    print(f"Done: {len(jobs)} image(s) in {total_seconds:.3f} s model time.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        return run(parser.parse_args(argv))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
