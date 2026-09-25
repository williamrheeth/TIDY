"""ONNX export and TensorRT engine building for TIDY.

Acceleration packages are imported lazily so the standard PyTorch inference
path keeps its minimal dependency set.
"""

from __future__ import annotations

import argparse
import gc
import sys
import warnings
from pathlib import Path

import torch

from .checkpoint import load_checkpoint
from .model import TIDYNet


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "weights" / "tidy.pth"
DEFAULT_ARTIFACT_DIRECTORY = PROJECT_ROOT / "artifacts"
DEFAULT_ONNX_MODEL = DEFAULT_ARTIFACT_DIRECTORY / "tidy.onnx"
DEFAULT_TENSORRT_ENGINE = DEFAULT_ARTIFACT_DIRECTORY / "tidy_fp16.engine"
ONNX_OPSET = 18


def export_onnx(
    checkpoint: str | Path,
    destination: str | Path = DEFAULT_ONNX_MODEL,
    *,
    force: bool = False,
) -> Path:
    """Export the released TIDY network with dynamic spatial dimensions."""
    output_path = Path(destination).expanduser().resolve()
    if output_path.exists() and not force:
        return output_path

    try:
        import onnx
    except ImportError as error:
        raise RuntimeError(
            "ONNX export requires the acceleration environment. Run: "
            "conda env create -f environment-accelerated.yml"
        ) from error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.temporary")
    temporary_path.unlink(missing_ok=True)

    print(f"Exporting ONNX model to {output_path} ...")
    model = TIDYNet()
    loaded_tensors = load_checkpoint(model, checkpoint)
    model.eval().requires_grad_(False)
    example = torch.zeros(1, 3, 256, 640, dtype=torch.float32)

    try:
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You are using the legacy TorchScript-based ONNX export.*",
                category=DeprecationWarning,
            )
            torch.onnx.export(
                model,
                example,
                temporary_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {2: "height", 3: "width"},
                    "output": {2: "height", 3: "width"},
                },
                opset_version=ONNX_OPSET,
                do_constant_folding=True,
                dynamo=False,
                external_data=False,
            )
        onnx.checker.check_model(str(temporary_path), full_check=False)
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        del model, example
        gc.collect()

    size_mib = output_path.stat().st_size / (1024**2)
    print(
        f"ONNX export complete: {loaded_tensors} tensors, opset {ONNX_OPSET}, "
        f"{size_mib:.1f} MiB."
    )
    return output_path


def _parse_shape(value: str) -> tuple[int, int, int, int]:
    normalized = value.lower().replace(" ", "")
    pieces = normalized.split("x")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("Shape must be HEIGHTxWIDTH, for example 256x640.")
    try:
        height, width = (int(piece) for piece in pieces)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Shape dimensions must be integers.") from error
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("Shape dimensions must be positive.")
    return (1, 3, height, width)


def build_tensorrt_engine(
    onnx_model: str | Path = DEFAULT_ONNX_MODEL,
    destination: str | Path = DEFAULT_TENSORRT_ENGINE,
    *,
    fp16: bool = True,
    minimum_shape: tuple[int, int, int, int] = (1, 3, 32, 32),
    optimum_shape: tuple[int, int, int, int] = (1, 3, 256, 640),
    maximum_shape: tuple[int, int, int, int] = (1, 3, 1080, 1920),
    workspace_gib: float = 4.0,
    force: bool = False,
) -> Path:
    """Build a dynamic-shape TensorRT engine from a TIDY ONNX model."""
    model_path = Path(onnx_model).expanduser().resolve()
    output_path = Path(destination).expanduser().resolve()
    if output_path.exists() and not force:
        return output_path
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("Building a TensorRT engine requires an NVIDIA CUDA GPU.")
    if workspace_gib <= 0:
        raise ValueError("TensorRT workspace size must be positive.")
    for minimum, optimum, maximum in zip(
        minimum_shape, optimum_shape, maximum_shape
    ):
        if not minimum <= optimum <= maximum:
            raise ValueError(
                "TensorRT shapes must satisfy minimum <= optimum <= maximum "
                "for every dimension."
            )

    try:
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError(
            "TensorRT is not installed. Create the acceleration environment with: "
            "conda env create -f environment-accelerated.yml"
        ) from error

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    print(f"Parsing ONNX model: {model_path}")
    if not parser.parse_from_file(str(model_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse the ONNX model:\n{errors}")
    if network.num_inputs != 1 or network.num_outputs != 1:
        raise RuntimeError(
            f"Expected one TensorRT input and output, got {network.num_inputs} and "
            f"{network.num_outputs}."
        )

    input_name = network.get_input(0).name
    profile = builder.create_optimization_profile()
    # TensorRT 10.9 validates by raising ValueError and returns None on success.
    profile.set_shape(input_name, minimum_shape, optimum_shape, maximum_shape)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gib * 1024**3)
    )
    config.add_optimization_profile(profile)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        # TIDY's channel normalization explicitly computes variance. Performing
        # that entire subgraph in FP16 can overflow on real checkpoint features,
        # even though the surrounding convolutions are safe in FP16.
        constrained_layers = 0
        for index in range(network.num_layers):
            layer = network.get_layer(index)
            if "/norm1/" not in layer.name and "/norm2/" not in layer.name:
                continue
            if layer.type == trt.LayerType.CONSTANT:
                continue
            layer.precision = trt.DataType.FLOAT
            for output_index in range(layer.num_outputs):
                output = layer.get_output(output_index)
                if output.dtype in {trt.DataType.FLOAT, trt.DataType.HALF}:
                    layer.set_output_type(output_index, trt.DataType.FLOAT)
            constrained_layers += 1
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        print(
            f"Keeping {constrained_layers} normalization layers in FP32 for "
            "numerical stability."
        )

    precision = "FP16" if fp16 else "FP32"
    print(
        f"Building {precision} TensorRT engine for "
        f"{minimum_shape[-2:]} .. {maximum_shape[-2:]} (optimum {optimum_shape[-2:]})."
    )
    print("This one-time optimization can take several minutes.")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT engine building failed; see the log above.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.temporary")
    temporary_path.write_bytes(serialized_engine)
    temporary_path.replace(output_path)
    size_mib = output_path.stat().st_size / (1024**2)
    print(f"TensorRT engine complete: {output_path} ({size_mib:.1f} MiB).")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tidy-deploy",
        description="Export TIDY to ONNX and build a TensorRT engine.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("onnx", help="Export the ONNX model.")
    export_parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    export_parser.add_argument("--output", type=Path, default=DEFAULT_ONNX_MODEL)
    export_parser.add_argument("--force", action="store_true")

    engine_parser = subparsers.add_parser(
        "tensorrt", help="Build a native TensorRT engine."
    )
    engine_parser.add_argument("--onnx-model", type=Path, default=DEFAULT_ONNX_MODEL)
    engine_parser.add_argument("--output", type=Path, default=DEFAULT_TENSORRT_ENGINE)
    engine_parser.add_argument("--fp32", action="store_true", help="Build an FP32 engine.")
    engine_parser.add_argument("--min-shape", type=_parse_shape, default=(1, 3, 32, 32))
    engine_parser.add_argument(
        "--opt-shape", type=_parse_shape, default=(1, 3, 256, 640)
    )
    engine_parser.add_argument(
        "--max-shape", type=_parse_shape, default=(1, 3, 1080, 1920)
    )
    engine_parser.add_argument("--workspace-gib", type=float, default=4.0)
    engine_parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "onnx":
            export_onnx(args.checkpoint, args.output, force=args.force)
        else:
            build_tensorrt_engine(
                args.onnx_model,
                args.output,
                fp16=not args.fp32,
                minimum_shape=args.min_shape,
                optimum_shape=args.opt_shape,
                maximum_shape=args.max_shape,
                workspace_gib=args.workspace_gib,
                force=args.force,
            )
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
