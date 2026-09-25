"""Interchangeable inference backends for TIDY."""

from __future__ import annotations

import ctypes
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .checkpoint import load_checkpoint
from .model import TIDYNet


BACKEND_NAMES = ("pytorch", "onnx", "tensorrt", "onnx-tensorrt", "both")


def normalize_backend(name: str) -> str:
    """Normalize the convenient ``both`` alias to ONNX Runtime TensorRT."""
    normalized = name.lower()
    if normalized not in BACKEND_NAMES:
        raise ValueError(
            f"Unknown backend '{name}'. Choose one of: {', '.join(BACKEND_NAMES)}."
        )
    return "onnx-tensorrt" if normalized == "both" else normalized


def cuda_device_index(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else device.index


class InferenceBackend(ABC):
    """Small common interface used by the image-folder CLI."""

    label: str

    @abstractmethod
    def infer(self, image: Tensor) -> Tensor:
        """Denoise one Bx3xHxW image tensor."""

    def synchronize(self) -> None:
        """Wait until queued backend work is complete for accurate timing."""


class PyTorchBackend(InferenceBackend):
    def __init__(
        self, checkpoint: Path, device: torch.device, fp16: bool = False
    ) -> None:
        self.device = device
        self.fp16 = fp16
        self.model = TIDYNet()
        self.loaded_tensors = load_checkpoint(self.model, checkpoint)
        self.model.eval().requires_grad_(False)
        if fp16:
            self.model.half()
        self.model.to(device)
        self.label = f"PyTorch {'FP16' if fp16 else 'FP32'}"

    def infer(self, image: Tensor) -> Tensor:
        if self.fp16:
            image = image.half()
        image = image.to(self.device, non_blocking=self.device.type == "cuda")
        return self.model(image)

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


class OnnxRuntimeBackend(InferenceBackend):
    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        *,
        tensorrt: bool = False,
        fp16: bool = False,
        cache_directory: Path | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError(
                "ONNX Runtime is not installed. Create the acceleration environment "
                "with: conda env create -f environment-accelerated.yml"
            ) from error

        if fp16 and not tensorrt:
            raise ValueError(
                "--fp16 is supported by the pytorch, tensorrt, and both backends; "
                "the plain onnx backend runs this exported model in FP32."
            )
        self.device = device
        self.ort = ort
        if hasattr(ort, "preload_dlls"):
            # Reuse CUDA and cuDNN libraries supplied by the PyTorch CUDA wheel.
            ort.preload_dlls()
        if tensorrt:
            _preload_tensorrt_libraries()

        provider_options: list[Any]
        if device.type == "cuda":
            index = cuda_device_index(device)
            cuda_options = {"device_id": index}
            if tensorrt:
                cache = (cache_directory or model_path.parent / "ort-trt-cache").resolve()
                cache.mkdir(parents=True, exist_ok=True)
                trt_options = {
                    "device_id": index,
                    "trt_fp16_enable": fp16,
                    "trt_layer_norm_fp32_fallback": fp16,
                    "trt_profile_min_shapes": "input:1x3x32x32",
                    "trt_profile_opt_shapes": "input:1x3x256x640",
                    "trt_profile_max_shapes": "input:1x3x1080x1920",
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(cache),
                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": str(cache),
                }
                provider_options = [
                    ("TensorrtExecutionProvider", trt_options),
                    ("CUDAExecutionProvider", cuda_options),
                    "CPUExecutionProvider",
                ]
            else:
                provider_options = [
                    ("CUDAExecutionProvider", cuda_options),
                    "CPUExecutionProvider",
                ]
        else:
            if tensorrt:
                raise ValueError("The both/onnx-tensorrt backend requires CUDA.")
            provider_options = ["CPUExecutionProvider"]

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=provider_options
        )
        active = self.session.get_providers()
        required_provider = (
            "TensorrtExecutionProvider"
            if tensorrt
            else "CUDAExecutionProvider"
            if device.type == "cuda"
            else "CPUExecutionProvider"
        )
        if required_provider not in active:
            raise RuntimeError(
                f"ONNX Runtime could not activate {required_provider}. Active providers: "
                f"{', '.join(active)}. Check the acceleration environment installation."
            )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.label = (
            f"ONNX Runtime TensorRT {'FP16' if fp16 else 'FP32'}"
            if tensorrt
            else f"ONNX Runtime FP32 ({required_provider})"
        )

    def infer(self, image: Tensor) -> Tensor:
        image = image.float().contiguous()
        if self.device.type == "cpu":
            output = self.session.run(
                [self.output_name], {self.input_name: image.numpy()}
            )[0]
            return torch.from_numpy(output)

        image = image.to(self.device, non_blocking=True)
        output = torch.empty_like(image)
        binding = self.session.io_binding()
        index = cuda_device_index(self.device)
        binding.bind_input(
            self.input_name,
            device_type="cuda",
            device_id=index,
            element_type=np.float32,
            shape=tuple(image.shape),
            buffer_ptr=image.data_ptr(),
        )
        binding.bind_output(
            self.output_name,
            device_type="cuda",
            device_id=index,
            element_type=np.float32,
            shape=tuple(output.shape),
            buffer_ptr=output.data_ptr(),
        )
        self.session.run_with_iobinding(binding)
        return output

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


class TensorRTBackend(InferenceBackend):
    def __init__(self, engine_path: Path, device: torch.device, fp16: bool) -> None:
        if device.type != "cuda":
            raise ValueError("The tensorrt backend requires a CUDA device.")
        try:
            import tensorrt as trt
        except ImportError as error:
            raise RuntimeError(
                "TensorRT is not installed. Create the acceleration environment with: "
                "conda env create -f environment-accelerated.yml"
            ) from error

        self.trt = trt
        self.device = device
        with torch.cuda.device(device):
            logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
            self.stream = torch.cuda.Stream(device=device)
        if self.engine is None:
            raise RuntimeError(
                "TensorRT could not load the engine. Engines are GPU- and TensorRT-"
                "version-specific; rebuild it with tidy-deploy tensorrt --force."
            )
        self.context = self.engine.create_execution_context()
        inputs: list[str] = []
        outputs: list[str] = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            (inputs if mode == trt.TensorIOMode.INPUT else outputs).append(name)
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(
                f"Expected one TensorRT input and output, got {len(inputs)} and "
                f"{len(outputs)}."
            )
        self.input_name = inputs[0]
        self.output_name = outputs[0]
        self.label = f"TensorRT {'FP16' if fp16 else 'FP32'}"

    def infer(self, image: Tensor) -> Tensor:
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            image = image.float().to(self.device, non_blocking=True).contiguous()
            shape = tuple(image.shape)
            if not self.context.set_input_shape(self.input_name, shape):
                minimum, optimum, maximum = self.engine.get_tensor_profile_shape(
                    self.input_name, 0
                )
                raise ValueError(
                    f"Input shape {shape} is outside this TensorRT engine profile "
                    f"({tuple(minimum)} .. {tuple(maximum)}, optimum {tuple(optimum)})."
                )
            unresolved = self.context.infer_shapes()
            if unresolved:
                raise RuntimeError(
                    "TensorRT could not resolve tensor shapes: " + ", ".join(unresolved)
                )
            output_shape = tuple(self.context.get_tensor_shape(self.output_name))
            output_dtype = _torch_dtype(
                self.trt.nptype(self.engine.get_tensor_dtype(self.output_name))
            )
            output = torch.empty(output_shape, dtype=output_dtype, device=self.device)
            self.context.set_tensor_address(self.input_name, image.data_ptr())
            self.context.set_tensor_address(self.output_name, output.data_ptr())
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT inference failed.")
        return output

    def synchronize(self) -> None:
        self.stream.synchronize()


def _torch_dtype(numpy_dtype: Any) -> torch.dtype:
    mapping = {
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float16): torch.float16,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.int64): torch.int64,
        np.dtype(np.bool_): torch.bool,
    }
    dtype = np.dtype(numpy_dtype)
    if dtype not in mapping:
        raise RuntimeError(f"Unsupported TensorRT output dtype: {dtype}.")
    return mapping[dtype]


def _preload_tensorrt_libraries() -> None:
    """Expose pip-installed TensorRT libraries to ONNX Runtime's provider."""
    try:
        import tensorrt_libs
    except ImportError as error:
        raise RuntimeError(
            "TensorRT native libraries are missing. Recreate the acceleration "
            "environment from environment-accelerated.yml."
        ) from error

    library_directory = Path(tensorrt_libs.__file__).resolve().parent
    for filename in (
        "libnvinfer.so.10",
        "libnvinfer_plugin.so.10",
        "libnvonnxparser.so.10",
    ):
        path = library_directory / filename
        if not path.is_file():
            raise RuntimeError(f"Required TensorRT library not found: {path}")
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
