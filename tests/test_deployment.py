from __future__ import annotations

import importlib.util
import tempfile
import unittest
import warnings
from pathlib import Path

import torch

from tidy.backends import BACKEND_NAMES, _torch_dtype, normalize_backend
from tidy.cli import build_parser
from tidy.model import ChannelLayerNorm2d, TIDYNet


class BackendInterfaceTests(unittest.TestCase):
    def test_backend_names_and_both_alias(self) -> None:
        self.assertIn("pytorch", BACKEND_NAMES)
        self.assertIn("onnx", BACKEND_NAMES)
        self.assertIn("tensorrt", BACKEND_NAMES)
        self.assertEqual(normalize_backend("both"), "onnx-tensorrt")
        self.assertEqual(normalize_backend("ONNX"), "onnx")
        with self.assertRaises(ValueError):
            normalize_backend("unknown")

    def test_original_cli_defaults_to_pytorch(self) -> None:
        args = build_parser().parse_args(["-i", "input.png", "-o", "output.png"])
        self.assertEqual(args.backend, "pytorch")
        self.assertFalse(args.fp16)

    def test_runtime_dtype_conversion(self) -> None:
        import numpy as np

        self.assertEqual(_torch_dtype(np.float32), torch.float32)
        self.assertEqual(_torch_dtype(np.float16), torch.float16)


class ExportabilityTests(unittest.TestCase):
    def test_standard_layer_norm_matches_explicit_formula(self) -> None:
        layer = ChannelLayerNorm2d(4).eval()
        layer.weight.data.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))
        layer.bias.data.copy_(torch.tensor([-0.2, -0.1, 0.1, 0.2]))
        image = torch.randn(2, 4, 7, 9)
        mean = image.mean(dim=1, keepdim=True)
        variance = (image - mean).square().mean(dim=1, keepdim=True)
        expected = (image - mean) * torch.rsqrt(variance + layer.eps)
        expected = expected * layer.weight.view(1, -1, 1, 1)
        expected = expected + layer.bias.view(1, -1, 1, 1)
        self.assertTrue(torch.allclose(layer(image), expected, atol=2e-6, rtol=2e-5))

    @unittest.skipUnless(
        importlib.util.find_spec("onnx") is not None,
        "optional ONNX package is not installed",
    )
    def test_small_model_exports_standard_layer_normalization(self) -> None:
        import onnx

        model = TIDYNet(
            width=4,
            encoder_blocks=(1, 1),
            middle_blocks=1,
            decoder_blocks=(1, 1),
        ).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.onnx"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                torch.onnx.export(
                    model,
                    torch.rand(1, 3, 31, 45),
                    path,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={
                        "input": {2: "height", 3: "width"},
                        "output": {2: "height", 3: "width"},
                    },
                    opset_version=18,
                    dynamo=False,
                )
            graph = onnx.load(path).graph
        self.assertIn("LayerNormalization", {node.op_type for node in graph.node})


if __name__ == "__main__":
    unittest.main()
