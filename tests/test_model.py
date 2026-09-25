from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from tidy.checkpoint import load_checkpoint
from tidy.model import TIDYNet


class HaarTransformTests(unittest.TestCase):
    def test_known_coefficients_and_round_trip(self) -> None:
        image = torch.arange(1, 17, dtype=torch.float32).reshape(1, 1, 4, 4)
        low, high = TIDYNet._haar_decompose(image)

        expected_low = torch.tensor([[[[7.0, 11.0], [23.0, 27.0]]]])
        expected_low_high = torch.full((1, 1, 2, 2), -4.0)
        expected_high_low = torch.full((1, 1, 2, 2), -1.0)
        self.assertTrue(torch.allclose(low, expected_low, atol=2e-6))
        self.assertTrue(torch.allclose(high[:, :, 0], expected_low_high, atol=2e-6))
        self.assertTrue(torch.allclose(high[:, :, 1], expected_high_low, atol=2e-6))
        self.assertTrue(
            torch.allclose(
                high[:, :, 2], torch.zeros_like(expected_low_high), atol=2e-6
            )
        )
        self.assertTrue(
            torch.allclose(TIDYNet._haar_reconstruct(low, high), image, atol=2e-6)
        )


class ModelTests(unittest.TestCase):
    @staticmethod
    def small_model() -> TIDYNet:
        return TIDYNet(
            image_channels=3,
            width=4,
            encoder_blocks=(1, 1),
            middle_blocks=1,
            decoder_blocks=(1, 1),
        )

    def test_arbitrary_input_size_is_preserved(self) -> None:
        model = self.small_model().eval()
        image = torch.rand(1, 3, 31, 45)
        with torch.inference_mode():
            output = model(image)
        self.assertEqual(output.shape, image.shape)

    def test_legacy_checkpoint_filters_are_ignored(self) -> None:
        source = self.small_model()
        legacy_state = dict(source.state_dict())
        legacy_state["dwt.h0_col"] = torch.ones(1, 1, 2, 1)
        legacy_state["idwt.g0_col"] = torch.ones(1, 1, 2, 1)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pth"
            torch.save({"params": legacy_state}, path)
            target = self.small_model()
            loaded = load_checkpoint(target, path)

        self.assertEqual(loaded, len(source.state_dict()))
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, target.state_dict()[key]), key)


if __name__ == "__main__":
    unittest.main()
