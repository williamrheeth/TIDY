from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tidy.images import find_images, read_image, write_image


class ImageTests(unittest.TestCase):
    def test_grayscale_input_becomes_rgb_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "thermal.png"
            output_path = root / "result.png"
            values = np.arange(20, dtype=np.uint8).reshape(4, 5)
            Image.fromarray(values).save(input_path)

            tensor = read_image(input_path)
            self.assertEqual(tuple(tensor.shape), (1, 3, 4, 5))
            self.assertTrue((tensor[:, 0] == tensor[:, 1]).all())
            write_image(tensor, output_path)

            with Image.open(output_path) as result:
                self.assertEqual(result.mode, "RGB")
                self.assertTrue(np.array_equal(np.asarray(result)[:, :, 0], values))

    def test_image_discovery_is_case_insensitive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("L", (2, 2)).save(root / "B.PNG")
            Image.new("L", (2, 2)).save(root / "a.png")
            (root / "ignore.txt").write_text("not an image", encoding="utf-8")
            self.assertEqual(
                [path.name for path in find_images(root)], ["a.png", "B.PNG"]
            )


if __name__ == "__main__":
    unittest.main()
