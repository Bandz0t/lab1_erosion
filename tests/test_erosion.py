"""Проверка ручной эрозии на точное совпадение с OpenCV."""

import unittest

import numpy as np

from erosion import binarize, erosion_native, erosion_opencv


class TestErosion(unittest.TestCase):
    def check_equal(self, data: np.ndarray) -> None:
        actual = erosion_native(data)
        expected = erosion_opencv(data)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.uint8)

    def test_all_white(self):
        self.check_equal(np.full((10, 10), 255, dtype=np.uint8))

    def test_all_black(self):
        self.check_equal(np.zeros((12, 7), dtype=np.uint8))

    def test_single_white_pixel(self):
        data = np.zeros((9, 9), dtype=np.uint8)
        data[4, 4] = 255
        self.check_equal(data)
        self.assertEqual(int(erosion_native(data).sum()), 0)

    def test_hole_in_square(self):
        data = np.full((9, 9), 255, dtype=np.uint8)
        data[4, 4] = 0
        self.check_equal(data)

    def test_images_smaller_than_kernel(self):
        for height in (1, 2, 3):
            for width in (1, 2, 3):
                with self.subTest(height=height, width=width):
                    self.check_equal(np.full((height, width), 255, dtype=np.uint8))

    def test_random_binary_images(self):
        generator = np.random.default_rng(42)
        for shape in [(5, 7), (32, 32), (51, 83), (128, 64)]:
            for p in (0.1, 0.5, 0.9):
                with self.subTest(shape=shape, density=p):
                    data = (generator.random(shape) < p).astype(np.uint8) * 255
                    self.check_equal(data)

    def test_threshold(self):
        data = np.array([[0, 100, 128, 200, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(binarize(data, 127), [[0, 0, 255, 255, 255]])
        np.testing.assert_array_equal(binarize(data, 127, invert=True), [[255, 255, 0, 0, 0]])


if __name__ == "__main__":
    unittest.main()
