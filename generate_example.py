"""Создание тестового полутонового изображения с фигурами и шумом."""

from pathlib import Path

import cv2
import numpy as np

from image_io import write_image


ROOT = Path(__file__).resolve().parent


def make_demo(path: Path = ROOT / "examples" / "input.png") -> Path:
    width, height = 420, 280
    image = np.full((height, width), 25, dtype=np.uint8)
    cv2.rectangle(image, (20, 30), (150, 170), 225, -1)
    cv2.rectangle(image, (65, 75), (105, 120), 20, -1)  # внутреннее отверстие
    cv2.circle(image, (260, 105), 63, 225, -1)
    cv2.rectangle(image, (220, 90), (310, 102), 25, -1)  # прорезь
    cv2.line(image, (25, 226), (185, 226), 230, 1)  # тонкая линия
    cv2.line(image, (220, 222), (390, 222), 240, 9)
    cv2.rectangle(image, (0, 0), (50, 12), 220, -1)  # объект на границе
    cv2.putText(image, "CV", (335, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 220, 2)

    rng = np.random.default_rng(2026)
    noise_y = rng.integers(0, height, size=500)
    noise_x = rng.integers(0, width, size=500)
    image[noise_y, noise_x] = 230  # одиночные светлые точки

    write_image(path, image)
    return path


if __name__ == "__main__":
    print(make_demo())
