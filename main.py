"""Сравнение эрозии 3×3 через OpenCV и на чистом Python.

Запуск: python main.py
        python main.py --input photo.png --threshold 135 --output-dir results_custom
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from erosion import binarize, erosion_native, erosion_opencv
from generate_example import make_demo
from image_io import read_grayscale, write_image

ROOT = Path(__file__).resolve().parent


def make_preview(images: list[tuple[str, np.ndarray]]) -> np.ndarray:
    """Лист с подписями: исходник, бинаризация, две эрозии, разность."""
    tiles = []
    for title, image in images:
        tile = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        tile = cv2.copyMakeBorder(tile, 36, 10, 8, 8, cv2.BORDER_CONSTANT, value=(235, 235, 235))
        cv2.putText(tile, title, (11, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 2)
        tiles.append(tile)
    top = np.concatenate(tiles[:3], axis=1)
    empty = np.full_like(tiles[0], 235)
    bottom = np.concatenate([tiles[3], tiles[4], empty], axis=1)
    return np.concatenate([top, bottom], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="ЛР1, вариант 7 — эрозия, ядро 3×3")
    parser.add_argument("--input", type=Path, help="Путь к своему изображению (если не указан, используется демонстрационный пример)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results", help="Папка с результатами")
    parser.add_argument("--threshold", type=int, default=127, help="Порог бинаризации: 0–255 (по умолчанию 127)")
    parser.add_argument("--invert", action="store_true", help="Инвертировать бинаризацию: тёмные объекты сделать белыми")
    args = parser.parse_args()

    input_path = args.input or ROOT / "examples" / "input.png"
    if args.input is None and not input_path.exists():
        make_demo(input_path)
    try:
        gray = read_grayscale(input_path)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    try:
        binary = binarize(gray, threshold=args.threshold, invert=args.invert)
    except ValueError as exc:
        parser.error(str(exc))

    library_result = erosion_opencv(binary)
    native_result = erosion_native(binary)
    diff = cv2.absdiff(library_result, native_result)
    different_pixels = int(np.count_nonzero(diff))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_image(args.output_dir / "01_binary.png", binary)
    write_image(args.output_dir / "02_opencv.png", library_result)
    write_image(args.output_dir / "03_native.png", native_result)
    write_image(args.output_dir / "04_difference.png", diff)
    write_image(args.output_dir / "05_comparison.png", make_preview([
        ("Original", gray),
        ("Binary", binary),
        ("OpenCV erosion", library_result),
        ("Python erosion", native_result),
        ("Difference", diff),
    ]))

    print(f"Входное изображение: {input_path} ({gray.shape[1]}×{gray.shape[0]})")
    print(f"Результаты сохранены: {args.output_dir}")
    print(f"Различающихся пикселей: {different_pixels}")
    if different_pixels != 0:
        raise AssertionError("Методы дали разные результаты!")


if __name__ == "__main__":
    main()
