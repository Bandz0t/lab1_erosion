"""Бинарная эрозия прямоугольным структурным элементом 3×3."""

import cv2
import numpy as np


KERNEL_3X3 = np.ones((3, 3), dtype=np.uint8)


def binarize(image: np.ndarray, threshold: int = 127, invert: bool = False) -> np.ndarray:
    """Подготовить ОДНО И ТО ЖЕ бинарное изображение для обоих методов.

    Бинаризация библиотечная: по условию задания писать её вручную не нужно.
    """
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("Ожидается одноканальное изображение uint8")
    if not 0 <= threshold <= 255:
        raise ValueError("Порог должен находиться в интервале 0..255")
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(image, threshold, 255, mode)
    return binary


def erosion_opencv(binary: np.ndarray) -> np.ndarray:
    """Библиотечная версия. За пределами изображения пиксели равны нулю."""
    return cv2.erode(
        binary,
        KERNEL_3X3,
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def erosion_native(binary: np.ndarray) -> np.ndarray:
    """Ручная версия на Python: каждый пиксель и все 9 соседей проверяются циклом.

    Внутри алгоритма нет cv2.erode, свёрток, готовых min-фильтров.
    На внешней рамке результат нулевой (так как за изображением находятся нули).
    """
    if binary.ndim != 2 or binary.dtype != np.uint8:
        raise ValueError("Ожидается одноканальное бинарное изображение uint8")

    height, width = binary.shape
    source = binary.tolist()
    result = [[0] * width for _ in range(height)]

    # При 1×N, 2×N и т. п. внутренней части нет: вся эрозия равна 0.
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            all_white = True
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if source[y + dy][x + dx] == 0:
                        all_white = False
                        break
                if not all_white:
                    break
            if all_white:
                result[y][x] = 255

    return np.asarray(result, dtype=np.uint8)
