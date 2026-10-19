"""Чтение и запись изображений с поддержкой путей Unicode (Windows)."""

from pathlib import Path

import cv2
import numpy as np


def read_grayscale(path: Path) -> np.ndarray:
    """Прочитать изображение независимо от кириллицы в пути."""
    path = Path(path)
    try:
        contents = path.read_bytes()
    except OSError as exc:
        raise OSError(f"Не удалось прочитать файл: {path}. Проверьте наличие и права доступа.") from exc
    if not contents:
        raise ValueError(f"Файл пуст: {path}")
    image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Формат изображения не поддерживается или файл повреждён: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    """Сохранить изображение независимо от кириллицы в пути."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix.lower()
    if not extension:
        raise ValueError(f"Укажите расширение выходного файла (.png, .jpg и т. п.): {path}")
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise OSError(f"Не удалось закодировать изображение в формат {extension}: {path}")
    try:
        path.write_bytes(encoded.tobytes())
    except OSError as exc:
        raise OSError(f"Не удалось сохранить изображение: {path}") from exc
