"""Время работы двух реализаций на одинаковых бинарных изображениях."""

import argparse
import csv
import platform
import statistics
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from erosion import erosion_native, erosion_opencv

ROOT = Path(__file__).resolve().parent


def make_binary(size: int) -> np.ndarray:
    """Детерминированные белые фигуры на чёрном фоне для любого размера."""
    image = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(image, (size // 12, size // 10), (size * 6 // 10, size * 8 // 10), 255, -1)
    cv2.circle(image, (size * 3 // 4, size // 2), max(1, size // 6), 255, -1)
    cv2.rectangle(image, (size // 5, size // 4), (size // 3, size // 2), 0, -1)
    rng = np.random.default_rng(777 + size)
    ys = rng.integers(0, size, size=max(10, size // 2))
    xs = rng.integers(0, size, size=max(10, size // 2))
    image[ys, xs] = 255
    return image


def median_runtime_ms(func, image: np.ndarray, repeats: int) -> float:
    func(image)  # прогрев
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        func(image)
        timings.append((time.perf_counter() - start) * 1000)
    return statistics.median(timings)


def draw_graph(records: list[dict], output: Path) -> None:
    sizes = [f'{r["size"]}×{r["size"]}' for r in records]
    native = [r["native_ms"] for r in records]
    library = [r["opencv_ms"] for r in records]
    xs = np.arange(len(records))
    width = 0.35

    plt.figure(figsize=(10, 5.5))
    plt.bar(xs - width / 2, library, width, label="OpenCV", color="#2474b5")
    plt.bar(xs + width / 2, native, width, label="Python (вручную)", color="#d98933")
    plt.xticks(xs, sizes)
    plt.yscale("log")
    plt.xlabel("Размер изображения (пиксели)")
    plt.ylabel("Время одной обработки, мс (логарифмическая шкала)")
    plt.title("Эрозия 3×3: сравнение времени выполнения")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def run(sizes: list[int], repeats: int, output_dir: Path) -> list[dict]:
    if any(x < 1 for x in sizes) or repeats < 1:
        raise ValueError("Размеры изображений и число повторов должны быть положительными")

    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    print(f"Python: {platform.python_version()} | OpenCV: {cv2.__version__} | NumPy: {np.__version__}")
    print(f"Повторов для каждого замера: {repeats} (медиана), без времени загрузки и бинаризации")
    print(f'{"Размер":>9} | {"OpenCV, мс":>12} | {"Python, мс":>12} | {"Ускорение":>12} | {"Различий":>8}')
    for size in sizes:
        image = make_binary(size)
        opencv_result = erosion_opencv(image)
        native_result = erosion_native(image)
        difference_count = int(np.count_nonzero(opencv_result != native_result))
        if difference_count:
            raise AssertionError(f"Ошибка совпадения результатов: {difference_count} пикселей для {size}")

        cv_ms = median_runtime_ms(erosion_opencv, image, repeats)
        py_ms = median_runtime_ms(erosion_native, image, repeats)
        speedup = py_ms / cv_ms
        record = {"size": size, "pixels": size * size, "opencv_ms": cv_ms,
                  "native_ms": py_ms, "speedup": speedup, "different_pixels": difference_count}
        records.append(record)
        print(f"{size:>4}×{size:<4} | {cv_ms:>12.4f} | {py_ms:>12.4f} | {speedup:>11.1f}× | {difference_count:>8}")

    csv_path = output_dir / "benchmark.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    draw_graph(records, output_dir / "benchmark.png")
    print(f"\nТаблица: {csv_path}\nГрафик: {output_dir / 'benchmark.png'}")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Замер времени эрозии OpenCV и Python")
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256, 384], help="Размеры квадратных бинарных изображений")
    parser.add_argument("--repeats", type=int, default=7, help="Количество замеров каждого метода")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    run(args.sizes, args.repeats, args.output_dir)
