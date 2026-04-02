"""
Тест шага 2: обработка одного видео через MediaPipe → EAR и MAR по кадрам.

Использование:
    python -m src.test_landmarks [--video PATH] [--model PATH] [--max_frames N]

Примеры:
    # Обработать первые 300 кадров (10 сек) для быстрой проверки:
    python -m src.test_landmarks --video data/uta_rldd/videos/Fold1_part1/01/0.mov --max_frames 300

    # Обработать всё видео целиком:
    python -m src.test_landmarks --video data/uta_rldd/videos/Fold1_part1/01/0.mov
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from src.landmarks import process_video, results_to_arrays, DEFAULT_MODEL_PATH


def print_stats(data: dict, fps: float):
    """Вывести базовую статистику по EAR и MAR."""
    face_mask = data["face_detected"]
    face_pct = face_mask.sum() / len(face_mask) * 100

    print(f"\n{'=' * 50}")
    print("РЕЗУЛЬТАТЫ ОБРАБОТКИ")
    print(f"{'=' * 50}")
    print(f"Всего кадров:       {len(face_mask)}")
    print(f"Лицо найдено:      {face_mask.sum()} ({face_pct:.1f}%)")
    print(f"Лицо не найдено:   {(~face_mask).sum()}")
    print(f"Длительность:       {len(face_mask) / fps:.1f} сек")

    if face_mask.sum() == 0:
        print("\nНет данных для анализа (лицо не обнаружено).")
        return

    # Статистика EAR (только кадры с лицом)
    ear = data["ear_avg"][face_mask]
    print(f"\nEAR (Eye Aspect Ratio):")
    print(f"  Среднее:  {ear.mean():.4f}")
    print(f"  Медиана:  {np.median(ear):.4f}")
    print(f"  Мин:      {ear.min():.4f}")
    print(f"  Макс:     {ear.max():.4f}")
    print(f"  Std:      {ear.std():.4f}")

    # Статистика MAR
    mar = data["mar"][face_mask]
    print(f"\nMAR (Mouth Aspect Ratio):")
    print(f"  Среднее:  {mar.mean():.4f}")
    print(f"  Медиана:  {np.median(mar):.4f}")
    print(f"  Мин:      {mar.min():.4f}")
    print(f"  Макс:     {mar.max():.4f}")
    print(f"  Std:      {mar.std():.4f}")

    # Базовые оценки
    ear_closed = (ear < 0.21).sum()
    ear_closed_pct = ear_closed / len(ear) * 100
    mar_yawn = (mar > 0.6).sum()
    mar_yawn_pct = mar_yawn / len(mar) * 100

    print(f"\nОценки (предварительные):")
    print(f"  Кадров с закрытыми глазами (EAR < 0.21): "
          f"{ear_closed} ({ear_closed_pct:.1f}%)")
    print(f"  Кадров с открытым ртом (MAR > 0.6):      "
          f"{mar_yawn} ({mar_yawn_pct:.1f}%)")
    print(f"{'=' * 50}")


def save_timeseries(data: dict, output_path: Path):
    """Сохранить временные ряды EAR и MAR в CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_idx", "timestamp_ms", "face_detected",
            "ear_avg", "ear_left", "ear_right", "mar"
        ])
        for i in range(len(data["timestamps_ms"])):
            writer.writerow([
                i,
                data["timestamps_ms"][i],
                int(data["face_detected"][i]),
                round(data["ear_avg"][i], 6),
                round(data["ear_left"][i], 6),
                round(data["ear_right"][i], 6),
                round(data["mar"][i], 6),
            ])

    print(f"\nВременные ряды сохранены: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Тест видеомодуля: MediaPipe → EAR/MAR"
    )
    parser.add_argument(
        "--video", type=str, required=True,
        help="Путь к видеофайлу"
    )
    parser.add_argument(
        "--model", type=str, default=str(DEFAULT_MODEL_PATH),
        help="Путь к модели face_landmarker.task"
    )
    parser.add_argument(
        "--max_frames", type=int, default=None,
        help="Макс. количество кадров для обработки"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Путь для CSV с результатами (по умолчанию: outputs/test_ear_mar.csv)"
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Ошибка: видео не найдено: {video_path}")
        return

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Ошибка: модель не найдена: {model_path}")
        print(f"\nСкачайте модель командой:")
        print(f"  wget -O {model_path} "
              f"https://storage.googleapis.com/mediapipe-models/"
              f"face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        return

    # Получаем FPS для статистики
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Обрабатываем видео
    results = process_video(
        video_path=video_path,
        model_path=model_path,
        max_frames=args.max_frames,
        progress_every=300,
    )

    # Конвертируем в массивы
    data = results_to_arrays(results)

    # Выводим статистику
    print_stats(data, fps)

    # Сохраняем CSV
    output_path = Path(args.output) if args.output else Path("outputs/test_ear_mar.csv")
    save_timeseries(data, output_path)


if __name__ == "__main__":
    main()
