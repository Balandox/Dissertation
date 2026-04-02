"""
Полный пайплайн видеомодуля: обработка всех видео → таблица признаков.

Для каждого видео:
  1. MediaPipe → landmarks → EAR/MAR по кадрам
  2. Детекция морганий (FSM)
  3. Детекция зеваний
  4. Агрегация в вектор 6 признаков по окнам
  5. Присвоение метки класса

Выход: CSV-таблица, где каждая строка — временной сегмент (окно),
6 признаков + метаинформация + метка класса.

Использование:
    python -m src.extract_features [--metadata PATH] [--output PATH] [--max_frames N]

Примеры:
    # Обработать все видео (долго):
    python -m src.extract_features

    # Быстрый тест — первые 3000 кадров (~100 сек) каждого видео:
    python -m src.extract_features --max_frames 3000

    # Конкретный CSV метаданных:
    python -m src.extract_features --metadata outputs/dataset_metadata.csv
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from src.landmarks import process_video, results_to_arrays, DEFAULT_MODEL_PATH
from src.blink_detector import BlinkDetector
from src.yawn_detector import YawnDetector
from src.feature_aggregator import FeatureAggregator


def load_metadata(metadata_path: Path) -> list[dict]:
    """Загрузить CSV метаданных датасета."""
    records = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def process_single_video(
    video_path: str,
    model_path: str,
    max_frames: int | None,
    ear_threshold: float,
    window_sec: float,
    yawn_window_sec: float,
    step_sec: float,
) -> tuple[list, list, list]:
    """
    Обработать одно видео: landmarks → blinks → yawns → features.

    Returns:
        (feature_vectors, blinks, yawns)
    """
    # Шаг 1: MediaPipe → покадровые EAR/MAR
    frame_results = process_video(
        video_path=video_path,
        model_path=model_path,
        max_frames=max_frames,
        progress_every=1000,
    )

    if not frame_results:
        return [], [], []

    data = results_to_arrays(frame_results)

    # Шаг 2: Детекция морганий
    blink_det = BlinkDetector(ear_threshold=ear_threshold)
    blinks = blink_det.detect(
        data["ear_avg"], data["timestamps_ms"], data["face_detected"]
    )

    # Шаг 3: Детекция зеваний
    yawn_det = YawnDetector()
    yawns = yawn_det.detect(
        data["mar"], data["timestamps_ms"], data["face_detected"]
    )

    # Шаг 4: Агрегация признаков
    aggregator = FeatureAggregator(
        window_sec=window_sec,
        yawn_window_sec=yawn_window_sec,
        step_sec=step_sec,
        ear_threshold=ear_threshold,
    )
    features = aggregator.aggregate(
        data["ear_avg"], data["timestamps_ms"], data["face_detected"],
        blinks, yawns,
    )

    return features, blinks, yawns


def save_features(
    all_rows: list[dict],
    output_path: Path,
):
    """Сохранить все признаки в CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "subject_id", "fold", "class_id", "class_name", "video_filename",
        "window_start_ms", "window_end_ms",
        "avg_ear", "blink_rate", "avg_blink_duration",
        "prolonged_blink_count", "perclos", "yawn_count",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nПризнаки сохранены: {output_path}")
    print(f"Всего строк (окон): {len(all_rows)}")


def main():
    parser = argparse.ArgumentParser(
        description="Извлечение признаков из всех видео датасета"
    )
    parser.add_argument(
        "--metadata", type=str, default="outputs/dataset_metadata.csv",
        help="CSV с метаданными видео"
    )
    parser.add_argument(
        "--model", type=str, default=str(DEFAULT_MODEL_PATH),
        help="Путь к модели face_landmarker.task"
    )
    parser.add_argument(
        "--output", type=str, default="outputs/video_features.csv",
        help="Выходной CSV с признаками"
    )
    parser.add_argument(
        "--max_frames", type=int, default=None,
        help="Макс. кадров на видео (None = все). Для теста: 3000"
    )
    parser.add_argument(
        "--ear_threshold", type=float, default=0.21,
        help="Порог EAR для закрытых глаз"
    )
    parser.add_argument(
        "--window_sec", type=float, default=60.0,
        help="Длина окна агрегации (сек)"
    )
    parser.add_argument(
        "--yawn_window_sec", type=float, default=300.0,
        help="Длина окна подсчёта зеваний (сек)"
    )
    parser.add_argument(
        "--step_sec", type=float, default=10.0,
        help="Шаг скольжения окна (сек)"
    )
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        print(f"Ошибка: метаданные не найдены: {metadata_path}")
        print("Сначала запустите: python -m src.dataset_parser")
        return

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Ошибка: модель не найдена: {model_path}")
        print("Запустите: python -m src.download_model")
        return

    # Загружаем метаданные
    records = load_metadata(metadata_path)
    print(f"Загружено {len(records)} видеозаписей из {metadata_path}")
    print(f"Параметры: окно={args.window_sec}с, шаг={args.step_sec}с, "
          f"EAR порог={args.ear_threshold}")
    if args.max_frames:
        print(f"Ограничение: первые {args.max_frames} кадров на видео")
    print()

    all_rows = []
    total_blinks = 0
    total_yawns = 0
    t_start_global = time.time()

    for idx, rec in enumerate(records):
        video_path = rec["video_path"]
        subject_id = rec["subject_id"]
        class_id = rec["class_id"]
        class_name = rec["class_name"]
        fold = rec["fold"]
        video_filename = rec["video_filename"]

        print(f"\n[{idx + 1}/{len(records)}] Участник {subject_id}, "
              f"класс: {class_name}, файл: {video_filename}")

        if not Path(video_path).exists():
            print(f"  ПРОПУЩЕНО: файл не найден")
            continue

        t_start = time.time()

        features, blinks, yawns = process_single_video(
            video_path=video_path,
            model_path=str(model_path),
            max_frames=args.max_frames,
            ear_threshold=args.ear_threshold,
            window_sec=args.window_sec,
            yawn_window_sec=args.yawn_window_sec,
            step_sec=args.step_sec,
        )

        t_elapsed = time.time() - t_start

        total_blinks += len(blinks)
        total_yawns += len(yawns)

        print(f"  Результат: {len(features)} окон, "
              f"{len(blinks)} морганий, {len(yawns)} зеваний "
              f"({t_elapsed:.1f} сек)")

        # Формируем строки для CSV
        for fv in features:
            all_rows.append({
                "subject_id": subject_id,
                "fold": fold,
                "class_id": class_id,
                "class_name": class_name,
                "video_filename": video_filename,
                "window_start_ms": fv.window_start_ms,
                "window_end_ms": fv.window_end_ms,
                "avg_ear": fv.avg_ear,
                "blink_rate": fv.blink_rate,
                "avg_blink_duration": fv.avg_blink_duration,
                "prolonged_blink_count": fv.prolonged_blink_count,
                "perclos": fv.perclos,
                "yawn_count": fv.yawn_count,
            })

    t_total = time.time() - t_start_global

    # Сохраняем
    output_path = Path(args.output)
    save_features(all_rows, output_path)

    # Итоговая статистика
    print(f"\n{'=' * 60}")
    print("ИТОГО")
    print(f"{'=' * 60}")
    print(f"Видео обработано: {len(records)}")
    print(f"Окон (строк):     {len(all_rows)}")
    print(f"Морганий всего:   {total_blinks}")
    print(f"Зеваний всего:    {total_yawns}")
    print(f"Время:            {t_total:.0f} сек ({t_total/60:.1f} мин)")

    # Статистика по классам
    if all_rows:
        print(f"\nРаспределение окон по классам:")
        for cid in ["0", "1", "2"]:
            cname = {"0": "normal", "1": "pre_fatigue", "2": "drowsy"}[cid]
            count = sum(1 for r in all_rows if str(r["class_id"]) == cid)
            print(f"  {cname:15s}: {count} окон")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
