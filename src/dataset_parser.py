"""
Парсер датасета UTA-RLDD.

Поддерживает реальную структуру Kaggle:
    data/uta_rldd/videos/
    ├── Fold1_part1/
    │   ├── 01/
    │   │   ├── 0.mov    (alert)
    │   │   ├── 5.mov    (low vigilant)
    │   │   └── 10.MOV   (drowsy)
    │   ├── 02/ ...
    │   └── 06/
    ├── Fold1_part2/
    │   ├── 07/ ...
    │   └── 12/
    ...

Использование:
    python -m src.dataset_parser [--dataset_root PATH] [--output PATH]
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ================================================================
# Константы
# ================================================================

VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".m4v",
    ".MP4", ".AVI", ".MOV", ".MKV", ".WMV", ".WEBM", ".M4V",
}

VALID_LABELS = {0, 5, 10}

LABEL_TO_CLASS = {0: 0, 5: 1, 10: 2}

CLASS_NAMES = {0: "normal", 1: "pre_fatigue", 2: "drowsy"}

FOLDS = {
    1: list(range(1, 13)),
    2: list(range(13, 25)),
    3: list(range(25, 37)),
    4: list(range(37, 49)),
    5: list(range(49, 61)),
}

FOLD_DIR_MAP = {
    "Fold1_part1": 1, "Fold1_part2": 1,
    "Fold2_part1": 2, "Fold2_part2": 2,
    "Fold3_part1": 3, "Fold3_part2": 3,
    "Fold4_part1": 4, "Fold4_part2": 4,
    "Fold5_part1": 5, "Fold5_part2": 5,
}


def get_fold(subject_id: int) -> int:
    """Определить номер фолда по ID участника."""
    for fold_num, subjects in FOLDS.items():
        if subject_id in subjects:
            return fold_num
    return -1


def get_video_info(video_path: Path) -> dict:
    """Получить метаинформацию о видеофайле через OpenCV."""
    empty = {"fps": None, "width": None, "height": None,
             "total_frames": None, "duration_sec": None}

    if not HAS_CV2:
        return empty

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return empty

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0
    cap.release()

    return {
        "fps": round(fps, 2),
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "duration_sec": round(duration_sec, 1),
    }


def parse_label_from_filename(stem: str) -> Optional[int]:
    """
    Извлечь метку из имени файла.
    Примеры: "0" → 0, "5" → 5, "10" → 10.
    """
    try:
        val = int(stem)
        if val in VALID_LABELS:
            return val
    except ValueError:
        pass

    stem_lower = stem.lower().strip()
    if "alert" in stem_lower and "low" not in stem_lower:
        return 0
    if "low" in stem_lower or "vigilant" in stem_lower:
        return 5
    if "drowsy" in stem_lower or "sleep" in stem_lower:
        return 10

    return None


def scan_dataset(dataset_root: Path, skip_video_info: bool = False) -> list[dict]:
    """
    Сканирует датасет с учётом структуры Kaggle:
        dataset_root/FoldX_partY/SUBJECT_ID/LABEL.ext

    Также поддерживает плоскую структуру:
        dataset_root/SUBJECT_ID/LABEL.ext
    """
    records = []

    if not dataset_root.exists():
        print(f"ОШИБКА: Директория не найдена: {dataset_root}")
        return records

    # Ищем все папки участников (числовые имена: "01", "1", "42" и т.д.)
    # Они могут быть на разных уровнях вложенности
    subject_dirs = []

    for item in sorted(dataset_root.rglob("*")):
        if not item.is_dir():
            continue
        # Папка с числовым именем, содержащая видеофайлы
        if not item.name.lstrip("0").isdigit() and item.name != "0":
            continue
        # Проверяем, что в ней есть видеофайлы
        videos = [f for f in item.iterdir()
                  if f.is_file() and f.suffix in VIDEO_EXTENSIONS]
        if videos:
            subject_dirs.append(item)

    if not subject_dirs:
        print("Не найдено папок участников с видеофайлами.")
        return records

    print(f"Найдено {len(subject_dirs)} папок участников")

    for subj_dir in subject_dirs:
        # Извлекаем subject_id (убираем ведущие нули: "01" → 1)
        subject_id = int(subj_dir.name)

        # Определяем фолд: по имени родительской папки или по номеру
        fold = _detect_fold(subj_dir, subject_id)

        # Определяем имя части (Fold1_part1 и т.д.)
        fold_part = _detect_fold_part(subj_dir)

        # Ищем видеофайлы
        video_files = [f for f in subj_dir.iterdir()
                       if f.is_file() and f.suffix in VIDEO_EXTENSIONS]

        for vf in sorted(video_files, key=lambda f: f.name):
            label = parse_label_from_filename(vf.stem)

            if label is None:
                print(f"  ПРЕДУПРЕЖДЕНИЕ: не удалось определить метку "
                      f"для {vf.name} (участник {subject_id})")
                continue

            class_id = LABEL_TO_CLASS.get(label)
            if class_id is None:
                continue

            info = (get_video_info(vf) if not skip_video_info
                    else {"fps": None, "width": None, "height": None,
                          "total_frames": None, "duration_sec": None})

            records.append({
                "subject_id": subject_id,
                "fold": fold,
                "fold_part": fold_part,
                "original_label": label,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "video_path": str(vf),
                "video_filename": vf.name,
                "fps": info["fps"],
                "width": info["width"],
                "height": info["height"],
                "total_frames": info["total_frames"],
                "duration_sec": info["duration_sec"],
            })

    records.sort(key=lambda r: (r["subject_id"], r["class_id"]))
    return records


def _detect_fold(subj_dir: Path, subject_id: int) -> int:
    """Определить фолд — по имени папки-родителя или по номеру участника."""
    for parent in subj_dir.parents:
        if parent.name in FOLD_DIR_MAP:
            return FOLD_DIR_MAP[parent.name]
    return get_fold(subject_id)


def _detect_fold_part(subj_dir: Path) -> str:
    """Определить имя части фолда (Fold1_part1 и т.д.)."""
    for parent in subj_dir.parents:
        if parent.name in FOLD_DIR_MAP:
            return parent.name
    return ""


def save_metadata(records: list[dict], output_path: Path):
    """Сохранить метаданные в CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "subject_id", "fold", "fold_part", "original_label",
        "class_id", "class_name", "video_path", "video_filename",
        "fps", "width", "height", "total_frames", "duration_sec",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nМетаданные сохранены: {output_path}")


def print_statistics(records: list[dict]):
    """Вывести сводную статистику."""
    if not records:
        print("\nНет данных для анализа.")
        return

    print("\n" + "=" * 60)
    print("СТАТИСТИКА ДАТАСЕТА")
    print("=" * 60)

    subjects = sorted(set(r["subject_id"] for r in records))
    print(f"\nВсего видеозаписей: {len(records)}")
    print(f"Участников: {len(subjects)}")
    print(f"ID участников: {subjects}")

    # По классам
    print("\nРаспределение по классам:")
    for cid in sorted(CLASS_NAMES.keys()):
        count = sum(1 for r in records if r["class_id"] == cid)
        print(f"  {CLASS_NAMES[cid]:15s}: {count} видео")

    # По фолдам/частям
    fold_parts = sorted(set(r["fold_part"] for r in records if r["fold_part"]))
    if fold_parts:
        print(f"\nЗагруженные части: {', '.join(fold_parts)}")
        for fp in fold_parts:
            fp_records = [r for r in records if r["fold_part"] == fp]
            fp_subjects = sorted(set(r["subject_id"] for r in fp_records))
            print(f"  {fp}: участники {fp_subjects} ({len(fp_records)} видео)")

    # Информация о видео
    durations = [r["duration_sec"] for r in records if r["duration_sec"]]
    if durations:
        total_min = sum(durations) / 60
        print(f"\nОбщая длительность: {total_min:.1f} мин")
        print(f"Средняя длительность видео: {sum(durations)/len(durations):.0f} сек")
        print(f"Диапазон: {min(durations):.0f}–{max(durations):.0f} сек")

    resolutions = {}
    fps_values = set()
    for r in records:
        if r["width"] and r["height"]:
            res = f"{r['width']}x{r['height']}"
            resolutions[res] = resolutions.get(res, 0) + 1
        if r["fps"]:
            fps_values.add(r["fps"])

    if resolutions:
        print(f"\nРазрешения:")
        for res, count in sorted(resolutions.items()):
            print(f"  {res}: {count} видео")

    if fps_values:
        print(f"FPS: {sorted(fps_values)}")

    # Проверка полноты загруженных участников
    print("\nЦелостность:")
    incomplete = []
    for sid in subjects:
        classes = set(r["class_id"] for r in records if r["subject_id"] == sid)
        if classes != {0, 1, 2}:
            missing = [CLASS_NAMES[c] for c in ({0, 1, 2} - classes)]
            incomplete.append((sid, missing))

    if incomplete:
        print(f"  ВНИМАНИЕ: у {len(incomplete)} участников неполные данные:")
        for sid, missing in incomplete:
            print(f"    Участник {sid}: нет {missing}")
    else:
        print(f"  ✓ Все {len(subjects)} участников имеют видео для всех 3 классов")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Парсер метаданных датасета UTA-RLDD"
    )
    parser.add_argument(
        "--dataset_root", type=str, default="data/uta_rldd/videos",
        help="Путь к корневой директории датасета"
    )
    parser.add_argument(
        "--output", type=str, default="outputs/dataset_metadata.csv",
        help="Путь для сохранения CSV"
    )
    parser.add_argument(
        "--no-video-info", action="store_true",
        help="Не извлекать информацию о видео (быстрее, без OpenCV)"
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_path = Path(args.output)

    print(f"Сканирование: {dataset_root.resolve()}")

    if not dataset_root.exists():
        print(f"\nОШИБКА: Директория не найдена: {dataset_root.resolve()}")
        print("\nПоместите видеофайлы датасета UTA-RLDD в:")
        print(f"  {dataset_root}/FoldX_partY/SUBJECT_ID/")
        sys.exit(1)

    records = scan_dataset(dataset_root, skip_video_info=args.no_video_info)

    if not records:
        print("\nНе найдено видеофайлов с корректными метками.")
        sys.exit(1)

    save_metadata(records, output_path)
    print_statistics(records)

    return records


if __name__ == "__main__":
    main()