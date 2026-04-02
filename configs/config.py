"""
Конфигурация датасета UTA-RLDD и общие настройки проекта.
"""

from pathlib import Path

# ============================================================
# ПУТИ
# ============================================================

# Корневая директория с видеофайлами датасета
# Реальная структура (Kaggle):
#   data/uta_rldd/videos/
#   ├── Fold1_part1/
#   │   ├── 01/
#   │   │   ├── 0.mov   (alert)
#   │   │   ├── 5.mov   (low vigilant)
#   │   │   └── 10.MOV  (drowsy)
#   │   ├── 02/ ...
#   │   └── 06/
#   ├── Fold1_part2/
#   │   ├── 07/ ...
#   │   └── 12/
#   ├── Fold2_part1/
#   │   ├── 13/ ...
#   ...

DATASET_ROOT = Path("data/uta_rldd/videos")

# Директория для результатов обработки
OUTPUT_DIR = Path("outputs")

# ============================================================
# МАППИНГ КЛАССОВ
# ============================================================

LABEL_MAP = {
    0:  0,   # alert       → normal (бодрый)
    5:  1,   # low vigilant → pre-fatigue (предусталость)
    10: 2,   # drowsy      → drowsy (сонливый)
}

LABEL_NAMES = {
    0: "normal",
    1: "pre_fatigue",
    2: "drowsy",
}

LABEL_NAMES_RU = {
    0: "Бодрый",
    1: "Предусталость",
    2: "Сонливый",
}

# ============================================================
# ФОЛДЫ (5-fold cross-validation)
# Структура Kaggle: FoldX_partY/
#   Fold1: участники 01–12  (part1: 01–06, part2: 07–12)
#   Fold2: участники 13–24  (part1: 13–18, part2: 19–24)
#   Fold3: участники 25–36  (part1: 25–30, part2: 31–36)
#   Fold4: участники 37–48  (part1: 37–42, part2: 43–48)
#   Fold5: участники 49–60  (part1: 49–54, part2: 55–60)
# ============================================================

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

# ============================================================
# ПАРАМЕТРЫ ДЕТЕКЦИИ ПРИЗНАКОВ
# ============================================================

TARGET_FPS = 30
MIN_RESOLUTION = (320, 240)

EAR_THRESHOLD = 0.21
EAR_CONSEC_FRAMES_BLINK = 2
PROLONGED_BLINK_MS = 500

MAR_THRESHOLD = 0.6
YAWN_MIN_DURATION_SEC = 1.5

WINDOW_PERCLOS_SEC = 60
WINDOW_BLINK_RATE_SEC = 60
WINDOW_YAWN_SEC = 300
FEATURE_STEP_SEC = 10

# ============================================================
# ВИДЕОФОРМАТЫ
# ============================================================

VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".m4v",
    ".MP4", ".AVI", ".MOV", ".MKV", ".WMV", ".WEBM", ".M4V",
}