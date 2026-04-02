"""
Модуль извлечения лицевых ориентиров (landmarks) через MediaPipe FaceLandmarker.

Включает:
- Инициализацию модели FaceLandmarker (tasks API, mediapipe >= 0.10)
- Покадровую обработку видео → 478 3D-точек лица
- Вычисление EAR (Eye Aspect Ratio) и MAR (Mouth Aspect Ratio)

Требования:
- pip install mediapipe opencv-python numpy
- Модель: models/face_landmarker.task
  Скачать: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

import mediapipe as mp
from mediapipe.tasks.python import vision

# ================================================================
# Индексы ключевых точек MediaPipe Face Mesh (478 landmarks)
# Документация: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
# ================================================================

# Правый глаз (с точки зрения наблюдателя, т.е. левый глаз человека)
# 6 точек для расчёта EAR: p1(внешний угол), p2(верх-внешний),
# p3(верх-внутренний), p4(внутренний угол), p5(низ-внутренний), p6(низ-внешний)
RIGHT_EYE = {
    "p1": 33,   # внешний угол
    "p2": 160,  # верхнее веко (внешняя часть)
    "p3": 158,  # верхнее веко (внутренняя часть)
    "p4": 133,  # внутренний угол
    "p5": 153,  # нижнее веко (внутренняя часть)
    "p6": 144,  # нижнее веко (внешняя часть)
}

# Левый глаз (с точки зрения наблюдателя, т.е. правый глаз человека)
LEFT_EYE = {
    "p1": 362,  # внешний угол
    "p2": 385,  # верхнее веко (внешняя часть)
    "p3": 387,  # верхнее веко (внутренняя часть)
    "p4": 263,  # внутренний угол
    "p5": 373,  # нижнее веко (внутренняя часть)
    "p6": 380,  # нижнее веко (внешняя часть)
}

# Точки рта для расчёта MAR
# Верхняя и нижняя губы (внутренний контур) + углы рта
MOUTH = {
    "left_corner":  61,   # левый угол рта
    "right_corner": 291,  # правый угол рта
    "upper_1": 81,        # верхняя губа (левая часть)
    "upper_2": 13,        # верхняя губа (центр)
    "upper_3": 311,       # верхняя губа (правая часть)
    "lower_1": 178,       # нижняя губа (левая часть)
    "lower_2": 14,        # нижняя губа (центр)
    "lower_3": 402,       # нижняя губа (правая часть)
}


# ================================================================
# Структуры данных
# ================================================================

@dataclass
class FrameResult:
    """Результат обработки одного кадра."""
    frame_idx: int
    timestamp_ms: int
    face_detected: bool
    ear_left: float = 0.0
    ear_right: float = 0.0
    ear_avg: float = 0.0
    mar: float = 0.0
    landmarks: np.ndarray = field(default_factory=lambda: np.empty(0))
    # landmarks shape: (478, 3) — x, y, z в нормализованных координатах


# ================================================================
# Создание FaceLandmarker
# ================================================================

DEFAULT_MODEL_PATH = Path("models/face_landmarker.task")


def create_landmarker(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    mode: str = "video",
    num_faces: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> vision.FaceLandmarker:
    """
    Создать и вернуть объект FaceLandmarker.

    Args:
        model_path: путь к файлу face_landmarker.task
        mode: "image" или "video"
        num_faces: максимальное количество лиц
        min_detection_confidence: порог уверенности детекции
        min_tracking_confidence: порог уверенности трекинга

    Returns:
        Инициализированный FaceLandmarker
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Модель не найдена: {model_path.resolve()}\n"
            f"Скачайте модель командой:\n"
            f"  wget -O {model_path} "
            f"https://storage.googleapis.com/mediapipe-models/"
            f"face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )

    running_mode = (vision.RunningMode.VIDEO if mode == "video"
                    else vision.RunningMode.IMAGE)

    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path)
        ),
        running_mode=running_mode,
        num_faces=num_faces,
        min_face_detection_confidence=min_detection_confidence,
        min_face_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    return vision.FaceLandmarker.create_from_options(options)


# ================================================================
# Обработка кадра
# ================================================================

def process_frame(
    landmarker: vision.FaceLandmarker,
    frame_bgr: np.ndarray,
    frame_idx: int,
    timestamp_ms: int,
    mode: str = "video",
) -> FrameResult:
    """
    Обработать один кадр: детекция лица → landmarks → EAR, MAR.

    Args:
        landmarker: инициализированный FaceLandmarker
        frame_bgr: кадр в формате BGR (OpenCV)
        frame_idx: порядковый номер кадра
        timestamp_ms: временная метка кадра в мс
        mode: "video" или "image"

    Returns:
        FrameResult с EAR, MAR и координатами landmarks
    """
    # BGR → RGB
    frame_rgb = frame_bgr[:, :, ::-1].copy()

    # Создаём MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb,
    )

    # Детекция
    if mode == "video":
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
    else:
        result = landmarker.detect(mp_image)

    # Проверяем, найдено ли лицо
    if not result.face_landmarks:
        return FrameResult(
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
            face_detected=False,
        )

    # Берём первое лицо
    face = result.face_landmarks[0]

    # Конвертируем в numpy array (478, 3) — нормализованные координаты
    landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face])

    # Масштабируем x и y в пиксели, чтобы EAR/MAR не зависели
    # от ориентации видео (портрет/ландшафт).
    # Без этого EAR занижается на портретных видео (1080×1920),
    # т.к. y покрывает больше пикселей на единицу норм. координат.
    h, w = frame_bgr.shape[:2]
    landmarks_px = landmarks.copy()
    landmarks_px[:, 0] *= w   # x → пиксели
    landmarks_px[:, 1] *= h   # y → пиксели

    # Вычисляем EAR для каждого глаза (в пиксельных координатах)
    ear_right = compute_ear(landmarks_px, RIGHT_EYE)
    ear_left = compute_ear(landmarks_px, LEFT_EYE)
    ear_avg = (ear_left + ear_right) / 2.0

    # Вычисляем MAR
    mar = compute_mar(landmarks_px, MOUTH)

    return FrameResult(
        frame_idx=frame_idx,
        timestamp_ms=timestamp_ms,
        face_detected=True,
        ear_left=ear_left,
        ear_right=ear_right,
        ear_avg=ear_avg,
        mar=mar,
        landmarks=landmarks,
    )


# ================================================================
# Вычисление EAR и MAR
# ================================================================

def _dist(p1: np.ndarray, p2: np.ndarray) -> float:
    """Евклидово расстояние между двумя точками."""
    return float(np.linalg.norm(p1 - p2))


def compute_ear(landmarks: np.ndarray, eye_indices: dict) -> float:
    """
    Вычислить Eye Aspect Ratio по формуле:
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

    Где p1–p6 — 6 ключевых точек глаза.

    При открытых глазах EAR ≈ 0.25–0.30
    При закрытых глазах EAR < 0.20
    """
    p1 = landmarks[eye_indices["p1"]][:2]  # берём только x, y
    p2 = landmarks[eye_indices["p2"]][:2]
    p3 = landmarks[eye_indices["p3"]][:2]
    p4 = landmarks[eye_indices["p4"]][:2]
    p5 = landmarks[eye_indices["p5"]][:2]
    p6 = landmarks[eye_indices["p6"]][:2]

    # Вертикальные расстояния
    vert1 = _dist(p2, p6)
    vert2 = _dist(p3, p5)

    # Горизонтальное расстояние
    horiz = _dist(p1, p4)

    if horiz < 1e-6:
        return 0.0

    return (vert1 + vert2) / (2.0 * horiz)


def compute_mar(landmarks: np.ndarray, mouth_indices: dict) -> float:
    """
    Вычислить Mouth Aspect Ratio по формуле:
        MAR = (||upper1 - lower1|| + ||upper2 - lower2|| + ||upper3 - lower3||)
              / (2 * ||left_corner - right_corner||)

    При закрытом рте MAR ≈ 0.1–0.3
    При зевании MAR > 0.6
    """
    upper1 = landmarks[mouth_indices["upper_1"]][:2]
    upper2 = landmarks[mouth_indices["upper_2"]][:2]
    upper3 = landmarks[mouth_indices["upper_3"]][:2]
    lower1 = landmarks[mouth_indices["lower_1"]][:2]
    lower2 = landmarks[mouth_indices["lower_2"]][:2]
    lower3 = landmarks[mouth_indices["lower_3"]][:2]
    left = landmarks[mouth_indices["left_corner"]][:2]
    right = landmarks[mouth_indices["right_corner"]][:2]

    vert1 = _dist(upper1, lower1)
    vert2 = _dist(upper2, lower2)
    vert3 = _dist(upper3, lower3)
    horiz = _dist(left, right)

    if horiz < 1e-6:
        return 0.0

    return (vert1 + vert2 + vert3) / (2.0 * horiz)


# ================================================================
# Обработка целого видеофайла
# ================================================================

def process_video(
    video_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    max_frames: int | None = None,
    progress_every: int = 500,
) -> list[FrameResult]:
    """
    Обработать видеофайл целиком: извлечь EAR и MAR для каждого кадра.

    Args:
        video_path: путь к видеофайлу
        model_path: путь к модели face_landmarker.task
        max_frames: максимальное количество кадров (None = все)
        progress_every: выводить прогресс каждые N кадров

    Returns:
        Список FrameResult для каждого кадра
    """
    import cv2

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Видео не найдено: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Видео: {video_path.name}")
    print(f"  Разрешение: {width}x{height}, FPS: {fps:.2f}, "
          f"Кадров: {total_frames}")

    if max_frames:
        total_frames = min(total_frames, max_frames)
        print(f"  Обработка первых {total_frames} кадров")

    # Инициализируем FaceLandmarker
    landmarker = create_landmarker(model_path, mode="video")

    results = []
    frame_idx = 0
    no_face_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames and frame_idx >= max_frames:
            break

        # Вычисляем timestamp в мс
        timestamp_ms = int(frame_idx * 1000.0 / fps)

        # Обрабатываем кадр
        fr = process_frame(
            landmarker, frame, frame_idx, timestamp_ms, mode="video"
        )
        results.append(fr)

        if not fr.face_detected:
            no_face_count += 1

        frame_idx += 1

        if progress_every and frame_idx % progress_every == 0:
            pct = frame_idx / total_frames * 100
            print(f"  Обработано: {frame_idx}/{total_frames} "
                  f"({pct:.0f}%), лицо не найдено: {no_face_count}")

    cap.release()
    landmarker.close()

    face_rate = (1 - no_face_count / max(len(results), 1)) * 100
    print(f"  Завершено: {len(results)} кадров, "
          f"лицо найдено: {face_rate:.1f}%")

    return results


# ================================================================
# Утилиты
# ================================================================

def results_to_arrays(results: list[FrameResult]) -> dict[str, np.ndarray]:
    """
    Конвертировать список FrameResult в словарь numpy-массивов
    для удобной работы с временными рядами.

    Returns:
        dict с ключами: timestamps_ms, ear_avg, ear_left, ear_right,
                        mar, face_detected
    """
    n = len(results)
    data = {
        "timestamps_ms": np.zeros(n, dtype=np.int64),
        "ear_avg": np.zeros(n, dtype=np.float64),
        "ear_left": np.zeros(n, dtype=np.float64),
        "ear_right": np.zeros(n, dtype=np.float64),
        "mar": np.zeros(n, dtype=np.float64),
        "face_detected": np.zeros(n, dtype=bool),
    }

    for i, fr in enumerate(results):
        data["timestamps_ms"][i] = fr.timestamp_ms
        data["ear_avg"][i] = fr.ear_avg
        data["ear_left"][i] = fr.ear_left
        data["ear_right"][i] = fr.ear_right
        data["mar"][i] = fr.mar
        data["face_detected"][i] = fr.face_detected

    return data