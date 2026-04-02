"""
Скачивание модели MediaPipe FaceLandmarker.

Использование:
    python -m src.download_model
"""

import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = Path("models/face_landmarker.task")


def download_model():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"Модель уже существует: {MODEL_PATH} ({size_mb:.1f} МБ)")
        return

    print(f"Скачивание модели FaceLandmarker...")
    print(f"  URL: {MODEL_URL}")
    print(f"  Сохранение в: {MODEL_PATH}")

    try:
        urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"  Готово! Размер: {size_mb:.1f} МБ")
    except Exception as e:
        print(f"  Ошибка загрузки: {e}")
        print(f"\n  Скачайте вручную:")
        print(f"    wget -O {MODEL_PATH} {MODEL_URL}")
        print(f"  Или через браузер: {MODEL_URL}")


if __name__ == "__main__":
    download_model()
