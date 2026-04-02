"""
Детектор зеваний на основе MAR (Mouth Aspect Ratio).

Зевание определяется как: MAR > порог в течение > min_duration секунд.
Это фильтрует короткие открытия рта (разговор, улыбка).
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class YawnEvent:
    """Одно событие зевания."""
    start_frame: int
    end_frame: int
    start_ms: int
    end_ms: int
    duration_ms: int
    max_mar: float


class YawnDetector:
    """
    Детектор зеваний.

    Параметры:
        mar_threshold: порог MAR для открытого рта (по умолчанию 0.6)
        min_duration_sec: мин. длительность для классификации как зевание (сек)
    """

    def __init__(
        self,
        mar_threshold: float = 0.6,
        min_duration_sec: float = 1.5,
    ):
        self.mar_threshold = mar_threshold
        self.min_duration_sec = min_duration_sec

    def detect(
        self,
        mar_values: np.ndarray,
        timestamps_ms: np.ndarray,
        face_detected: np.ndarray,
    ) -> list[YawnEvent]:
        """
        Обнаружить зевания в временном ряду MAR.

        Args:
            mar_values: массив MAR по кадрам
            timestamps_ms: массив временных меток в мс
            face_detected: массив bool — найдено ли лицо

        Returns:
            Список YawnEvent
        """
        yawns = []
        n = len(mar_values)
        min_duration_ms = self.min_duration_sec * 1000

        in_yawn = False
        yawn_start_frame = 0
        yawn_start_ms = 0
        yawn_max_mar = 0.0

        for i in range(n):
            if not face_detected[i]:
                # Лицо потеряно — если были в зевании, проверяем длительность
                if in_yawn:
                    duration_ms = int(timestamps_ms[i - 1]) - yawn_start_ms
                    if duration_ms >= min_duration_ms:
                        yawns.append(YawnEvent(
                            start_frame=yawn_start_frame,
                            end_frame=i - 1,
                            start_ms=yawn_start_ms,
                            end_ms=int(timestamps_ms[i - 1]),
                            duration_ms=duration_ms,
                            max_mar=yawn_max_mar,
                        ))
                    in_yawn = False
                continue

            mar = mar_values[i]
            ts = int(timestamps_ms[i])

            if not in_yawn:
                if mar > self.mar_threshold:
                    # Начало потенциального зевания
                    in_yawn = True
                    yawn_start_frame = i
                    yawn_start_ms = ts
                    yawn_max_mar = mar
            else:
                if mar > self.mar_threshold:
                    # Продолжение зевания
                    yawn_max_mar = max(yawn_max_mar, mar)
                else:
                    # Рот закрылся — проверяем длительность
                    duration_ms = ts - yawn_start_ms
                    if duration_ms >= min_duration_ms:
                        yawns.append(YawnEvent(
                            start_frame=yawn_start_frame,
                            end_frame=i,
                            start_ms=yawn_start_ms,
                            end_ms=ts,
                            duration_ms=duration_ms,
                            max_mar=yawn_max_mar,
                        ))
                    in_yawn = False

        # Обработка зевания, которое не завершилось до конца видео
        if in_yawn and n > 0:
            last_ts = int(timestamps_ms[n - 1])
            duration_ms = last_ts - yawn_start_ms
            if duration_ms >= min_duration_ms:
                yawns.append(YawnEvent(
                    start_frame=yawn_start_frame,
                    end_frame=n - 1,
                    start_ms=yawn_start_ms,
                    end_ms=last_ts,
                    duration_ms=duration_ms,
                    max_mar=yawn_max_mar,
                ))

        return yawns
