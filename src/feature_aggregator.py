"""
Агрегатор признаков: вычисляет 6-компонентный вектор по временным окнам.

Выходной вектор:
    [avg_EAR, blink_rate, avg_blink_duration, prolonged_blink_count, PERCLOS, yawn_count]

Окна:
    - PERCLOS, blink_rate, avg_blink_duration, prolonged_blink_count: 60 сек
    - yawn_count: 5 мин (300 сек)
    - Шаг: каждые 10 сек (настраивается)
"""

from dataclasses import dataclass
import numpy as np

from src.blink_detector import BlinkEvent
from src.yawn_detector import YawnEvent


@dataclass
class FeatureVector:
    """Один вектор признаков за временное окно."""
    window_start_ms: int      # начало окна (мс)
    window_end_ms: int        # конец окна (мс)
    avg_ear: float            # среднее EAR за окно
    blink_rate: float         # морганий в минуту
    avg_blink_duration: float # средняя длительность моргания (мс)
    prolonged_blink_count: int  # кол-во длинных морганий (>500 мс)
    perclos: float            # доля кадров с закрытыми глазами [0, 1]
    yawn_count: int           # количество зеваний за окно yawn


class FeatureAggregator:
    """
    Агрегирует покадровые данные в вектор признаков по скользящим окнам.

    Параметры:
        window_sec: длина окна для EAR-метрик (сек)
        yawn_window_sec: длина окна для подсчёта зеваний (сек)
        step_sec: шаг скольжения окна (сек)
        ear_threshold: порог EAR для PERCLOS
    """

    def __init__(
        self,
        window_sec: float = 60.0,
        yawn_window_sec: float = 300.0,
        step_sec: float = 10.0,
        ear_threshold: float = 0.21,
    ):
        self.window_ms = window_sec * 1000
        self.yawn_window_ms = yawn_window_sec * 1000
        self.step_ms = step_sec * 1000
        self.ear_threshold = ear_threshold

    def aggregate(
        self,
        ear_values: np.ndarray,
        timestamps_ms: np.ndarray,
        face_detected: np.ndarray,
        blinks: list[BlinkEvent],
        yawns: list[YawnEvent],
    ) -> list[FeatureVector]:
        """
        Вычислить вектор признаков для каждого временного окна.

        Args:
            ear_values: EAR (avg) по кадрам
            timestamps_ms: временные метки (мс)
            face_detected: лицо найдено (bool)
            blinks: список событий морганий
            yawns: список событий зеваний

        Returns:
            Список FeatureVector для каждого окна
        """
        if len(timestamps_ms) == 0:
            return []

        total_duration_ms = int(timestamps_ms[-1] - timestamps_ms[0])

        # Минимальное окно — нужно хотя бы window_sec данных
        if total_duration_ms < self.window_ms:
            # Если видео короче окна — вычисляем один вектор по всему видео
            fv = self._compute_window(
                ear_values, timestamps_ms, face_detected,
                blinks, yawns,
                int(timestamps_ms[0]), int(timestamps_ms[-1]),
            )
            return [fv] if fv else []

        features = []
        t_start = int(timestamps_ms[0])
        t_end_max = int(timestamps_ms[-1])

        while t_start + self.window_ms <= t_end_max:
            t_end = int(t_start + self.window_ms)

            fv = self._compute_window(
                ear_values, timestamps_ms, face_detected,
                blinks, yawns,
                t_start, t_end,
            )
            if fv:
                features.append(fv)

            t_start += int(self.step_ms)

        return features

    def _compute_window(
        self,
        ear_values: np.ndarray,
        timestamps_ms: np.ndarray,
        face_detected: np.ndarray,
        blinks: list[BlinkEvent],
        yawns: list[YawnEvent],
        t_start: int,
        t_end: int,
    ) -> FeatureVector | None:
        """Вычислить признаки для одного временного окна [t_start, t_end]."""

        # Выбираем кадры, попадающие в окно
        mask = (timestamps_ms >= t_start) & (timestamps_ms < t_end)
        mask_face = mask & face_detected

        n_frames = mask.sum()
        n_face = mask_face.sum()

        if n_face < 10:  # слишком мало данных
            return None

        # 1. avg_EAR — среднее EAR за окно (только кадры с лицом)
        avg_ear = float(ear_values[mask_face].mean())

        # 2. PERCLOS — доля кадров с EAR < порог (от всех кадров с лицом)
        closed_frames = (ear_values[mask_face] < self.ear_threshold).sum()
        perclos = float(closed_frames / n_face)

        # 3. Моргания, попадающие в окно
        window_blinks = [
            b for b in blinks
            if b.start_ms >= t_start and b.start_ms < t_end
        ]

        # blink_rate — морганий в минуту
        window_duration_min = (t_end - t_start) / 60000.0
        blink_rate = len(window_blinks) / window_duration_min if window_duration_min > 0 else 0.0

        # avg_blink_duration — средняя длительность в мс
        if window_blinks:
            avg_blink_duration = float(np.mean([b.duration_ms for b in window_blinks]))
        else:
            avg_blink_duration = 0.0

        # prolonged_blink_count — количество длинных морганий (>500 мс)
        prolonged_blink_count = sum(1 for b in window_blinks if b.is_prolonged)

        # 4. Зевания — используем расширенное окно (yawn_window)
        # Для yawn_count берём окно от (t_end - yawn_window) до t_end
        yawn_t_start = max(int(timestamps_ms[0]), t_end - int(self.yawn_window_ms))
        window_yawns = [
            y for y in yawns
            if y.start_ms >= yawn_t_start and y.start_ms < t_end
        ]
        yawn_count = len(window_yawns)

        return FeatureVector(
            window_start_ms=t_start,
            window_end_ms=t_end,
            avg_ear=round(avg_ear, 6),
            blink_rate=round(blink_rate, 2),
            avg_blink_duration=round(avg_blink_duration, 1),
            prolonged_blink_count=prolonged_blink_count,
            perclos=round(perclos, 6),
            yawn_count=yawn_count,
        )
