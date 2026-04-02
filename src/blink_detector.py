"""
Детектор морганий на основе конечного автомата (FSM).

Входные данные: временной ряд EAR (покадрово).
Выходные данные: список событий морганий с временными метками и длительностью.

Конечный автомат:
    OPEN → EAR падает ниже порога → CLOSED
    CLOSED → EAR поднимается выше порога → OPEN (моргание завершено)

Для защиты от шума требуется минимум EAR_CONSEC_FRAMES кадров
подряд ниже порога, чтобы зафиксировать начало моргания.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class BlinkEvent:
    """Одно событие моргания."""
    start_frame: int       # кадр начала (EAR упал ниже порога)
    end_frame: int         # кадр конца (EAR вернулся выше порога)
    start_ms: int          # временная метка начала (мс)
    end_ms: int            # временная метка конца (мс)
    duration_ms: int       # длительность моргания (мс)
    min_ear: float         # минимальное EAR во время моргания
    is_prolonged: bool     # длительность > порога (500 мс)


class BlinkDetector:
    """
    Детектор морганий на основе конечного автомата.

    Параметры:
        ear_threshold: порог EAR для определения закрытых глаз (по умолчанию 0.21)
        min_consec_frames: мин. кадров подряд ниже порога для начала моргания
        prolonged_threshold_ms: порог для длительного моргания (мс)
    """

    # Состояния автомата
    STATE_OPEN = "open"
    STATE_CLOSED = "closed"

    def __init__(
        self,
        ear_threshold: float = 0.21,
        min_consec_frames: int = 2,
        prolonged_threshold_ms: int = 500,
    ):
        self.ear_threshold = ear_threshold
        self.min_consec_frames = min_consec_frames
        self.prolonged_threshold_ms = prolonged_threshold_ms

        # Внутреннее состояние
        self._state = self.STATE_OPEN
        self._consec_below = 0       # счётчик кадров подряд ниже порога
        self._blink_start_frame = 0
        self._blink_start_ms = 0
        self._blink_min_ear = 1.0

    def reset(self):
        """Сбросить состояние автомата."""
        self._state = self.STATE_OPEN
        self._consec_below = 0
        self._blink_start_frame = 0
        self._blink_start_ms = 0
        self._blink_min_ear = 1.0

    def detect(
        self,
        ear_values: np.ndarray,
        timestamps_ms: np.ndarray,
        face_detected: np.ndarray,
    ) -> list[BlinkEvent]:
        """
        Обнаружить моргания в временном ряду EAR.

        Args:
            ear_values: массив значений EAR (avg) по кадрам
            timestamps_ms: массив временных меток в мс
            face_detected: массив bool — найдено ли лицо

        Returns:
            Список BlinkEvent
        """
        self.reset()
        blinks = []
        n = len(ear_values)

        for i in range(n):
            # Если лицо не найдено — пропускаем, но не сбрасываем состояние
            if not face_detected[i]:
                continue

            ear = ear_values[i]
            ts = int(timestamps_ms[i])

            if self._state == self.STATE_OPEN:
                if ear < self.ear_threshold:
                    self._consec_below += 1
                    if self._consec_below >= self.min_consec_frames:
                        # Переход в состояние CLOSED
                        self._state = self.STATE_CLOSED
                        # Начало моргания — откатываемся к первому кадру ниже порога
                        offset = self._consec_below - 1
                        self._blink_start_frame = i - offset
                        self._blink_start_ms = int(timestamps_ms[i - offset])
                        self._blink_min_ear = min(
                            ear_values[i - offset:i + 1]
                        )
                else:
                    self._consec_below = 0

            elif self._state == self.STATE_CLOSED:
                if ear < self.ear_threshold:
                    # Всё ещё закрыты
                    self._blink_min_ear = min(self._blink_min_ear, ear)
                else:
                    # Глаза открылись — моргание завершено
                    duration_ms = ts - self._blink_start_ms

                    blink = BlinkEvent(
                        start_frame=self._blink_start_frame,
                        end_frame=i,
                        start_ms=self._blink_start_ms,
                        end_ms=ts,
                        duration_ms=duration_ms,
                        min_ear=self._blink_min_ear,
                        is_prolonged=duration_ms > self.prolonged_threshold_ms,
                    )
                    blinks.append(blink)

                    # Возврат в OPEN
                    self._state = self.STATE_OPEN
                    self._consec_below = 0
                    self._blink_min_ear = 1.0

        return blinks
