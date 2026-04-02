# Видеомодуль системы предупреждения засыпания водителя

Модуль извлечения поведенческих признаков усталости из видеопотока.
Часть магистерской диссертации по разработке интеллектуальной системы предупреждения засыпания водителя.

## Выходной вектор признаков

```
[avg_EAR, blink_rate, avg_blink_duration, prolonged_blink_count, PERCLOS, yawn_count]
```

| Признак                 | Описание                                  | Единицы    |
|-------------------------|-------------------------------------------|------------|
| `avg_EAR`               | Среднее Eye Aspect Ratio за окно          | безразм.   |
| `blink_rate`            | Частота морганий                          | морг./мин  |
| `avg_blink_duration`    | Средняя длительность моргания             | мс         |
| `prolonged_blink_count` | Количество длинных морганий (>500 мс)     | шт.        |
| `PERCLOS`               | Доля времени с закрытыми глазами          | 0–1        |
| `yawn_count`            | Количество зеваний за 5 минут             | шт.        |

## Датасет

**UTA-RLDD** (University of Texas at Arlington Real-Life Drowsiness Dataset)
- 60 участников, 180 видеозаписей (~10 мин каждое)
- 3 класса: alert (бодрый), low vigilant (предусталость), drowsy (сонливый)
- Реальная сонливость (не симулированная)
- Подробности: `docs/dataset_analysis.md`

## Быстрый старт

### 1. Установка зависимостей

```bash
pip install opencv-python mediapipe numpy pandas matplotlib seaborn
```

### 2. Загрузка датасета

```bash
# Вариант A: через Kaggle CLI
pip install kaggle
python -m src.download_dataset --method kaggle

# Вариант B: инструкции для ручной загрузки
python -m src.download_dataset --method manual

# Вариант C: распаковка скачанного ZIP
python -m src.download_dataset --method unzip --zip_path ~/Downloads/archive.zip
```

### 3. Парсинг метаданных

```bash
python -m src.dataset_parser
```

Создаст `outputs/dataset_metadata.csv` с метаинформацией о всех видео.

### 4. Извлечение признаков (будет реализовано далее)

```bash
python -m src.extract_features --metadata outputs/dataset_metadata.csv
```

## Структура проекта

```
drowsiness_detection/
├── configs/
│   ├── __init__.py
│   └── config.py              # Параметры: пороги, окна, пути
├── src/
│   ├── __init__.py
│   ├── download_dataset.py    # Загрузка UTA-RLDD
│   ├── dataset_parser.py      # Парсинг метаданных → CSV
│   ├── landmarks.py           # [шаг 2] MediaPipe Face Mesh
│   ├── ear_calculator.py      # [шаг 3] Eye Aspect Ratio
│   ├── blink_detector.py      # [шаг 4] Детектор морганий (FSM)
│   ├── mar_yawn_detector.py   # [шаг 5] MAR + детектор зеваний
│   ├── feature_aggregator.py  # [шаг 6] Агрегация по окнам
│   └── extract_features.py    # [шаг 7] Обработка всего датасета
├── data/
│   └── uta_rldd/
│       └── videos/            # Видеофайлы датасета (не в git)
├── outputs/                   # Результаты обработки
├── docs/
│   └── dataset_analysis.md    # Анализ и выбор датасета
└── README.md
```

## Рабочий процесс (пайплайн)

```
Видео → MediaPipe → Landmarks → EAR/MAR → Детекторы → Агрегация → CSV
         Face Mesh    468 точек    покадрово   моргания    по окнам   признаки
                                               зевания     60 сек     + метки
```

## Ссылки

- Датасет: https://sites.google.com/view/utarldd/home
- Kaggle: https://www.kaggle.com/datasets/rishab260/uta-reallife-drowsiness-dataset
- Статья: Ghoddoosian et al., "A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection", CVPRW 2019
