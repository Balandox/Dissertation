#!/usr/bin/env python3
"""
Скрипт загрузки датасета UTA-RLDD.

Поддерживает несколько методов загрузки:
  1. Kaggle API (автоматически)
  2. Ручная загрузка (инструкции)

Использование:
    python -m src.download_dataset [--method kaggle|manual] [--target_dir PATH]

Примечание:
    Для использования Kaggle API необходим файл ~/.kaggle/kaggle.json
    с вашим API-ключом. Получить его можно на https://www.kaggle.com/settings
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


DEFAULT_TARGET = Path("data/uta_rldd/videos")

# Kaggle идентификаторы датасета (существуют несколько копий)
KAGGLE_DATASETS = [
    "rishab260/uta-reallife-drowsiness-dataset",
    "minhngt02/uta-rldd",
]


def check_kaggle_cli() -> bool:
    """Проверить наличие Kaggle CLI."""
    try:
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_kaggle_auth() -> bool:
    """Проверить наличие Kaggle API-ключа."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True
    # Также проверяем переменные окружения
    return bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))


def download_via_kaggle(target_dir: Path, dataset_slug: str) -> bool:
    """Скачать датасет через Kaggle API."""
    print(f"\nЗагрузка {dataset_slug} через Kaggle API...")
    print("(Это может занять значительное время — датасет ~111 ГБ)")
    
    tmp_dir = target_dir.parent / "tmp_download"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        result = subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", dataset_slug,
                "-p", str(tmp_dir),
                "--unzip",
            ],
            capture_output=True, text=True, timeout=None,  # без таймаута
        )
        
        if result.returncode != 0:
            print(f"Ошибка загрузки: {result.stderr}")
            return False
        
        # Перемещаем файлы в целевую директорию
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Ищем корень датасета (папки с номерами участников)
        found_root = find_dataset_root(tmp_dir)
        if found_root:
            # Перемещаем содержимое
            for item in found_root.iterdir():
                dest = target_dir / item.name
                if not dest.exists():
                    shutil.move(str(item), str(dest))
            print(f"\nДатасет успешно загружен в {target_dir}")
        else:
            # Просто перемещаем всё
            for item in tmp_dir.iterdir():
                dest = target_dir / item.name
                if not dest.exists():
                    shutil.move(str(item), str(dest))
        
        # Удаляем временную директорию
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return True
        
    except Exception as e:
        print(f"Ошибка: {e}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False


def find_dataset_root(search_dir: Path) -> Path | None:
    """
    Найти корневую директорию датасета (содержит папки с номерами 1-60).
    """
    # Проверяем текущую директорию
    numbered_dirs = [
        d for d in search_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and 1 <= int(d.name) <= 60
    ]
    if len(numbered_dirs) >= 10:  # нашли достаточно папок участников
        return search_dir
    
    # Ищем глубже (один уровень)
    for subdir in search_dir.iterdir():
        if subdir.is_dir():
            numbered_dirs = [
                d for d in subdir.iterdir()
                if d.is_dir() and d.name.isdigit() and 1 <= int(d.name) <= 60
            ]
            if len(numbered_dirs) >= 10:
                return subdir
    
    return None


def handle_zip(zip_path: Path, target_dir: Path):
    """Распаковать ZIP-архив датасета."""
    print(f"\nРаспаковка {zip_path.name}...")
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, "r") as zf:
        total = len(zf.namelist())
        for i, member in enumerate(zf.namelist()):
            zf.extract(member, target_dir)
            if (i + 1) % 50 == 0 or i + 1 == total:
                print(f"  Распаковано: {i + 1}/{total}", end="\r")
    
    print(f"\nРаспаковка завершена: {target_dir}")
    
    # Ищем корень и перемещаем если нужно
    found_root = find_dataset_root(target_dir)
    if found_root and found_root != target_dir:
        print(f"Корень датасета найден в {found_root.relative_to(target_dir)}")
        # Перемещаем вверх
        for item in found_root.iterdir():
            dest = target_dir / item.name
            if not dest.exists():
                shutil.move(str(item), str(dest))


def print_manual_instructions(target_dir: Path):
    """Вывести инструкции для ручной загрузки."""
    print("\n" + "=" * 60)
    print("ИНСТРУКЦИЯ ПО РУЧНОЙ ЗАГРУЗКЕ UTA-RLDD")
    print("=" * 60)
    print()
    print("Способ 1: Kaggle (рекомендуется)")
    print("-" * 40)
    print("1. Откройте браузер и перейдите:")
    print("   https://www.kaggle.com/datasets/rishab260/uta-reallife-drowsiness-dataset")
    print("2. Войдите в свой аккаунт Kaggle (или создайте новый)")
    print("3. Нажмите кнопку 'Download' (↓)")
    print("4. Дождитесь скачивания ZIP-архива (~111 ГБ)")
    print(f"5. Распакуйте содержимое в: {target_dir.resolve()}")
    print()
    print("Способ 2: Установка Kaggle CLI")
    print("-" * 40)
    print("1. pip install kaggle")
    print("2. Перейдите на https://www.kaggle.com/settings")
    print("3. Нажмите 'Create New Token' → скачается kaggle.json")
    print("4. Поместите файл: ~/.kaggle/kaggle.json")
    print("5. chmod 600 ~/.kaggle/kaggle.json")
    print("6. Запустите снова: python -m src.download_dataset --method kaggle")
    print()
    print("Способ 3: Частичная загрузка (для быстрого тестирования)")
    print("-" * 40)
    print("Если не хотите качать все 111 ГБ, можно скачать несколько")
    print("участников вручную и проверить пайплайн на них.")
    print()
    print(f"Целевая директория: {target_dir.resolve()}")
    print(f"Ожидаемая структура:")
    print(f"  {target_dir}/")
    print(f"  ├── 1/")
    print(f"  │   ├── 0.mp4")
    print(f"  │   ├── 5.mp4")
    print(f"  │   └── 10.mp4")
    print(f"  ├── 2/ ...")
    print(f"  └── 60/")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Загрузка датасета UTA-RLDD"
    )
    parser.add_argument(
        "--method",
        choices=["kaggle", "manual", "unzip"],
        default="kaggle",
        help="Метод загрузки: kaggle (автоматически), manual (инструкции), "
             "unzip (распаковка уже скачанного ZIP)"
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default=str(DEFAULT_TARGET),
        help=f"Целевая директория (по умолчанию: {DEFAULT_TARGET})"
    )
    parser.add_argument(
        "--zip_path",
        type=str,
        default=None,
        help="Путь к ZIP-архиву (для --method unzip)"
    )
    args = parser.parse_args()
    
    target_dir = Path(args.target_dir)
    
    # Проверяем, не скачан ли уже датасет
    if target_dir.exists():
        numbered = [d for d in target_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if len(numbered) >= 10:
            print(f"Датасет уже найден в {target_dir} ({len(numbered)} участников)")
            print("Для повторной загрузки удалите директорию и запустите снова.")
            return
    
    if args.method == "unzip":
        if not args.zip_path:
            print("Ошибка: укажите --zip_path для метода unzip")
            sys.exit(1)
        zip_path = Path(args.zip_path)
        if not zip_path.exists():
            print(f"Ошибка: файл не найден: {zip_path}")
            sys.exit(1)
        handle_zip(zip_path, target_dir)
        
    elif args.method == "kaggle":
        if not check_kaggle_cli():
            print("Kaggle CLI не установлен.")
            print("Установите: pip install kaggle")
            print("\nИли используйте ручную загрузку: --method manual")
            sys.exit(1)
        
        if not check_kaggle_auth():
            print("Kaggle API-ключ не найден.")
            print("Получите ключ: https://www.kaggle.com/settings → Create New Token")
            print("Поместите в: ~/.kaggle/kaggle.json")
            sys.exit(1)
        
        # Пробуем каждый вариант датасета
        for slug in KAGGLE_DATASETS:
            if download_via_kaggle(target_dir, slug):
                break
        else:
            print("\nНе удалось загрузить датасет автоматически.")
            print_manual_instructions(target_dir)
            
    elif args.method == "manual":
        print_manual_instructions(target_dir)


if __name__ == "__main__":
    main()
