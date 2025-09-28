#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

# === Укажи путь к файлу ===
INPUT_FILE = "train.csv"      # исходный файл

def clean_text(text: str) -> str:
    # нижний регистр
    text = text.lower()

    # удалить html-теги
    text = re.sub(r"<[^>]+>", " ", text)

    # 🔹 сначала удалить даты целиком
    # варианты: yyyy-mm-dd, dd.mm.yyyy, dd/mm/yy и т.д.
    text = re.sub(
        r"(\d{4}[-./]\d{1,2}[-./]\d{1,2})|(\d{1,2}[-./]\d{1,2}[-./]\d{2,4})",
        "",
        text,
    )

    # удалить смайлики и спецсимволы (оставляем буквы, цифры, знаки препинания, пробелы)
    text = re.sub(r"[^\w\s.,!?-]", " ", text, flags=re.UNICODE)

    # схлопывание повторяющихся символов (2 и более → 1)
    text = re.sub(r"(.)\1+", r"\1", text)

    # удалить последовательности цифр 3+ подряд
    text = re.sub(r"\d{3,}", "", text)

    # нормализация пробелов
    text = re.sub(r"\s+", " ", text).strip()

    # Удаление всех пробелов (если нужно, раскомментируй ↓)
    text = text.replace(" ", "")

    return text

if __name__ == "__main__":
    with open(INPUT_FILE, "r", encoding="utf-8") as fin:
        for line in fin:
            cleaned = clean_text(line)
            if cleaned:
                print(cleaned)
