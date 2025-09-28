#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Быстрый тест системы для машинного обучения
"""

import sys
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def quick_test():
    """Быстрый тест с небольшим датасетом"""
    print("🚀 Быстрый тест системы машинного обучения")
    print("=" * 50)
    
    # Проверяем доступные библиотеки
    print("📦 Проверка библиотек...")
    try:
        import pandas as pd
        print("✅ Pandas доступен")
    except ImportError:
        print("❌ Pandas недоступен")
        return
    
    try:
        import sklearn
        print("✅ Scikit-learn доступен")
    except ImportError:
        print("❌ Scikit-learn недоступен")
        return
    
    try:
        import numpy as np
        print("✅ NumPy доступен")
    except ImportError:
        print("❌ NumPy недоступен")
        return
    
    # Создаем небольшой тестовый датасет
    print("\n🔄 Создание тестового датасета...")
    test_data = [
        ("Привет, как дела?", 0),
        ("Ты идиот!", 1),
        ("Спасибо за помощь", 0),
        ("Убийца!", 1),
        ("Хорошая погода", 0),
        ("Ненавижу тебя", 1),
        ("Отличная работа", 0),
        ("Ты тупой", 1),
        ("Спасибо", 0),
        ("Маньяк", 1)
    ] * 100  # Увеличиваем датасет
    
    df = pd.DataFrame(test_data, columns=['text', 'label'])
    print(f"📊 Создан датасет: {len(df)} записей")
    
    # Предобработка
    print("🔄 Предобработка данных...")
    start_time = time.time()
    
    # Простая предобработка
    df['text_clean'] = df['text'].str.lower()
    
    # Векторизация
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['text_clean'])
    y = df['label']
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    prep_time = time.time() - start_time
    print(f"⏱️  Время предобработки: {prep_time:.2f} сек")
    
    # Обучение модели
    print("🔄 Обучение модели...")
    start_time = time.time()
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    print(f"⏱️  Время обучения: {train_time:.2f} сек")
    
    # Тестирование
    print("🔄 Тестирование модели...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"🎯 Точность: {accuracy:.4f}")
    
    # Тест на новых примерах
    print("\n🧪 Тестирование на новых примерах...")
    test_texts = [
        "Привет, как дела?",
        "Ты дебил!",
        "Спасибо за помощь",
        "Убийца и маньяк!"
    ]
    
    for text in test_texts:
        text_clean = text.lower()
        X_new = vectorizer.transform([text_clean])
        pred = model.predict(X_new)[0]
        proba = model.predict_proba(X_new)[0][1]
        
        print(f"📝 '{text}' -> {'Токсично' if pred else 'Нетоксично'} (вероятность: {proba:.3f})")
    
    print("\n✅ Тест завершен успешно!")
    print("🚀 Система готова для полного обучения!")
    print("\n📝 Следующие шаги:")
    print("1. Запустите: python train_sklearn_optimized.py")
    print("2. Или настройте CUDA и запустите: python train_optimized.py")

if __name__ == "__main__":
    quick_test()
