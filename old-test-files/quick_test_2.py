import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def test_fine_tuned_model():
    """Быстрое тестирование дообученной модели"""
    print("🚀 БЫСТРОЕ ТЕСТИРОВАНИЕ ДООБУЧЕННОЙ МОДЕЛИ")
    print("=" * 60)
    print(f"📅 Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        # Загрузка дообученной модели
        print("🔄 Загрузка дообученной модели...")
        model = tf.keras.models.load_model('model_tf_2.h5')

        print("🔄 Загрузка обновленного векторизатора...")
        with open('../old_models/tokenizer_tf_2.pkl', 'rb') as f:
            vectorizer = pickle.load(f)

        print("✅ Модель и векторизатор загружены!")
        print()

        # Тестовые комментарии
        test_texts = [
            "Отличный товар, рекомендую к покупке!",
            "Спасибо за качественный сервис.",
            "Полное дерьмо, не покупайте!",
            "Ужасный товар, обманщики!",
            "Хорошее качество за разумную цену.",
            "Идиоты, все сломано!"
        ]

        print("🧪 Тестирование на примерах:")
        print("-" * 60)

        for i, text in enumerate(test_texts, 1):
            # Векторизация
            text_tfidf = vectorizer.transform([text])
            text_dense = text_tfidf.toarray()

            # Предсказание
            prob = model.predict(text_dense, verbose=0)[0][0]
            pred = int(prob > 0.5)
            label = "ТОКСИЧНЫЙ" if pred == 1 else "НЕТОКСИЧНЫЙ"
            confidence = f"{prob*100:.1f}%" if pred == 1 else f"{(1-prob)*100:.1f}%"

            emoji = "🔴" if pred == 1 else "🟢"
            print(f"{emoji} {i}. {label} ({confidence})")
            print(f"   💬 \"{text}\"")
            print(f"   📊 Вероятность: {prob:.3f}")
            print()

        print("✅ БЫСТРОЕ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    test_fine_tuned_model()
