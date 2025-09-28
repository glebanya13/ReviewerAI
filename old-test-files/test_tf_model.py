import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TensorFlowModelTester:
    """Класс для тестирования обученной TensorFlow модели"""
    
    def __init__(self, model_path='model_tf.h5', tokenizer_path='tokenizer_tf.pkl'):
        self.model = None
        self.vectorizer = None
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

    def load_model_and_vectorizer(self):
        """Загрузка обученной модели и векторизатора"""
        try:
            print(f"🔄 Загрузка модели: {self.model_path}...")
            self.model = tf.keras.models.load_model(self.model_path)

            print(f"🔄 Загрузка векторизатора: {self.tokenizer_path}...")
            with open(self.tokenizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
                
            print("✅ Модель и векторизатор успешно загружены!")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            return False
    
    def predict_text(self, text):
        """Предсказание токсичности для одного текста"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Модель не загружена!")
        
        # Векторизация текста
        text_tfidf = self.vectorizer.transform([text])
        text_dense = text_tfidf.toarray()
        
        # Предсказание
        prediction_proba = self.model.predict(text_dense, verbose=0)[0][0]
        prediction_class = int(prediction_proba > 0.5)
        
        return {
            'text': text,
            'probability': float(prediction_proba),
            'prediction': prediction_class,
            'label': 'Токсичный' if prediction_class == 1 else 'Нетоксичный',
            'confidence': f"{prediction_proba*100:.1f}%" if prediction_class == 1 else f"{(1-prediction_proba)*100:.1f}%"
        }
    
    def test_multiple_texts(self, texts):
        """Тестирование списка текстов"""
        results = []
        
        print(f"🧪 Тестирование {len(texts)} комментариев...")
        print("=" * 80)
        
        for i, text in enumerate(texts, 1):
            result = self.predict_text(text)
            results.append(result)
            
            # Красивый вывод результата
            emoji = "🔴" if result['prediction'] == 1 else "🟢"
            print(f"{emoji} Тест {i:2d} | {result['label']:12} | {result['confidence']:6} | {text[:60]}...")
            
        return results
    
    def get_statistics(self, results):
        """Получение статистики по результатам тестирования"""
        total = len(results)
        toxic_count = sum(1 for r in results if r['prediction'] == 1)
        non_toxic_count = total - toxic_count
        
        avg_toxic_prob = np.mean([r['probability'] for r in results if r['prediction'] == 1]) if toxic_count > 0 else 0
        avg_non_toxic_prob = np.mean([1 - r['probability'] for r in results if r['prediction'] == 0]) if non_toxic_count > 0 else 0
        
        return {
            'total': total,
            'toxic': toxic_count,
            'non_toxic': non_toxic_count,
            'toxic_percentage': (toxic_count / total) * 100,
            'avg_toxic_confidence': avg_toxic_prob,
            'avg_non_toxic_confidence': avg_non_toxic_prob
        }

def main():
    """Основная функция тестирования"""
    print("🚀 ТЕСТИРОВАНИЕ МОДЕЛИ TENSORFLOW")
    print("=" * 50)
    print(f"📅 Время тестирования: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Выбор модели для тестирования
    print("Выберите модель для тестирования:")
    print("1. Оригинальная модель (model_tf.h5)")
    print("2. Дообученная модель (model_tf_2.h5)")

    choice = input("Введите номер (1 или 2): ").strip()

    if choice == "2":
        tester = TensorFlowModelTester('model_tf_2.h5', '../old_models/tokenizer_tf_2.pkl')
        print("📊 Выбрана дообученная модель")
    else:
        tester = TensorFlowModelTester('../model_tf.h5', 'old_models/tokenizer_tf.pkl')
        print("📊 Выбрана оригинальная модель")

    # Загрузка модели
    if not tester.load_model_and_vectorizer():
        return
    
    # 20 разнообразных тестовых комментариев
    test_comments = [
        # Нетоксичные комментарии
        "Отличный товар, очень доволен покупкой! Рекомендую всем.",
        "Спасибо за быструю доставку. Качество на высоте.",
        "Хорошая цена за такое качество. Буду заказывать еще.",
        "Товар соответствует описанию. Упаковка аккуратная.",
        "Отзывчивый продавец, помог с выбором. Молодцы!",
        "Пришло быстро, все в порядке. Спасибо за сервис.",
        "Качественный продукт, использую уже месяц - все отлично.",
        "Хороший магазин, всегда заказываю здесь. Надежно.",
        "Товар как на фото, размер подошел идеально.",
        "Отличное соотношение цена-качество. Рекомендую.",
        
        # Токсичные комментарии
        "Полный отстой! Деньги на ветер, не покупайте это дерьмо!",
        "Ужасный товар, продавец обманщик! Кидают на деньги!",
        "Фу, какая гадость! Отвратительное качество, верните деньги!",
        "Это просто издевательство! Хуже товара я не видел!",
        "Мошенники! Прислали полную чушь вместо нормального товара!",
        "Идиоты, не умеют работать! Все сломано и работает плохо!",
        "Наглые обманщики! Товар говно, сервис еще хуже!",
        "Дерьмовый магазин, кидают покупателей! Не связывайтесь!",
        "Ужас просто! Хлам за такие деньги, это воровство!",
        "Полнейший бред! Выкинул деньги на это убожество!"
    ]
    
    print()
    
    # Тестирование
    results = tester.test_multiple_texts(test_comments)
    
    print("\n" + "=" * 80)
    print("📊 СТАТИСТИКА ТЕСТИРОВАНИЯ")
    print("=" * 80)
    
    # Получение и вывод статистики
    stats = tester.get_statistics(results)
    
    print(f"📝 Всего протестировано комментариев: {stats['total']}")
    print(f"🔴 Токсичных: {stats['toxic']} ({stats['toxic_percentage']:.1f}%)")
    print(f"🟢 Нетоксичных: {stats['non_toxic']} ({100-stats['toxic_percentage']:.1f}%)")
    
    if stats['toxic'] > 0:
        print(f"📈 Средняя уверенность для токсичных: {stats['avg_toxic_confidence']*100:.1f}%")
    if stats['non_toxic'] > 0:
        print(f"📈 Средняя уверенность для нетоксичных: {stats['avg_non_toxic_confidence']*100:.1f}%")
    
    print("\n" + "=" * 80)
    print("🎯 ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        status_icon = "🔴 ТОКСИЧНЫЙ  " if result['prediction'] == 1 else "🟢 НЕТОКСИЧНЫЙ"
        print(f"\n{i:2d}. {status_icon} (уверенность: {result['confidence']})")
        print(f"    💬 \"{result['text']}\"")
        print(f"    📊 Вероятность токсичности: {result['probability']:.3f}")
    
    print("\n" + "=" * 80)
    print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
    print("=" * 80)

if __name__ == "__main__":
    main()
