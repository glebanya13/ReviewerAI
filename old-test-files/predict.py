import pickle
import tensorflow as tf
import re
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class ToxicityPredictor:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.max_length = 200
        
    def preprocess_text(self, text):
        """Предобработка текста"""
        if pd.isna(text):
            return ""
        
        # Приводим к нижнему регистру
        text = str(text).lower()
        
        # Удаляем лишние символы, оставляем только буквы, цифры и пробелы
        text = re.sub(r'[^а-яёa-z0-9\s]', ' ', text)
        
        # Удаляем множественные пробелы
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def load_model(self, model_path='model_tf.h5', tokenizer_path='tokenizer_tf.pkl'):
        """Загрузка обученной модели"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            print("✅ Модель успешно загружена!")
            return True
        except Exception as e:
            print(f"❌ Ошибка при загрузке модели: {e}")
            return False
    
    def predict_toxicity(self, text):
        """Предсказание токсичности для текста"""
        if self.model is None or self.tokenizer is None:
            print("❌ Модель не загружена! Сначала вызовите load_model()")
            return None

        # Предобработка текста
        processed_text = self.preprocess_text(text)
        if not processed_text:
            print(f"⚠️ После предобработки текст пустой: '{text}'")
            return {
                'text': text,
                'processed_text': processed_text,
                'is_toxic': False,
                'toxicity_probability': 0.0,
                'confidence': 1.0,
                'error': 'Пустой текст после предобработки'
            }

        # Определяем тип токенизатора
        if hasattr(self.tokenizer, 'texts_to_sequences'):
            # Keras Tokenizer
            sequence = self.tokenizer.texts_to_sequences([processed_text])
            if not sequence or not any(sequence[0]):
                print(f"⚠️ После токенизации последовательность пуста: '{processed_text}'")
                return {
                    'text': text,
                    'processed_text': processed_text,
                    'is_toxic': False,
                    'toxicity_probability': 0.0,
                    'confidence': 1.0,
                    'error': 'Пустая последовательность после токенизации'
                }
            padded_sequence = pad_sequences(sequence, maxlen=self.max_length, padding='post', truncating='post')
            if padded_sequence.shape[1] == 0:
                print(f"⚠️ После pad_sequences получена пустая последовательность: '{processed_text}'")
                return {
                    'text': text,
                    'processed_text': processed_text,
                    'is_toxic': False,
                    'toxicity_probability': 0.0,
                    'confidence': 1.0,
                    'error': 'Пустая последовательность после pad_sequences'
                }
            input_data = padded_sequence
        elif hasattr(self.tokenizer, 'transform'):
            # TfidfVectorizer
            input_data = self.tokenizer.transform([processed_text]) if isinstance(processed_text, list) else self.tokenizer.transform([processed_text])
        else:
            print("❌ Неизвестный тип токенизатора!")
            return None

        # Предсказание
        try:
            probability = self.model.predict(input_data, verbose=0)[0][0]
            prediction = 1 if probability > 0.5 else 0
        except Exception as e:
            print(f"❌ Ошибка при предсказании: {e}")
            return {
                'text': text,
                'processed_text': processed_text,
                'is_toxic': False,
                'toxicity_probability': 0.0,
                'confidence': 1.0,
                'error': str(e)
            }

        return {
            'text': text,
            'processed_text': processed_text,
            'is_toxic': bool(prediction),
            'toxicity_probability': probability,
            'confidence': max(probability, 1 - probability)
        }
    
    def predict_batch(self, texts):
        """Предсказание токсичности для списка текстов"""
        results = []
        for text in texts:
            result = self.predict_toxicity(text)
            if result:
                results.append(result)
        return results

def main():
    """Пример использования"""
    # Создаем предсказатель
    predictor = ToxicityPredictor()
    
    # Загружаем модель
    if not predictor.load_model():
        print("Сначала обучите модель, запустив train.py")
        return
    
    # Тестируем на примерах
    test_texts = [
        "сукаблять",
        "у дениса маленькие сиськи",
        "у глеюаса маленький член",
        "Отличный продукт, очень доволен!",
        "у жени большие сиськи чем у дениса",
        "Ты дебил и идиот!",
        "Спасибо за помощь",
        "Убийца и маньяк!",
        "Хорошая погода сегодня",
        "Иди нахуй!",
        "Отличная работа!",
        "Сука блядь!",
        "Сначала помоги себе сам, а потом другим. Поэтому я бл не помогу тебе. Понял меня, падла?"
    ]
    
    print("\n" + "="*60)
    print("ПРОВЕРКА ТОКСИЧНОСТИ ТЕКСТОВ")
    print("="*60)
    
    for text in test_texts:
        result = predictor.predict_toxicity(text)
        if result:
            status = "🔴 ТОКСИЧНО" if result['is_toxic'] else "🟢 НЕТОКСИЧНО"
            print(f"\nТекст: '{result['text']}'")
            print(f"Результат: {status}")
            print(f"Вероятность токсичности: {result['toxicity_probability']}")
            print(f"Уверенность: {result['confidence']}")
            print("-" * 40)

if __name__ == "__main__":
    main()
