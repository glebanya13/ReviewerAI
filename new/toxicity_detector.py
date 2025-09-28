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
        """Улучшенная предобработка текста"""
        # Проверяем None и NaN
        if text is None or pd.isna(text):
            return ""
        
        # Преобразуем в строку
        text = str(text)
        
        # Проверяем пустую строку
        if not text.strip():
            return ""
        
        try:
            # Приводим к нижнему регистру
            text = text.lower()
            
            # Удаляем лишние символы, оставляем только буквы, цифры и пробелы
            # Добавляем поддержку больше символов для лучшей токенизации
            text = re.sub(r'[^а-яёa-z0-9\s]', ' ', text)
            
            # Удаляем множественные пробелы
            text = re.sub(r'\s+', ' ', text)
            
            # Удаляем пробелы в начале и конце
            text = text.strip()
            
            # Финальная проверка на пустоту
            if not text:
                return ""
                
            return text
            
        except Exception as e:
            print(f"⚠️ Ошибка при предобработке текста '{str(text)[:50]}...': {e}")
            return ""
    
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

        # Сохраняем оригинальный текст
        original_text = text
        
        # Предобработка текста
        processed_text = self.preprocess_text(text)
        if not processed_text:
            print(f"⚠️ После предобработки текст пустой: '{text}'")
            return {
                'text': original_text,
                'processed_text': processed_text,
                'is_toxic': False,
                'toxicity_probability': 0.0,
                'confidence': 1.0,
                'error': 'Пустой текст после предобработки'
            }

        # Определяем тип токенизатора
        try:
            if hasattr(self.tokenizer, 'texts_to_sequences'):
                # Keras Tokenizer (старый тип)
                sequence = self.tokenizer.texts_to_sequences([processed_text])
                if not sequence or not any(sequence[0]):
                    print(f"⚠️ После токенизации последовательность пуста: '{processed_text}'")
                    return {
                        'text': original_text,
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
                        'text': original_text,
                        'processed_text': processed_text,
                        'is_toxic': False,
                        'toxicity_probability': 0.0,
                        'confidence': 1.0,
                        'error': 'Пустая последовательность после pad_sequences'
                    }
                input_data = padded_sequence
                
            elif hasattr(self.tokenizer, 'transform'):
                # TfidfVectorizer
                # Проверяем, что текст не пустой
                if not processed_text or not processed_text.strip():
                    return {
                        'text': original_text,
                        'processed_text': processed_text,
                        'is_toxic': False,
                        'toxicity_probability': 0.0,
                        'confidence': 1.0,
                        'error': 'Пустой текст для TF-IDF'
                    }
                
                # Преобразуем в TF-IDF вектор
                input_data = self.tokenizer.transform([processed_text])
                
                # Проверяем размерность векторизации
                if input_data.shape[1] == 0:
                    return {
                        'text': original_text,
                        'processed_text': processed_text,
                        'is_toxic': False,
                        'toxicity_probability': 0.0,
                        'confidence': 1.0,
                        'error': 'Пустой TF-IDF вектор'
                    }
                    
            else:
                print("❌ Неизвестный тип токенизатора!")
                return {
                    'text': original_text,
                    'processed_text': processed_text,
                    'is_toxic': False,
                    'toxicity_probability': 0.0,
                    'confidence': 1.0,
                    'error': 'Неизвестный тип токенизатора'
                }

        except Exception as e:
            print(f"❌ Ошибка при токенизации: {e}")
            return {
                'text': original_text,
                'processed_text': processed_text,
                'is_toxic': False,
                'toxicity_probability': 0.0,
                'confidence': 1.0,
                'error': f'Ошибка токенизации: {str(e)}'
            }

        # Предсказание
        try:
            # Преобразуем разреженную матрицу в плотную для TF-IDF
            if hasattr(input_data, 'toarray'):
                input_data = input_data.toarray()
                
            prediction_result = self.model.predict(input_data, verbose=0)
            
            # Обрабатываем результат предсказания
            if len(prediction_result.shape) == 2 and prediction_result.shape[1] == 1:
                probability = float(prediction_result[0][0])
            elif len(prediction_result.shape) == 1:
                probability = float(prediction_result[0])
            else:
                probability = float(prediction_result.flatten()[0])
                
            # Ограничиваем вероятность диапазоном [0, 1]
            probability = max(0.0, min(1.0, probability))
            prediction = 1 if probability > 0.5 else 0
            
        except Exception as e:
            print(f"❌ Ошибка при предсказании: {e}")
            return {
                'text': original_text,
                'processed_text': processed_text,
                'is_toxic': False,
                'toxicity_probability': 0.0,
                'confidence': 1.0,
                'error': f'Ошибка предсказания: {str(e)}'
            }

        return {
            'text': original_text,
            'processed_text': processed_text,
            'is_toxic': bool(prediction),
            'toxicity_probability': probability,
            'confidence': max(probability, 1 - probability)
        }
    
    def predict_batch(self, texts):
        """Предсказание токсичности для списка текстов с улучшенной обработкой ошибок"""
        results = []
        if not texts:
            return results
            
        for text in texts:
            # Проверяем, что текст не None и не пустой
            if text is None:
                results.append({
                    'text': None,
                    'processed_text': '',
                    'is_toxic': False,
                    'toxicity_probability': 0.0,
                    'confidence': 1.0,
                    'error': 'Пустой входной текст (None)'
                })
                continue
                
            # Преобразуем в строку если нужно
            text_str = str(text) if not isinstance(text, str) else text
            
            # Обрабатываем пустые строки
            if not text_str.strip():
                results.append({
                    'text': text_str,
                    'processed_text': '',
                    'is_toxic': False,
                    'toxicity_probability': 0.0,
                    'confidence': 1.0,
                    'error': 'Пустой входной текст'
                })
                continue
            
            try:
                result = self.predict_toxicity(text_str)
                if result:
                    results.append(result)
                else:
                    # Если predict_toxicity вернул None
                    results.append({
                        'text': text_str,
                        'processed_text': '',
                        'is_toxic': False,
                        'toxicity_probability': 0.0,
                        'confidence': 1.0,
                        'error': 'Ошибка предсказания (вернул None)'
                    })
            except Exception as e:
                # Обрабатываем любые неожиданные ошибки
                print(f"❌ Неожиданная ошибка при обработке текста '{text_str[:50]}...': {e}")
                results.append({
                    'text': text_str,
                    'processed_text': '',
                    'is_toxic': False,
                    'toxicity_probability': 0.0,
                    'confidence': 1.0,
                    'error': f'Неожиданная ошибка: {str(e)}'
                })
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
        "$ук@?",
        "Ты дебил и идиот!",
        "Спасибо за помощь",
        "Убийца и маньяк!",
        "сукаблядь",
        "Иди нахуй!",
        "ебааааать какой дебил!!!",
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
