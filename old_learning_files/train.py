import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, GlobalMaxPooling1D, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import os

class F1ScoreCallback(Callback):
    """Кастомный callback для вычисления F1-score"""
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        self.best_f1 = 0.0
        self.best_accuracy = 0.0
        self.best_val_accuracy = 0.0
        self.best_loss = float('inf')
        self.best_val_loss = float('inf')
        
    def on_epoch_end(self, epoch, logs=None):
        # Получаем предсказания для валидационных данных
        X_val, y_val = self.validation_data
        y_pred_proba = self.model.predict(X_val, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Вычисляем F1-score
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        # Обновляем лучшие метрики
        current_accuracy = logs.get('accuracy', 0)
        current_val_accuracy = logs.get('val_accuracy', 0)
        current_loss = logs.get('loss', float('inf'))
        current_val_loss = logs.get('val_loss', float('inf'))
        
        improved = False
        
        if f1 > self.best_f1:
            self.best_f1 = f1
            improved = True
            
        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy
            improved = True
            
        if current_val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = current_val_accuracy
            improved = True
            
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            improved = True
            
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            improved = True
        
        # Добавляем F1-score в логи
        logs['f1_score'] = f1
        logs['best_f1'] = self.best_f1
        logs['best_accuracy'] = self.best_accuracy
        logs['best_val_accuracy'] = self.best_val_accuracy
        logs['best_loss'] = self.best_loss
        logs['best_val_loss'] = self.best_val_loss
        
        # Сохраняем модель если она улучшилась
        if improved:
            self.model.save('model_tf_long.h5')
            with open('tokenizer.pkl', 'wb') as f:
                pickle.dump(self.tokenizer, f)
            print(f"\n🎉 Новая лучшая модель сохранена! F1: {f1:.4f}")
        
        print(f"\n📊 Эпоха {epoch + 1} - Метрики:")
        print(f"   Accuracy: {current_accuracy:.4f} (лучшая: {self.best_accuracy:.4f})")
        print(f"   Val Accuracy: {current_val_accuracy:.4f} (лучшая: {self.best_val_accuracy:.4f})")
        print(f"   Loss: {current_loss:.4f} (лучшая: {self.best_loss:.4f})")
        print(f"   Val Loss: {current_val_loss:.4f} (лучшая: {self.best_val_loss:.4f})")
        print(f"   F1-Score: {f1:.4f} (лучшая: {self.best_f1:.4f})")

class ToxicityClassifier:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.max_features = 50000  # Увеличиваем словарь
        self.max_length = 300      # Увеличиваем длину последовательности
        self.embedding_dim = 256  # Увеличиваем размерность эмбеддингов
        
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
    
    def load_data(self, csv_path):
        """Загрузка и предобработка данных"""
        print("Загружаем данные...")
        df = pd.read_csv(csv_path)
        
        # Проверяем структуру данных
        print(f"Колонки в датасете: {list(df.columns)}")
        
        # Проверяем наличие нужных колонок
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("В датасете должны быть колонки 'text' и 'label'")
        
        print(f"Загружено {len(df)} записей")
        print(f"Распределение классов:")
        print(df['label'].value_counts())
        
        # Предобработка текста
        print("Предобрабатываем текст...")
        df['text_processed'] = df['text'].apply(self.preprocess_text)
        
        # Удаляем пустые тексты
        df = df[df['text_processed'].str.len() > 0]
        
        print(f"После предобработки осталось {len(df)} записей")
        
        return df
    
    def prepare_data(self, df):
        """Подготовка данных для обучения"""
        print("Подготавливаем данные для обучения...")
        
        # Разделяем на признаки и метки
        X = df['text_processed'].values
        y = df['label'].values
        
        # Создаем токенизатор
        self.tokenizer = Tokenizer(num_words=self.max_features, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(X)
        
        # Преобразуем тексты в последовательности
        X_sequences = self.tokenizer.texts_to_sequences(X)
        X_padded = pad_sequences(X_sequences, maxlen=self.max_length, padding='post', truncating='post')
        
        # Разделяем на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X_padded, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Размер обучающей выборки: {X_train.shape}")
        print(f"Размер тестовой выборки: {X_test.shape}")
        print(f"Размер словаря: {len(self.tokenizer.word_index)}")
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, vocab_size):
        """Создание архитектуры нейронной сети с ~20M параметров"""
        print("Создаем модель...")
        print(f"Размер словаря: {vocab_size}")
        print(f"Максимальная длина последовательности: {self.max_length}")
        print(f"Размерность эмбеддингов: {self.embedding_dim}")
        
        model = Sequential([
            # Эмбеддинг слой - это самый большой слой по параметрам
            Embedding(vocab_size, self.embedding_dim, input_length=self.max_length, name='embedding'),
            
            # Первый двунаправленный LSTM
            Bidirectional(LSTM(512, return_sequences=True, dropout=0.3, recurrent_dropout=0.3), name='bidirectional_lstm_1'),
            
            # Второй двунаправленный LSTM
            Bidirectional(LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.3), name='bidirectional_lstm_2'),
            
            # Третий LSTM
            LSTM(128, dropout=0.3, recurrent_dropout=0.3, name='lstm_3'),
            
            # Полносвязные слои
            Dense(1024, activation='relu', name='dense_1'),
            Dropout(0.5),
            
            Dense(512, activation='relu', name='dense_2'),
            Dropout(0.4),
            
            Dense(256, activation='relu', name='dense_3'),
            Dropout(0.3),
            
            Dense(128, activation='relu', name='dense_4'),
            Dropout(0.2),
            
            # Выходной слой
            Dense(1, activation='sigmoid', name='output')
        ])
        
        # Компилируем модель с оптимизатором Adam
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("Архитектура модели:")
        model.summary()
        
        # Подсчитываем общее количество параметров
        # Сначала строим модель с примером входных данных
        print("\n🔧 Строим модель...")
        dummy_input = tf.random.normal((1, self.max_length))
        _ = model(dummy_input)
        
        total_params = model.count_params()
        print(f"\n🔢 Общее количество параметров: {total_params:,}")
        print(f"📊 Примерно {total_params/1_000_000:.1f}M параметров")
        
        # Детальная диагностика по слоям
        print("\n📋 Детальная информация по слоям:")
        for i, layer in enumerate(model.layers):
            layer_params = layer.count_params()
            print(f"  Слой {i+1} ({layer.name}): {layer_params:,} параметров")
            if layer_params == 0:
                print(f"    ⚠️  Слой {layer.name} имеет 0 параметров!")
        
        # Проверяем, что параметры действительно есть
        if total_params == 0:
            print("\n⚠️  ВНИМАНИЕ: Количество параметров равно 0! Возможна проблема с архитектурой модели.")
            print("🔍 Возможные причины:")
            print("   - Слишком маленький размер словаря")
            print("   - Проблемы с инициализацией слоев")
            print("   - Неправильная архитектура модели")
        elif total_params < 1_000_000:
            print("\n⚠️  ВНИМАНИЕ: Количество параметров меньше 1M. Модель может быть слишком простой.")
        else:
            print("\n✅ Модель имеет достаточное количество параметров для сложных задач.")
        
        return model
    
    def train(self, X_train, y_train, X_test, y_test):
        """Обучение модели"""
        print("Начинаем обучение...")
        
        vocab_size = len(self.tokenizer.word_index) + 1
        self.model = self.build_model(vocab_size)
        
        # Создаем callback для F1-score
        f1_callback = F1ScoreCallback((X_test, y_test))
        f1_callback.tokenizer = self.tokenizer
        
        # Callbacks для улучшения обучения
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            'model_tf_long.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        # Обучение
        print("\n🚀 Начинаем обучение модели...")
        history = self.model.fit(
            X_train, y_train,
            epochs=20,  # Увеличиваем количество эпох
            batch_size=64,  # Увеличиваем размер батча
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, model_checkpoint, f1_callback],
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """Оценка модели"""
        print("Оцениваем модель...")
        
        # Предсказания
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Метрики
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Точность: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        print("\nОтчет по классификации:")
        print(classification_report(y_test, y_pred, target_names=['Нетоксично', 'Токсично']))
        
        # Матрица ошибок
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Нетоксично', 'Токсично'],
                    yticklabels=['Нетоксично', 'Токсично'])
        plt.title('Матрица ошибок')
        plt.ylabel('Истинные значения')
        plt.xlabel('Предсказанные значения')
        plt.savefig('confusion_matrix_tf_long.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy, y_pred
    
    def predict_toxicity(self, text):
        """Предсказание токсичности для нового текста"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Модель не обучена! Сначала вызовите train()")
        
        # Предобработка текста
        processed_text = self.preprocess_text(text)
        
        # Токенизация
        sequence = self.tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_length, padding='post', truncating='post')
        
        # Предсказание
        probability = self.model.predict(padded_sequence, verbose=0)[0][0]
        prediction = 1 if probability > 0.5 else 0
        
        return {
            'text': text,
            'processed_text': processed_text,
            'is_toxic': bool(prediction),
            'toxicity_probability': float(probability),
            'confidence': float(max(probability, 1 - probability))
        }
    
    def save_model(self):
        """Сохранение модели"""
        if self.model is None:
            raise ValueError("Модель не обучена!")
        
        # Сохраняем модель Keras
        self.model.save('model_tf_long.h5')
        
        # Сохраняем токенизатор
        with open('tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        print(f"✅ Модель сохранена: model_tf_long.h5")
        print(f"✅ Токенизатор сохранен: tokenizer.pkl")
    
    def load_model(self, model_path='model_tf_long.h5', tokenizer_path='tokenizer.pkl'):
        """Загрузка модели"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            print("Модель успешно загружена!")
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")

def main():
    """Основная функция для обучения модели"""
    print("🎯 Обучение модели классификации токсичности")
    print("=" * 50)
    
    # Создаем классификатор
    classifier = ToxicityClassifier()
    
    # Загружаем данные
    df = classifier.load_data('C:/Andrey/Study/5 сем/HAKATON/HAKATON/dataset/train_final_complete.csv')
    
    # Подготавливаем данные
    X_train, X_test, y_train, y_test = classifier.prepare_data(df)
    
    # Обучаем модель
    history = classifier.train(X_train, y_train, X_test, y_test)
    
    # Оцениваем модель
    accuracy, predictions = classifier.evaluate(X_test, y_test)
    
    # Сохраняем модель
    classifier.save_model()
    
    # Тестируем на примерах
    print("\n" + "="*50)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ НА ПРИМЕРАХ")
    print("="*50)
    
    test_texts = [
        "Привет, как дела?",
        "Ты дебил и идиот!",
        "Спасибо за помощь",
        "Убийца и маньяк!",
        "Хорошая погода сегодня"
    ]
    
    for text in test_texts:
        result = classifier.predict_toxicity(text)
        print(f"\nТекст: '{result['text']}'")
        print(f"Токсично: {result['is_toxic']}")
        print(f"Вероятность токсичности: {result['toxicity_probability']:.3f}")
        print(f"Уверенность: {result['confidence']:.3f}")

if __name__ == "__main__":
    main()