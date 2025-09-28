import pandas as pd
import numpy as np
import time
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import psutil
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# TensorFlow с максимальными оптимизациями для RTX 4060
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

# Кастомная F1-метрика для мониторинга во время обучения
def f1_metric(y_true, y_pred):
    """Кастомная F1-метрика для TensorFlow"""
    y_pred = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    return 2 * precision * recall / (precision + recall + K.epsilon())

class RTXModelTrainer:
    """Максимально оптимизированный тренер модели для RTX 4060"""

    def __init__(self):
        self.setup_gpu()
        self.setup_mixed_precision()
        self.vectorizer = None
        self.model = None
        self.history = None
        self.results = {}

    def setup_gpu(self):
        """Настройка GPU для максимальной производительности RTX 4060"""
        print("🚀 Настройка вычислительных ресурсов...")

        # Настройка CPU для максимальной производительности
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['OMP_NUM_THREADS'] = str(psutil.cpu_count())
        os.environ['TF_NUM_INTEROP_THREADS'] = str(psutil.cpu_count())
        os.environ['TF_NUM_INTRAOP_THREADS'] = str(psutil.cpu_count())

        # Проверка доступности GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Включение роста памяти GPU
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                print(f"✅ Обнаружено GPU: {len(gpus)}")
                for i, gpu in enumerate(gpus):
                    try:
                        details = tf.config.experimental.get_device_details(gpu)
                        print(f"   GPU {i}: {details.get('device_name', 'RTX 4060')}")
                    except:
                        print(f"   GPU {i}: RTX 4060 (GPU активен)")

            except RuntimeError as e:
                print(f"⚠️ Ошибка настройки GPU: {e}")
                print("   Переключаюсь на оптимизированный CPU режим...")
                gpus = []
        else:
            print("⚠️ GPU не обнаружен TensorFlow, используется оптимизированный CPU")

        # Настройка CPU оптимизаций
        tf.config.threading.set_inter_op_parallelism_threads(psutil.cpu_count())
        tf.config.threading.set_intra_op_parallelism_threads(psutil.cpu_count())

        print(f"✅ CPU ядер: {psutil.cpu_count()}")
        print(f"✅ RAM: {psutil.virtual_memory().total / (1024**3):.1f} ГБ")

        return len(gpus) > 0

    def setup_mixed_precision(self):
        """Настройка смешанной точности для ускорения на RTX 4060"""
        try:
            set_global_policy('mixed_float16')
            print("✅ Включена смешанная точность (mixed_float16)")
        except Exception as e:
            print(f"⚠️ Не удалось включить смешанную точность: {e}")

    def load_and_prepare_data(self, data_path):
        """Загрузка и подготовка данных"""
        print("📊 Загрузка данных...")

        # Загрузка данных
        df = pd.read_csv(data_path)
        print(f"   Размер данных: {df.shape}")
        print(f"   Распределение классов: {df['label'].value_counts().to_dict()}")

        # Проверка на пропуски
        if df['text'].isnull().sum() > 0:
            print(f"   Удаление {df['text'].isnull().sum()} строк с пропусками")
            df = df.dropna(subset=['text'])

        X = df['text'].astype(str)
        y = df['label'].astype(int)

        return X, y

    def create_tfidf_features(self, X_train, X_test, max_features=8000):  # Уменьшено с 15000 до 8000
        """Создание TF-IDF признаков с оптимальными параметрами для экономии памяти"""
        print("🔤 Создание TF-IDF признаков...")

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,  # Уменьшено для экономии памяти
            ngram_range=(1, 2),  # Только униграммы и биграммы
            min_df=5,  # Увеличено для фильтрации редких слов
            max_df=0.85,  # Уменьшено для исключения частых слов
            sublinear_tf=True,
            use_idf=True,
            smooth_idf=True,
            stop_words=None,  # Данные уже предобработаны
            dtype=np.float32  # Используем float32 вместо float64
        )

        print("   Обучение векторизатора...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)

        print(f"   Размер признакового пространства: {X_train_tfidf.shape[1]}")

        # Исправляем расчет размера памяти для разреженных матриц
        memory_size = (X_train_tfidf.data.nbytes + X_train_tfidf.indices.nbytes + X_train_tfidf.indptr.nbytes) / (1024**3)
        print(f"   Размер в памяти: {memory_size:.2f} ГБ")

        # Возвращаем разреженные матрицы для экономии памяти
        return X_train_tfidf, X_test_tfidf

    def apply_smote(self, X_train, y_train):
        """Применение SMOTE для балансировки классов с разреженными матрицами"""
        print("⚖️ Применение SMOTE...")

        unique, counts = np.unique(y_train, return_counts=True)
        print(f"   До SMOTE: {dict(zip(unique, counts))}")

        # Преобразуем разреженную матрицу в плотную только для SMOTE
        print("   Преобразование в плотный формат для SMOTE...")
        X_train_dense = X_train.toarray() if hasattr(X_train, 'toarray') else X_train

        # Настройки SMOTE для оптимальной работы
        smote = SMOTE(
            sampling_strategy='auto',
            k_neighbors=3,  # Уменьшено для экономии памяти
            random_state=42
        )

        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_dense, y_train)

        unique, counts = np.unique(y_train_balanced, return_counts=True)
        print(f"   После SMOTE: {dict(zip(unique, counts))}")

        return X_train_balanced, y_train_balanced

    def apply_smote_batched(self, X_train, y_train, batch_size=10000):
        """Применение SMOTE с батчевой обработкой для экономии памяти"""
        print("⚖️ Применение SMOTE с батчевой обработкой...")

        unique, counts = np.unique(y_train, return_counts=True)
        print(f"   До SMOTE: {dict(zip(unique, counts))}")

        # Если данных немного, используем обычный SMOTE
        if X_train.shape[0] <= 20000:
            print("   Применение обычного SMOTE...")
            X_train_dense = X_train.toarray()

            smote = SMOTE(
                sampling_strategy='auto',
                k_neighbors=3,
                random_state=42
            )

            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_dense, y_train)
        else:
            # Для больших данных используем подвыборку и дублирование
            print("   Применение оптимизированной балансировки...")

            # Получаем индексы для каждого класса
            class_0_indices = np.where(y_train == 0)[0]
            class_1_indices = np.where(y_train == 1)[0]

            # Определяем размер меньшего класса
            min_class_size = min(len(class_0_indices), len(class_1_indices))
            target_size = min_class_size * 2  # Увеличиваем в 2 раза

            print(f"   Целевой размер для каждого класса: {target_size}")

            # Случайная выборка и дублирование для балансировки
            np.random.seed(42)

            if len(class_0_indices) < target_size:
                # Дублируем класс 0
                additional_needed = target_size - len(class_0_indices)
                additional_indices = np.random.choice(class_0_indices, additional_needed, replace=True)
                balanced_0_indices = np.concatenate([class_0_indices, additional_indices])
            else:
                balanced_0_indices = np.random.choice(class_0_indices, target_size, replace=False)

            if len(class_1_indices) < target_size:
                # Дублируем класс 1
                additional_needed = target_size - len(class_1_indices)
                additional_indices = np.random.choice(class_1_indices, additional_needed, replace=True)
                balanced_1_indices = np.concatenate([class_1_indices, additional_indices])
            else:
                balanced_1_indices = np.random.choice(class_1_indices, target_size, replace=False)

            # Объединяем индексы
            all_indices = np.concatenate([balanced_0_indices, balanced_1_indices])
        return X_train_balanced, y_train_balanced

    def create_model(self, input_dim):
        """Создание оптимизированной нейронной сети для RTX 4060"""
        print("🧠 Создание нейронной сети...")

        model = Sequential([
            Input(shape=(input_dim,), name='input_layer'),

            # Первый блок
            Dense(1024, activation='relu', kernel_regularizer=l2(0.001), name='dense_1'),
            BatchNormalization(name='bn_1'),
            Dropout(0.3, name='dropout_1'),

            # Второй блок
            Dense(512, activation='relu', kernel_regularizer=l2(0.001), name='dense_2'),
            BatchNormalization(name='bn_2'),
            Dropout(0.4, name='dropout_2'),

            # Третий блок
            Dense(256, activation='relu', kernel_regularizer=l2(0.001), name='dense_3'),
            BatchNormalization(name='bn_3'),
            Dropout(0.3, name='dropout_3'),

            # Четвертый блок
            Dense(128, activation='relu', kernel_regularizer=l2(0.001), name='dense_4'),
            BatchNormalization(name='bn_4'),
            Dropout(0.2, name='dropout_4'),

            # Выходной слой
            Dense(1, activation='sigmoid', dtype='float32', name='output')
        ])

        # Оптимизатор с оптимальными параметрами для RTX 4060
        optimizer = Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            clipnorm=1.0
        )

        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', f1_metric]
        )

        print("   Архитектура модели:")
        model.summary()

        return model

    def create_callbacks(self):
        """Создание callback'ов для оптимального обучения"""
        callbacks = [
            EarlyStopping(
                monitor='val_f1_metric',  # Мониторим валидационный F1-score
                patience=15,
                restore_best_weights=True,
                verbose=1,
                mode='max'  # Максимизируем F1-score
            ),
            ReduceLROnPlateau(
                monitor='val_f1_metric',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1,
                mode='max'
            ),
            ModelCheckpoint(
                filepath='model_tf_31_best.h5',
                monitor='val_f1_metric',  # Сохраняем лучшую модель по F1-score
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
                mode='max'
            )
        ]

        return callbacks

    def train_model(self, X_train, y_train, X_val, y_val):
        """Обучение модели с оптимальными параметрами"""
        print("🏋️ Начало обучения модели...")

        # Создание модели
        self.model = self.create_model(X_train.shape[1])

        # Callback'и
        callbacks = self.create_callbacks()

        start_time = time.time()

        # Обучение с оптимальными параметрами
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=512,  # Оптимальный размер батча
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )

        training_time = time.time() - start_time
        print(f"⏱️ Время обучения: {training_time:.2f} секунд")

        return training_time

    def evaluate_model(self, X_test, y_test):
        """Оценка модели"""
        print("📊 Оценка модели...")

        # Предсказания
        y_pred_proba = self.model.predict(X_test, batch_size=1024, verbose=1)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        # Метрики
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        self.results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

        print(f"✅ Точность: {accuracy:.4f}")
        print(f"✅ F1-score: {f1:.4f}")
        print(f"✅ ROC-AUC: {roc_auc:.4f}")

        return self.results

    def save_model_and_results(self):
        """Сохранение модели и результатов"""
        print("💾 Сохранение результатов...")

        # Сохранение модели
        self.model.save('model_tf_31.h5')
        print("   ✅ Модель сохранена: model_tf_31.h5")

        # Сохранение векторизатора
        with open('tokenizer_tf_31.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print("   ✅ Векторизатор сохранен: tokenizer_tf_31.pkl")

        # Сохранение результатов
        with open('results_tf_31.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        print("   ✅ Результаты сохранены: results_tf_31.pkl")

    def plot_training_history(self):
        """Построение графиков обучения"""
        print("📈 Создание графиков...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # Увеличиваем до 2x3 для F1-score

        # Точность
        axes[0, 0].plot(self.history.history['accuracy'], label='Обучение', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Валидация', linewidth=2)
        axes[0, 0].set_title('Точность модели', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Эпоха')
        axes[0, 0].set_ylabel('Точность')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Функция потерь
        axes[0, 1].plot(self.history.history['loss'], label='Обучение', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], label='Валидация', linewidth=2)
        axes[0, 1].set_title('Функция потерь', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Эпоха')
        axes[0, 1].set_ylabel('Потери')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # F1-score - основной график для мониторинга
        if 'f1_metric' in self.history.history:
            axes[0, 2].plot(self.history.history['f1_metric'], label='F1 Обучение', linewidth=2, color='green')
            axes[0, 2].plot(self.history.history['val_f1_metric'], label='F1 Валидация', linewidth=2, color='red')
            axes[0, 2].set_title('F1-Score', fontsize=14, fontweight='bold')
            axes[0, 2].set_xlabel('Эпоха')
            axes[0, 2].set_ylabel('F1-Score')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)

            # Добавляем аннотацию с максимальным F1-score
            max_val_f1 = max(self.history.history['val_f1_metric'])
            max_epoch = self.history.history['val_f1_metric'].index(max_val_f1)
            axes[0, 2].annotate(f'Max F1: {max_val_f1:.4f}\n(Эпоха {max_epoch+1})',
                              xy=(max_epoch, max_val_f1),
                              xytext=(10, 10),
                              textcoords='offset points',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                              arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        # Precision
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Обучение', linewidth=2)
            axes[1, 0].plot(self.history.history['val_precision'], label='Валидация', linewidth=2)
            axes[1, 0].set_title('Precision', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Эпоха')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Recall
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Обучение', linewidth=2)
            axes[1, 1].plot(self.history.history['val_recall'], label='Валидация', linewidth=2)
            axes[1, 1].set_title('Recall', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Эпоха')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        # Комбинированный график всех метрик на валидации
        axes[1, 2].plot(self.history.history['val_accuracy'], label='Accuracy', linewidth=2)
        if 'val_precision' in self.history.history:
            axes[1, 2].plot(self.history.history['val_precision'], label='Precision', linewidth=2)
        if 'val_recall' in self.history.history:
            axes[1, 2].plot(self.history.history['val_recall'], label='Recall', linewidth=2)
        if 'val_f1_metric' in self.history.history:
            axes[1, 2].plot(self.history.history['val_f1_metric'], label='F1-Score', linewidth=3, color='red')
        axes[1, 2].set_title('Все метрики (Валидация)', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Эпоха')
        axes[1, 2].set_ylabel('Значение метрики')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_history_tf_31.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ График обучения сохранен: training_history_tf_31.png")

        # Выводим статистику F1-score
        if 'f1_metric' in self.history.history:
            print(f"   📊 Максимальный F1-score на валидации: {max(self.history.history['val_f1_metric']):.4f}")
            print(f"   📊 Финальный F1-score на валидации: {self.history.history['val_f1_metric'][-1]:.4f}")
            print(f"   📊 Максимальный F1-score на обучении: {max(self.history.history['f1_metric']):.4f}")

    def plot_confusion_matrix(self):
        """Построение матрицы ошибок"""
        plt.figure(figsize=(8, 6))
        cm = self.results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Матрица ошибок')
        plt.xlabel('Предсказанные значения')
        plt.ylabel('Истинные значения')
        plt.savefig('confusion_matrix_tf_31.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ Матрица ошибок сохранена: confusion_matrix_tf_31.png")

def main():
    """Основная функция обучения"""
    print("🚀 Запуск обучения нейронной сети на RTX 4060")
    print("=" * 60)

    # Создание тренера
    trainer = RTXModelTrainer()

    # Загрузка данных
    X, y = trainer.load_and_prepare_data('C:/Andrey/Study/5 сем/HAKATON/HAKATON/dataset/train_final_complete.csv')

    # Разделение данных
    print("🔄 Разделение данных...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"   Обучение: {len(X_train)}")
    print(f"   Валидация: {len(X_val)}")
    print(f"   Тест: {len(X_test)}")

    # Создание TF-IDF признаков
    X_train_tfidf, X_val_tfidf = trainer.create_tfidf_features(X_train, X_val)
    X_test_tfidf = trainer.vectorizer.transform(X_test)

    # Преобразование валидационных данных в плотный формат
    print("🔄 Преобразование валидационных данных...")
    X_val_dense = X_val_tfidf.toarray()

    # Применение SMOTE
    X_train_balanced, y_train_balanced = trainer.apply_smote(X_train_tfidf, y_train)

    # Обучение модели
    training_time = trainer.train_model(X_train_balanced, y_train_balanced, X_val_dense, y_val)

    # Преобразование тестовых данных в плотный формат для оценки
    print("🔄 Преобразование тестовых данных...")
    X_test_dense = X_test_tfidf.toarray()

    # Оценка модели
    results = trainer.evaluate_model(X_test_dense, y_test)

    # Сохранение результатов
    trainer.save_model_and_results()

    # Построение графиков
    trainer.plot_training_history()
    trainer.plot_confusion_matrix()

    # Финальный отчет
    print("\n" + "=" * 60)
    print("🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 60)
    print(f"⏱️ Время обучения: {training_time:.2f} секунд")
    print(f"🎯 Финальная точность: {results['accuracy']:.4f}")
    print(f"📊 F1-score: {results['f1_score']:.4f}")
    print(f"📈 ROC-AUC: {results['roc_auc']:.4f}")
    print("\n📁 Сохраненные файлы:")
    print("   • model_tf_31.h5 - TensorFlow модель")
    print("   • tokenizer_tf_31.pkl - TF-IDF векторизатор")
    print("   • results_tf_31.pkl - Метрики обучения")
    print("   • confusion_matrix_tf_31.png - Матрица ошибок")
    print("   • training_history_tf_31.png - График обучения")
    print("=" * 60)

if __name__ == "__main__":
    main()
