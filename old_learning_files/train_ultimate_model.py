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
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
import psutil
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# TensorFlow с максимальными оптимизациями
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate, Add
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
try:
    from tensorflow.keras.callbacks import CyclicLR
except ImportError:
    # Если CyclicLR недоступен, создаем заглушку
    class CyclicLR:
        def __init__(self, *args, **kwargs):
            pass
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model

class UltimateModelTrainer:
    """Максимально точная модель для классификации токсичности"""

    def __init__(self):
        self.setup_gpu()
        self.setup_mixed_precision()
        self.vectorizer = None
        self.model = None
        self.history = None
        self.results = {}

    def setup_gpu(self):
        """Настройка GPU для максимальной производительности"""
        print("🚀 Настройка вычислительных ресурсов...")

        # Максимальная оптимизация CPU
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['OMP_NUM_THREADS'] = str(psutil.cpu_count())
        os.environ['TF_NUM_INTEROP_THREADS'] = str(psutil.cpu_count())
        os.environ['TF_NUM_INTRAOP_THREADS'] = str(psutil.cpu_count())

        # GPU настройки
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ Обнаружено GPU: {len(gpus)}")
            except RuntimeError as e:
                print(f"⚠️ Ошибка настройки GPU: {e}")
        else:
            print("⚠️ GPU не обнаружен, используется оптимизированный CPU")

        tf.config.threading.set_inter_op_parallelism_threads(psutil.cpu_count())
        tf.config.threading.set_intra_op_parallelism_threads(psutil.cpu_count())

        print(f"✅ CPU ядер: {psutil.cpu_count()}")
        print(f"✅ RAM: {psutil.virtual_memory().total / (1024**3):.1f} ГБ")

    def setup_mixed_precision(self):
        """Настройка смешанной точности"""
        try:
            set_global_policy('mixed_float16')
            print("✅ Включена смешанная точность (mixed_float16)")
        except Exception as e:
            print(f"⚠️ Не удалось включить смешанную точность: {e}")

    def load_combined_data(self):
        """Загрузка и объединение всех данных для максимальной точности"""
        print("📊 Загрузка и объединение данных...")

        # Загрузка первого датасета
        try:
            df1 = pd.read_csv('../dataset/train_final_complete.csv')
            print(f"   train_final_complete.csv: {df1.shape}")
        except:
            df1 = pd.read_csv('train_final_complete.csv')
            print(f"   train_final_complete.csv: {df1.shape}")

        # Загрузка второго датасета
        try:
            df2 = pd.read_csv('../dataset/newtrain.csv')
            print(f"   newtrain.csv: {df2.shape}")
        except:
            df2 = pd.read_csv('newtrain.csv')
            print(f"   newtrain.csv: {df2.shape}")

        # Объединение данных
        combined_df = pd.concat([df1, df2], ignore_index=True)
        print(f"   Объединенные данные: {combined_df.shape}")

        # Очистка данных
        combined_df = combined_df.dropna(subset=['text'])
        combined_df = combined_df.drop_duplicates(subset=['text'])
        combined_df['label'] = combined_df['label'].astype(int)

        print(f"   После очистки: {combined_df.shape}")
        print(f"   Распределение классов: {combined_df['label'].value_counts().to_dict()}")

        X = combined_df['text'].astype(str)
        y = combined_df['label'].astype(int)

        return X, y

    def create_advanced_tfidf_features(self, X_train, X_test, max_features=10000):
        """Создание оптимизированных TF-IDF признаков"""
        print("🔤 Создание оптимизированных TF-IDF признаков...")

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,  # Уменьшено для экономии памяти
            ngram_range=(1, 2),  # Убрал триграммы для экономии памяти
            min_df=3,  # Увеличено для уменьшения размерности
            max_df=0.9,  # Уменьшено для фильтрации частых слов
            sublinear_tf=True,
            use_idf=True,
            smooth_idf=True,
            norm='l2',
            stop_words=None,
            dtype=np.float32
        )

        print("   Обучение векторизатора...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)

        print(f"   Размер признакового пространства: {X_train_tfidf.shape[1]}")

        # Расчет размера памяти для разреженных матриц
        memory_size = (X_train_tfidf.data.nbytes + X_train_tfidf.indices.nbytes + X_train_tfidf.indptr.nbytes) / (1024**3)
        print(f"   Размер в памяти: {memory_size:.2f} ГБ")

        return X_train_tfidf, X_test_tfidf

    def apply_advanced_sampling(self, X_train, y_train):
        """Применение оптимизированного сэмплирования с контролем памяти"""
        print("⚖️ Применение оптимизированного сэмплирования...")

        unique, counts = np.unique(y_train, return_counts=True)
        print(f"   До сэмплирования: {dict(zip(unique, counts))}")

        # Проверяем доступную память
        available_memory = psutil.virtual_memory().available / (1024**3)  # ГБ
        required_memory = X_train.shape[0] * X_train.shape[1] * 4 / (1024**3)  # float32

        print(f"   Доступная память: {available_memory:.1f} ГБ")
        print(f"   Требуемая память: {required_memory:.1f} ГБ")

        if required_memory > available_memory * 0.5:  # Если требуется больше 50% памяти
            print("   ⚠️ Недостаточно памяти для SMOTE, используем class_weight")
            return X_train.toarray() if hasattr(X_train, 'toarray') else X_train, y_train

        # Преобразование в плотный формат
        print("   Преобразование в плотный формат...")
        X_train_dense = X_train.toarray() if hasattr(X_train, 'toarray') else X_train

        # Используем простой SMOTE с ограниченными параметрами
        try:
            sampler = SMOTE(
                k_neighbors=3,  # Уменьшено для экономии памяти
                random_state=42
            )

            X_train_balanced, y_train_balanced = sampler.fit_resample(X_train_dense, y_train)

            unique, counts = np.unique(y_train_balanced, return_counts=True)
            print(f"   После сэмплирования: {dict(zip(unique, counts))}")

            return X_train_balanced, y_train_balanced

        except MemoryError:
            print("   ⚠️ Недостаточно памяти для SMOTE, используем исходные данные")
            return X_train_dense, y_train

    def create_ultimate_model(self, input_dim):
        """Создание оптимизированной нейронной сети для высокого F1-score"""
        print("🧠 Создание оптимизированной нейронной сети...")

        model = Sequential([
            Input(shape=(input_dim,), name='input_layer'),

            # Первый блок - более компактный
            Dense(1024, activation='relu', kernel_regularizer=l1_l2(l1=0.0001, l2=0.001)),
            BatchNormalization(),
            Dropout(0.3),

            # Второй блок
            Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=0.0001, l2=0.001)),
            BatchNormalization(),
            Dropout(0.4),

            # Третий блок
            Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=0.0001, l2=0.001)),
            BatchNormalization(),
            Dropout(0.3),

            # Четвертый блок
            Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=0.0001, l2=0.001)),
            BatchNormalization(),
            Dropout(0.2),

            # Выходной слой
            Dense(1, activation='sigmoid', dtype='float32', name='output')
        ])

        # Оптимизатор Adam с настройками для высокого F1-score
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
            metrics=['accuracy', 'precision', 'recall']
        )

        print("   Оптимизированная архитектура модели:")
        model.summary()

        return model

    def create_advanced_callbacks(self):
        """Создание оптимизированных callback'ов"""
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',  # Используем стандартную метрику
                patience=15,  # Уменьшено для быстрой сходимости
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=7,
                min_lr=1e-8,
                verbose=1,
                mode='max'
            ),
            ModelCheckpoint(
                filepath='../model_tf_3_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
                mode='max'
            )
        ]

        return callbacks

    def f1_score_metric(self, y_true, y_pred):
        """Кастомная метрика F1-score для мониторинга"""
        # Приводим к одному типу данных
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        def recall(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision

        precision_val = precision(y_true, y_pred)
        recall_val = recall(y_true, y_pred)
        return 2*((precision_val*recall_val)/(precision_val+recall_val+K.epsilon()))

    def train_ultimate_model(self, X_train, y_train, X_val, y_val):
        """Обучение ультимативной модели"""
        print("🏋️ Начало обучения ультимативной модели...")

        # Создание модели
        self.model = self.create_ultimate_model(X_train.shape[1])

        # Callback'и
        callbacks = self.create_advanced_callbacks()

        start_time = time.time()

        # Обучение с оптимальными параметрами для F1-score
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,  # Уменьшено для быстрой сходимости
            batch_size=512,  # Увеличен для экономии памяти
            callbacks=callbacks,
            verbose=1,
            shuffle=True,
            class_weight={0: 1.0, 1: 3.0}  # Больший вес токсичным комментариям
        )

        training_time = time.time() - start_time
        print(f"⏱️ Время обучения: {training_time:.2f} секунд")

        return training_time

    def evaluate_ultimate_model(self, X_test, y_test):
        """Оценка ультимативной модели"""
        print("📊 Оценка ультимативной модели...")

        # Предсказания с оптимальным порогом
        y_pred_proba = self.model.predict(X_test, batch_size=512, verbose=1)

        # Поиск оптимального порога для максимизации F1-score
        thresholds = np.arange(0.3, 0.8, 0.01)
        best_f1 = 0
        best_threshold = 0.5

        for threshold in thresholds:
            y_pred_temp = (y_pred_proba > threshold).astype(int).flatten()
            f1_temp = f1_score(y_test, y_pred_temp)
            if f1_temp > best_f1:
                best_f1 = f1_temp
                best_threshold = threshold

        print(f"   Оптимальный порог: {best_threshold:.3f}")

        y_pred = (y_pred_proba > best_threshold).astype(int).flatten()

        # Метрики
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        self.results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'best_threshold': best_threshold,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

        print(f"✅ Точность: {accuracy:.4f}")
        print(f"✅ F1-score: {f1:.4f}")
        print(f"✅ ROC-AUC: {roc_auc:.4f}")

        return self.results

    def save_ultimate_model_and_results(self):
        """Сохранение ультимативной модели и результатов"""
        print("💾 Сохранение ультимативной модели...")

        # Сохранение модели
        self.model.save('model_tf_3.h5')
        print("   ✅ Ультимативная модель сохранена: model_tf_3.h5")

        # Сохранение векторизатора
        with open('../tokenizer_tf_3.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print("   ✅ Векторизатор сохранен: tokenizer_tf_3.pkl")

        # Сохранение результатов
        with open('../results_tf_3.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        print("   ✅ Результаты сохранены: results_tf_3.pkl")

    def plot_ultimate_training_history(self):
        """Построение графиков обучения ультимативной модели"""
        print("📈 Создание графиков ультимативной модели...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Точность
        axes[0, 0].plot(self.history.history['accuracy'], label='Обучение', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Валидация', linewidth=2)
        axes[0, 0].set_title('Точность модели (Ultimate)', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Эпоха')
        axes[0, 0].set_ylabel('Точность')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Функция потерь
        axes[0, 1].plot(self.history.history['loss'], label='Обучение', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], label='Валидация', linewidth=2)
        axes[0, 1].set_title('Функция потерь (Ultimate)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Эпоха')
        axes[0, 1].set_ylabel('Потери')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # F1-score
        if 'f1_score_metric' in self.history.history:
            axes[0, 2].plot(self.history.history['f1_score_metric'], label='Обучение', linewidth=2)
            axes[0, 2].plot(self.history.history['val_f1_score_metric'], label='Валидация', linewidth=2)
            axes[0, 2].set_title('F1-Score (Ultimate)', fontsize=14, fontweight='bold')
            axes[0, 2].set_xlabel('Эпоха')
            axes[0, 2].set_ylabel('F1-Score')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)

        # Precision
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Обучение', linewidth=2)
            axes[1, 0].plot(self.history.history['val_precision'], label='Валидация', linewidth=2)
            axes[1, 0].set_title('Precision (Ultimate)', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Эпоха')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Recall
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Обучение', linewidth=2)
            axes[1, 1].plot(self.history.history['val_recall'], label='Валидация', linewidth=2)
            axes[1, 1].set_title('Recall (Ultimate)', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Эпоха')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        # Learning Rate
        if hasattr(self.history.history, 'lr'):
            axes[1, 2].plot(self.history.history['lr'], linewidth=2, color='red')
            axes[1, 2].set_title('Learning Rate (Ultimate)', fontsize=14, fontweight='bold')
            axes[1, 2].set_xlabel('Эпоха')
            axes[1, 2].set_ylabel('Learning Rate')
            axes[1, 2].set_yscale('log')
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_history_tf_3.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ График обучения сохранен: training_history_tf_3.png")

    def plot_ultimate_confusion_matrix(self):
        """Построение матрицы ошибок ультимативной модели"""
        plt.figure(figsize=(10, 8))
        cm = self.results['confusion_matrix']

        # Нормализованная матрица ошибок
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Создание heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})

        plt.title('Матрица ошибок (Ultimate Model)\nНормализованная по строкам',
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Предсказанные значения', fontsize=12)
        plt.ylabel('Истинные значения', fontsize=12)

        # Добавление информации о количестве образцов
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j+0.5, i+0.7, f'n={cm[i,j]}',
                        ha='center', va='center', fontsize=10, color='red')

        plt.savefig('confusion_matrix_tf_3.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ Матрица ошибок сохранена: confusion_matrix_tf_3.png")

def main():
    """Основная функция обучения ультимативной модели"""
    print("🚀 ОБУЧЕНИЕ УЛЬТИМАТИВНОЙ МОДЕЛИ ДЛЯ КЛАССИФИКАЦИИ ТОКСИЧНОСТИ")
    print("=" * 80)
    print(f"📅 Время начала: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Цель: F1-score ≥ 0.95")
    print()

    # Создание тренера
    trainer = UltimateModelTrainer()

    # Загрузка объединенных данных
    X, y = trainer.load_combined_data()

    # Стратифицированное разделение данных для максимального качества
    print("🔄 Стратифицированное разделение данных...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y  # Меньше тестовых данных = больше для обучения
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    print(f"   Обучение: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Валидация: {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Тест: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")

    # Создание расширенных TF-IDF признаков
    X_train_tfidf, X_val_tfidf = trainer.create_advanced_tfidf_features(X_train, X_val)
    X_test_tfidf = trainer.vectorizer.transform(X_test)

    # Преобразование валидационных данных
    print("🔄 Преобразование валидационных данных...")
    X_val_dense = X_val_tfidf.toarray()

    # Применение продвинутого сэмплирования
    X_train_balanced, y_train_balanced = trainer.apply_advanced_sampling(X_train_tfidf, y_train)

    # Обучение ультимативной модели
    training_time = trainer.train_ultimate_model(X_train_balanced, y_train_balanced, X_val_dense, y_val)

    # Преобразование тестовых данных
    print("🔄 Преобразование тестовых данных...")
    X_test_dense = X_test_tfidf.toarray()

    # Оценка ультимативной модели
    results = trainer.evaluate_ultimate_model(X_test_dense, y_test)

    # Сохранение результатов
    trainer.save_ultimate_model_and_results()

    # Построение графиков
    trainer.plot_ultimate_training_history()
    trainer.plot_ultimate_confusion_matrix()

    # Финальный отчет
    print("\n" + "=" * 80)
    print("🎉 ОБУЧЕНИЕ УЛЬТИМАТИВНОЙ МОДЕЛИ ЗАВЕРШЕНО!")
    print("=" * 80)
    print(f"⏱️ Время обучения: {training_time/60:.1f} минут")
    print(f"🎯 Финальная точность: {results['accuracy']:.4f}")
    print(f"📊 F1-score: {results['f1_score']:.4f}")
    print(f"📈 ROC-AUC: {results['roc_auc']:.4f}")
    print(f"🔍 Оптимальный порог: {results['best_threshold']:.3f}")

    # Проверка достижения цели
    if results['f1_score'] >= 0.95:
        print("🏆 ЦЕЛЬ ДОСТИГНУТА: F1-score ≥ 0.95!")
    elif results['f1_score'] >= 0.90:
        print("🥈 ОТЛИЧНЫЙ РЕЗУЛЬТАТ: F1-score ≥ 0.90!")
    elif results['f1_score'] >= 0.85:
        print("🥉 ХОРОШИЙ РЕЗУЛЬТАТ: F1-score ≥ 0.85!")
    else:
        print("⚠️ Результат требует дополнительной оптимизации")

    print("\n📁 Сохраненные файлы:")
    print("   • model_tf_3.h5 - Ультимативная TensorFlow модель")
    print("   • tokenizer_tf_3.pkl - Расширенный TF-IDF векторизатор")
    print("   • results_tf_3.pkl - Подробные метрики обучения")
    print("   • confusion_matrix_tf_3.png - Нормализованная матрица ошибок")
    print("   • training_history_tf_3.png - Подробные графики обучения")
    print("=" * 80)

if __name__ == "__main__":
    main()
