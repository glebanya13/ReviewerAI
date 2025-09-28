import sys
import os
import pandas as pd

# Добавляем путь к модулям (как в вашем тесте)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from text_cleaner import clean_text
from lemm import ProfanityDetectorPreprocessor  # Или импортируйте FastProfanityPreprocessor, если нужно

def main():
    # Инициализируем препроцессор (как в тесте)
    preprocessor = ProfanityDetectorPreprocessor(
        max_sequence_length=1000,
        handle_typos=True,
        normalize_repeating_chars=True
    )
    
    # Читаем CSV (замените 'data.csv' на имя вашего файла)
    df = pd.read_csv('train.csv')
    
    # Сначала чистим все тексты
    cleaned_texts = df['text'].apply(clean_text).tolist()
    
    # Обрабатываем батчем для эффективности (как в test_batch_processing)
    batch_results = preprocessor.process_batch(cleaned_texts, return_attention_mask=True)
    
    # Добавляем обработанный текст: леммы, соединённые пробелами (для удобства в CSV)
    # Можно взять 'normalized_text' вместо лемм, если нужно: batch_results['normalized_texts']
    df['processed_text'] = [' '.join(lemmas) for lemmas in batch_results['lemmas']]
    
    # Сохраняем новый CSV (с id, text (оригинал), processed_text, label)
    df.to_csv('processed_data.csv', index=False)
    
    print("Обработка завершена. Новый файл: processed_data.csv")

if __name__ == "__main__":
    main()