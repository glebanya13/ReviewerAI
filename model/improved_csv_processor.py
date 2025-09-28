#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Improved CSV processor with enhanced toxicity detection and better error handling.
"""

import pandas as pd
import re
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time
import warnings

# Import our improved toxicity detector
from toxicity_detector import ToxicityPredictor

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to import optional dependencies with better error handling
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    
    # Create a simple progress indicator
    class tqdm:
        def __init__(self, iterable, desc="Processing", **kwargs):
            self.iterable = iterable
            self.desc = desc
            self.total = len(iterable) if hasattr(iterable, '__len__') else 0
            self.current = 0
            print(f"{desc}: 0/{self.total}")
            
        def __iter__(self):
            for item in self.iterable:
                self.current += 1
                if self.current % max(1, self.total // 10) == 0:  # Show progress every 10%
                    print(f"{self.desc}: {self.current}/{self.total}")
                yield item


class EnhancedTextPreprocessor:
    """
    Enhanced text preprocessor with improved obfuscation handling
    and better performance.
    """
    
    def __init__(self, use_simple_lemmatization: bool = True):
        """
        Initialize the preprocessor.
        
        Args:
            use_simple_lemmatization: Whether to apply simple lemmatization
        """
        self.use_simple_lemmatization = use_simple_lemmatization
        
        # Character replacements for obfuscation handling
        self.char_replacements = {
            '@': 'а', '4': 'ч', '6': 'б', '3': 'з', '0': 'о',
            '1': 'и', '7': 'т', '5': 'п', '9': 'д', '8': 'в',
            'a': 'а', 'e': 'е', 'o': 'о', 'p': 'р', 'c': 'с',
            'y': 'у', 'x': 'х', 'k': 'к', 'm': 'м', 'h': 'н',
            't': 'т', 'b': 'в'
        }
        
        # Quick translation table for performance
        self.translation_table = str.maketrans(
            '@4630175980aeopcyxkmhtb',
            'ачбзоитпдоваеорсухкмнтв'
        )
        
        # Simple lemmatization endings
        self.endings = [
            'ый', 'ая', 'ое', 'ые', 'ых', 'ым', 'ыми',
            'ий', 'яя', 'ее', 'ие', 'их', 'им', 'ими',
            'ов', 'ова', 'ово', 'овы', 'овых', 'овым', 'овыми',
            'ев', 'ева', 'ево', 'евы', 'евых', 'евым', 'евыми',
            'ть', 'ти', 'л', 'ла', 'ло', 'ли', 'лсь', 'лась', 'лось', 'лись',
            'ся', 'сь', 'ать', 'ять', 'еть', 'ить', 'уть', 'ыть',
            'аю', 'аешь', 'ает', 'аем', 'аете', 'ают',
            'яю', 'яешь', 'яет', 'яем', 'яете', 'яют',
            'ею', 'еешь', 'еет', 'еем', 'еете', 'еют',
            'ию', 'ишь', 'ит', 'им', 'ите', 'ят'
        ]

    def clean_text(self, text: str) -> str:
        """
        Enhanced text cleaning with better obfuscation handling.
        """
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text).lower()
        
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        
        # Remove dates
        text = re.sub(
            r"(\d{4}[-./]\d{1,2}[-./]\d{1,2})|(\d{1,2}[-./]\d{1,2}[-./]\d{2,4})",
            "",
            text,
        )
        
        # Handle spaced obfuscation (e.g., "с у к а")
        text = re.sub(r'(\w)\s+(\w)\s+(\w)', r'\1\2\3', text)
        
        # Handle obfuscation: remove separators within words
        words = text.split()
        processed_words = []
        for word in words:
            if len(word) > 2 and any(sep in word for sep in '-_.'):
                clean_word = re.sub(r'[\-_\.]', '', word)
                processed_words.append(clean_word)
            else:
                processed_words.append(word)
        text = ' '.join(processed_words)
        
        # Character replacements for obfuscation
        text = text.translate(self.translation_table)
        
        # Normalize repeating characters (3+ -> 2)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # Remove emojis and special symbols, keeping letters, digits, punctuation, spaces
        text = re.sub(r"[^\w\s.,!?-]", " ", text, flags=re.UNICODE)
        
        # Remove number sequences of 3+ digits
        text = re.sub(r"\d{3,}", "", text)
        
        # Normalize spaces
        text = re.sub(r"\s+", " ", text).strip()
        
        return text

    def simple_lemmatize(self, text: str) -> str:
        """
        Simple lemmatization with space preservation.
        """
        words = text.split()
        lemmatized = []
        
        for word in words:
            if len(word) < 3:
                lemmatized.append(word)
                continue
            
            # Try to remove endings
            found = False
            for ending in sorted(self.endings, key=len, reverse=True):
                if word.endswith(ending) and len(word) > len(ending) + 1:
                    lemmatized.append(word[:-len(ending)])
                    found = True
                    break
            
            if not found:
                lemmatized.append(word)
        
        return ' '.join(lemmatized)

    def preprocess_text(self, text: str) -> str:
        """
        Complete text preprocessing pipeline.
        """
        if pd.isna(text) or text == '':
            return ''
        
        # 1. Clean text
        cleaned = self.clean_text(text)
        
        # 2. Remove single characters
        words = cleaned.split()
        words = [word for word in words if len(word) > 1]
        cleaned = ' '.join(words)
        
        # 3. Apply simple lemmatization if enabled
        if self.use_simple_lemmatization:
            processed = self.simple_lemmatize(cleaned)
        else:
            processed = cleaned
        
        return processed


class ImprovedCSVToxicityProcessor:
    """
    Improved processor class for handling CSV files with enhanced toxicity detection.
    """
    
    def __init__(
        self,
        model_path: str = 'model_tf.h5',
        tokenizer_path: str = 'tokenizer_tf.pkl',
        use_preprocessing: bool = True,
        use_lemmatization: bool = True,
        batch_size: int = 100
    ):
        """
        Initialize the improved CSV toxicity processor.
        
        Args:
            model_path: Path to the trained model
            tokenizer_path: Path to the tokenizer
            use_preprocessing: Whether to apply advanced preprocessing
            use_lemmatization: Whether to apply lemmatization
            batch_size: Batch size for processing
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.use_preprocessing = use_preprocessing
        self.batch_size = batch_size
        
        # Initialize components
        self.toxicity_predictor = ToxicityPredictor()
        self.preprocessor = EnhancedTextPreprocessor(
            use_simple_lemmatization=use_lemmatization
        ) if use_preprocessing else None
        
        # Load model
        self.model_loaded = False

    def load_model(self) -> bool:
        """
        Load the toxicity detection model.
        """
        print("🔄 Загрузка улучшенной модели...")
        success = self.toxicity_predictor.load_model(
            self.model_path, 
            self.tokenizer_path
        )
        self.model_loaded = success
        return success

    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        Preprocess a list of texts.
        """
        if not self.use_preprocessing or not self.preprocessor:
            return texts
        
        processed = []
        for text in texts:
            processed_text = self.preprocessor.preprocess_text(text)
            processed.append(processed_text)
        
        return processed

    def process_csv_file(
        self,
        input_file: str,
        output_file: str,
        text_column: str = 'text',
        id_column: str = 'id',
        simple_output: bool = False,
        preserve_columns: bool = True,
        show_progress: bool = True
    ) -> Dict[str, any]:
        """
        Process a CSV file and add improved toxicity predictions.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file
            text_column: Name of the column containing text data
            id_column: Name of the column containing ID data
            simple_output: If True, output only id and label columns (1=toxic, 0=non-toxic)
            preserve_columns: Whether to preserve all original columns (ignored if simple_output=True)
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary with processing statistics
        """
        if not self.model_loaded:
            print("❌ Модель не загружена! Сначала вызовите load_model()")
            return None
        
        print(f"📖 Загрузка CSV файла: {input_file}")
        
        try:
            df = pd.read_csv(input_file)
        except Exception as e:
            print(f"❌ Ошибка при загрузке файла: {e}")
            return None
        
        if text_column not in df.columns:
            print(f"❌ Колонка '{text_column}' не найдена в файле!")
            print(f"Доступные колонки: {list(df.columns)}")
            return None
        
        if simple_output and id_column not in df.columns:
            print(f"❌ Колонка '{id_column}' не найдена в файле!")
            print(f"Доступные колонки: {list(df.columns)}")
            return None
        
        print(f"✅ Загружено {len(df)} записей")
        print(f"📊 Обрабатываем колонку: '{text_column}'")
        
        # Prepare results
        results = []
        processed_texts = []
        original_texts = []
        total_texts = len(df)
        
        # Process in batches for better performance
        print(f"🔄 Обработка текстов (batch_size={self.batch_size})...")
        
        start_time = time.time()
        
        # Create progress iterator
        batch_range = range(0, total_texts, self.batch_size)
        if show_progress:
            iterator = tqdm(batch_range, desc="Обработка батчей")
        else:
            iterator = batch_range
        
        for i in iterator:
            batch_texts = df[text_column].iloc[i:i+self.batch_size].tolist()
            original_texts.extend(batch_texts)
            
            # Preprocess if enabled
            if self.use_preprocessing:
                batch_processed = self.preprocess_texts(batch_texts)
                processed_texts.extend(batch_processed)
                # Use processed texts for prediction
                prediction_texts = batch_processed
            else:
                prediction_texts = batch_texts
                processed_texts.extend([''] * len(batch_texts))  # Empty processed texts
            
            # Get toxicity predictions with improved method
            batch_results = self.toxicity_predictor.predict_batch(
                prediction_texts
            )
            results.extend(batch_results)
        
        processing_time = time.time() - start_time
        
        # Prepare prediction results
        is_toxic = [r['is_toxic'] if r else False for r in results]
        toxicity_prob = [r['toxicity_probability'] if r else 0.0 for r in results]
        confidence = [r['confidence'] if r else 0.0 for r in results]
        methods = [r.get('method', 'unknown') if r else 'error' for r in results]
        errors = [r.get('error', '') if r else 'Ошибка обработки' for r in results]
        notes = [r.get('note', '') if r else '' for r in results]
        
        # Create output dataframe based on output format
        if simple_output:
            # Simple output: only id and label columns
            output_df = pd.DataFrame()
            output_df['id'] = df[id_column]
            output_df['label'] = [1 if toxic else 0 for toxic in is_toxic]
        else:
            # Full output: preserve columns and add detailed predictions
            if preserve_columns:
                output_df = df.copy()
            else:
                output_df = pd.DataFrame()
                output_df[text_column] = df[text_column]
            
            # Add preprocessing column if enabled
            if self.use_preprocessing:
                output_df['processed_text'] = processed_texts
            
            # Add improved toxicity prediction columns
            output_df['is_toxic'] = is_toxic
            output_df['toxicity_probability'] = toxicity_prob
            output_df['confidence'] = confidence
            output_df['detection_method'] = methods
            output_df['prediction_error'] = errors
            output_df['notes'] = notes
        
        # Save results
        print(f"💾 Сохранение результатов в: {output_file}")
        try:
            output_df.to_csv(output_file, index=False, encoding='utf-8')
        except Exception as e:
            print(f"❌ Ошибка при сохранении: {e}")
            return None
        
        # Calculate enhanced statistics
        toxic_count = sum(is_toxic)
        valid_predictions = sum(1 for r in results if r and 'error' not in r)
        method_counts = {}
        for method in methods:
            method_counts[method] = method_counts.get(method, 0) + 1
        
        stats = {
            'total_texts': total_texts,
            'valid_predictions': valid_predictions,
            'toxic_count': toxic_count,
            'non_toxic_count': valid_predictions - toxic_count,
            'error_count': total_texts - valid_predictions,
            'toxic_percentage': (toxic_count / valid_predictions * 100) if valid_predictions > 0 else 0,
            'processing_time': processing_time,
            'texts_per_second': total_texts / processing_time if processing_time > 0 else 0,
            'method_counts': method_counts
        }
        
        return stats

    def print_statistics(self, stats: Dict[str, any]):
        """
        Print enhanced processing statistics.
        """
        if not stats:
            return
        
        print("\n" + "="*70)
        print("📊 РАСШИРЕННАЯ СТАТИСТИКА ОБРАБОТКИ")
        print("="*70)
        print(f"Всего текстов: {stats['total_texts']}")
        print(f"Успешно обработано: {stats['valid_predictions']}")
        print(f"Ошибок обработки: {stats['error_count']}")
        print(f"Токсичных текстов: {stats['toxic_count']}")
        print(f"Нетоксичных текстов: {stats['non_toxic_count']}")
        print(f"Процент токсичности: {stats['toxic_percentage']:.2f}%")
        print(f"Время обработки: {stats['processing_time']:.2f} сек")
        print(f"Скорость: {stats['texts_per_second']:.2f} текстов/сек")
        
        print("\n📈 МЕТОДЫ ДЕТЕКЦИИ:")
        method_emojis = {
            'model': '🤖 Модель',
            'pattern_fallback': '🔍 Паттерны',
            'combined': '🤖+🔍 Комбинированный',
            'empty_text': '⚪ Пустой текст',
            'tokenization_failed': '⚠️ Ошибка токенизации',
            'padding_failed': '⚠️ Ошибка padding',
            'model_failed': '❌ Ошибка модели',
            'error_fallback': '❌ Ошибка с fallback',
            'error': '❌ Общая ошибка',
            'unknown': '❓ Неизвестно'
        }
        
        for method, count in stats['method_counts'].items():
            emoji_method = method_emojis.get(method, f"❓ {method}")
            percentage = (count / stats['total_texts'] * 100) if stats['total_texts'] > 0 else 0
            print(f"  {emoji_method}: {count} ({percentage:.1f}%)")
        
        print("="*70)


def main():
    """
    Main function with improved command-line interface.
    """
    parser = argparse.ArgumentParser(
        description="Улучшенная обработка CSV файлов с детекцией токсичности текста"
    )
    parser.add_argument(
        'input_file',
        help='Путь к входному CSV файлу'
    )
    parser.add_argument(
        '-o', '--output',
        default='improved_output_with_toxicity.csv',
        help='Путь к выходному CSV файлу (по умолчанию: improved_output_with_toxicity.csv)'
    )
    parser.add_argument(
        '-c', '--column',
        default='text',
        help='Название колонки с текстом (по умолчанию: text)'
    )
    parser.add_argument(
        '--id-column',
        default='id',
        help='Название колонки с ID (по умолчанию: id)'
    )
    parser.add_argument(
        '--simple-output',
        action='store_true',
        help='Выводить только колонки id и label (1=токсично, 0=нетоксично)'
    )
    parser.add_argument(
        '--model',
        default='model_tf.h5',
        help='Путь к файлу модели (по умолчанию: model_tf.h5)'
    )
    parser.add_argument(
        '--tokenizer',
        default='tokenizer_tf.pkl',
        help='Путь к файлу токенизатора (по умолчанию: tokenizer_tf.pkl)'
    )
    parser.add_argument(
        '--no-preprocessing',
        action='store_true',
        help='Отключить предобработку текста'
    )
    parser.add_argument(
        '--no-lemmatization',
        action='store_true',
        help='Отключить лемматизацию'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Размер батча для обработки (по умолчанию: 50)'
    )
    parser.add_argument(
        '--preserve-columns',
        action='store_true',
        default=True,
        help='Сохранить все исходные колонки (по умолчанию: включено)'
    )
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Отключить индикатор прогресса'
    )

    args = parser.parse_args()

    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"❌ Файл {args.input_file} не найден!")
        return 1

    # Check if model files exist
    if not Path(args.model).exists():
        print(f"❌ Файл модели {args.model} не найден!")
        return 1
    
    if not Path(args.tokenizer).exists():
        print(f"❌ Файл токенизатора {args.tokenizer} не найден!")
        return 1

    print("🚀 ЗАПУСК УЛУЧШЕННОЙ ОБРАБОТКИ CSV С ДЕТЕКЦИЕЙ ТОКСИЧНОСТИ")
    print("="*70)
    print(f"Входной файл: {args.input_file}")
    print(f"Выходной файл: {args.output}")
    print(f"Колонка с текстом: {args.column}")
    if args.simple_output:
        print(f"Колонка с ID: {args.id_column}")
        print("Формат вывода: Простой (id, label)")
    else:
        print("Формат вывода: Расширенный")
    print(f"Модель: {args.model}")
    print(f"Токенизатор: {args.tokenizer}")
    print(f"Предобработка: {'выключена' if args.no_preprocessing else 'включена'}")
    print(f"Лемматизация: {'выключена' if args.no_lemmatization else 'включена'}")
    print(f"Размер батча: {args.batch_size}")
    print(f"Прогресс: {'выключен' if args.no_progress else 'включен'}")
    print("="*70)

    # Initialize improved processor
    processor = ImprovedCSVToxicityProcessor(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        use_preprocessing=not args.no_preprocessing,
        use_lemmatization=not args.no_lemmatization,
        batch_size=args.batch_size
    )

    # Load model
    if not processor.load_model():
        print("❌ Не удалось загрузить модель!")
        return 1

    # Process CSV file
    stats = processor.process_csv_file(
        input_file=args.input_file,
        output_file=args.output,
        text_column=args.column,
        id_column=args.id_column,
        simple_output=args.simple_output,
        preserve_columns=args.preserve_columns,
        show_progress=not args.no_progress
    )

    if stats:
        processor.print_statistics(stats)
        print(f"\n✅ Обработка завершена! Результаты сохранены в: {args.output}")
        return 0
    else:
        print("❌ Ошибка при обработке файла!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
