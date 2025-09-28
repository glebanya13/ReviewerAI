#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple toxicity classifier that outputs id,label format.
This is a simplified wrapper around the improved processor for easy use.
"""

import argparse
import sys
from pathlib import Path
from improved_csv_processor import ImprovedCSVToxicityProcessor

def main():
    """
    Simple command-line interface for toxicity classification.
    Always outputs in id,label format where 1=toxic, 0=non-toxic.
    """
    parser = argparse.ArgumentParser(
        description="Простой классификатор токсичности текста - выводит id,label (1=токсично, 0=нетоксично)"
    )
    parser.add_argument(
        'input_file',
        help='Путь к входному CSV файлу'
    )
    parser.add_argument(
        'output_file',
        help='Путь к выходному CSV файлу'
    )
    parser.add_argument(
        '--text-column',
        default='text',
        help='Название колонки с текстом (по умолчанию: text)'
    )
    parser.add_argument(
        '--id-column',
        default='id',
        help='Название колонки с ID (по умолчанию: id)'
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
        '--batch-size',
        type=int,
        default=50,
        help='Размер батча для обработки (по умолчанию: 50)'
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

    print("🔥 ПРОСТОЙ КЛАССИФИКАТОР ТОКСИЧНОСТИ")
    print("="*50)
    print(f"Входной файл: {args.input_file}")
    print(f"Выходной файл: {args.output_file}")
    print(f"Колонка с текстом: {args.text_column}")
    print(f"Колонка с ID: {args.id_column}")
    print("Формат вывода: id,label (1=токсично, 0=нетоксично)")
    print("="*50)

    # Initialize processor
    processor = ImprovedCSVToxicityProcessor(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        use_preprocessing=True,  # Always use preprocessing
        use_lemmatization=True,  # Always use lemmatization
        batch_size=args.batch_size
    )

    # Load model
    if not processor.load_model():
        print("❌ Не удалось загрузить модель!")
        return 1

    # Process CSV file with simple output
    stats = processor.process_csv_file(
        input_file=args.input_file,
        output_file=args.output_file,
        text_column=args.text_column,
        id_column=args.id_column,
        simple_output=True,  # Always use simple output
        preserve_columns=False,  # Not needed for simple output
        show_progress=not args.no_progress
    )

    if stats:
        # Print simplified statistics
        print("\n" + "="*50)
        print("📊 РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ")
        print("="*50)
        print(f"Всего текстов: {stats['total_texts']}")
        print(f"Токсичных (label=1): {stats['toxic_count']}")
        print(f"Нетоксичных (label=0): {stats['non_toxic_count']}")
        print(f"Процент токсичности: {stats['toxic_percentage']:.1f}%")
        print(f"Время обработки: {stats['processing_time']:.1f} сек")
        print("="*50)
        print(f"\n✅ Результаты сохранены в: {args.output_file}")
        print("📋 Формат: id,label (где 1=токсично, 0=нетоксично)")
        return 0
    else:
        print("❌ Ошибка при обработке файла!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
