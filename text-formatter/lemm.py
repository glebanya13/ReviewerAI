
import re
import string
from typing import List, Tuple, Dict, Optional, Union
import numpy as np

# Импорт необходимых библиотек
try:
    import pymorphy2
    morph = pymorphy2.MorphAnalyzer()
    PYMORPHY_AVAILABLE = True
except ImportError:
    PYMORPHY_AVAILABLE = False
    print("⚠️ PyMorphy2 не установлен. Установите: pip install pymorphy2")

try:
    from razdel import tokenize
    RAZDEL_AVAILABLE = True
except ImportError:
    RAZDEL_AVAILABLE = False
    print("⚠️ Razdel не установлен. Установите: pip install razdel")

class ProfanityDetectorPreprocessor:
    """
    Препроцессор текста для нейросети, определяющей нецензурную лексику.
    Оптимизирован для сохранения признаков, важных для детекции мата.
    """

    def __init__(
        self,
        max_sequence_length: int = 100,
        preserve_word_boundaries: bool = True,
        handle_typos: bool = True,
        normalize_repeating_chars: bool = True,
        keep_original_forms: bool = True
    ):
        """
        Args:
            max_sequence_length: Максимальная длина последовательности токенов
            preserve_word_boundaries: Сохранять границы слов для контекста
            handle_typos: Обрабатывать типичные замены символов (@ -> а, 0 -> о)
            normalize_repeating_chars: Нормализовать повторяющиеся символы (ааааа -> аа)
            keep_original_forms: Сохранять оригинальные формы вместе с леммами
        """
        self.max_sequence_length = max_sequence_length
        self.preserve_word_boundaries = preserve_word_boundaries
        self.handle_typos = handle_typos
        self.normalize_repeating_chars = normalize_repeating_chars
        self.keep_original_forms = keep_original_forms
        
        # Словарь замен для обхода цензуры
        self.char_replacements = {
            '@': 'а', '4': 'ч', '6': 'б', '3': 'з', '0': 'о',
            '1': 'и', '7': 'т', '5': 'п', '9': 'д', '8': 'в',
            'a': 'а', 'e': 'е', 'o': 'о', 'p': 'р', 'c': 'с',
            'y': 'у', 'x': 'х', 'k': 'к', 'm': 'м', 'h': 'н',
            't': 'т', 'b': 'в'
        }
        
        # Паттерны для разделителей в обфусцированном тексте
        self.separator_pattern = re.compile(r'[\s\-_\.\,\!\?\*]+')
        
        # Инициализация морфологического анализатора
        self.morph = morph if PYMORPHY_AVAILABLE else None

    def clean_text(self, text: str) -> str:
        """
        Базовая очистка текста с сохранением важных признаков.
        """
        # Приведение к нижнему регистру
        text = text.lower()
        
        # Удаление лишних пробелов
        text = ' '.join(text.split())
        
        return text

    def handle_obfuscation(self, text: str) -> str:
        """
        Обработка обфусцированного текста (замены символов, разделители).
        """
        if not self.handle_typos:
            return text
        
        # Удаление разделителей внутри слов (х_у_й -> хуй)
        words = text.split()
        processed_words = []
        
        for word in words:
            # Проверка на наличие подозрительных разделителей
            if len(word) > 2 and any(sep in word for sep in '-_.'):
                # Удаление разделителей, если они разбивают слово
                clean_word = re.sub(r'[\-_\.]', '', word)
                processed_words.append(clean_word)
            else:
                processed_words.append(word)
        
        text = ' '.join(processed_words)
        
        # Замена символов на буквы
        for old_char, new_char in self.char_replacements.items():
            text = text.replace(old_char, new_char)
        
        return text

    def normalize_repeating(self, text: str) -> str:
        """
        Нормализация повторяющихся символов (блллляяяя -> бляя).
        """
        if not self.normalize_repeating_chars:
            return text
        
        # Заменяем 3+ повторения на 2 повторения
        pattern = re.compile(r'(.)\1{2,}')
        text = pattern.sub(r'\1\1', text)
        
        return text

    def tokenize_text(self, text: str) -> List[str]:
        """
        Токенизация текста с учетом особенностей задачи.
        """
        if RAZDEL_AVAILABLE:
            # Используем razdel для качественной токенизации
            tokens = [token.text for token in tokenize(text)]
        else:
            # Fallback на простую токенизацию
            # Сохраняем пунктуацию как отдельные токены для контекста
            pattern = r'[\w]+|[^\w\s]'
            tokens = re.findall(pattern, text)
        
        return tokens

    def lemmatize_token(self, token: str) -> Tuple[str, str]:
        """
        Лемматизация токена с сохранением оригинальной формы.
        
        Returns:
            (оригинальный_токен, лемма)
        """
        if not self.morph or not token.isalpha():
            return (token, token)
        
        try:
            parsed = self.morph.parse(token)[0]
            lemma = parsed.normal_form
            return (token, lemma)
        except:
            return (token, token)

    def process_for_nn(
        self,
        text: str,
        return_attention_mask: bool = True
    ) -> Dict[str, Union[List[str], List[Tuple[str, str]], np.ndarray]]:
        """
        Полная обработка текста для нейросети.
        
        Args:
            text: Входной текст
            return_attention_mask: Возвращать ли маску внимания для паддинга
            
        Returns:
            Словарь с обработанными данными
        """
        # 1. Базовая очистка
        cleaned = self.clean_text(text)
        
        # 2. Сохраняем оригинал для сравнения
        original_cleaned = cleaned
        
        # 3. Обработка обфускации
        deobfuscated = self.handle_obfuscation(cleaned)
        
        # 4. Нормализация повторений
        normalized = self.normalize_repeating(deobfuscated)
        
        # 5. Токенизация
        tokens = self.tokenize_text(normalized)
        
        # 6. Лемматизация с сохранением оригиналов
        token_lemma_pairs = []
        lemmas = []
        
        for token in tokens:
            orig, lemma = self.lemmatize_token(token)
            token_lemma_pairs.append((orig, lemma))
            lemmas.append(lemma)
        
        # 7. Обрезка или паддинг до max_sequence_length
        if len(lemmas) > self.max_sequence_length:
            lemmas = lemmas[:self.max_sequence_length]
            token_lemma_pairs = token_lemma_pairs[:self.max_sequence_length]
            attention_mask = [1] * self.max_sequence_length
        else:
            padding_length = self.max_sequence_length - len(lemmas)
            attention_mask = [1] * len(lemmas) + [0] * padding_length
            lemmas = lemmas + ['[PAD]'] * padding_length
            token_lemma_pairs = token_lemma_pairs + [('[PAD]', '[PAD]')] * padding_length
        
        result = {
            'original_text': text,
            'cleaned_text': original_cleaned,
            'deobfuscated_text': deobfuscated,
            'normalized_text': normalized,
            'tokens': tokens[:self.max_sequence_length],
            'lemmas': lemmas,
            'token_lemma_pairs': token_lemma_pairs if self.keep_original_forms else None,
            'sequence_length': min(len(tokens), self.max_sequence_length)
        }
        
        if return_attention_mask:
            result['attention_mask'] = np.array(attention_mask)
        
        return result

    def process_batch(
        self,
        texts: List[str],
        return_attention_mask: bool = True
    ) -> Dict[str, Union[List, np.ndarray]]:
        """
        Пакетная обработка текстов для нейросети.
        
        Args:
            texts: Список текстов
            return_attention_mask: Возвращать ли маски внимания
            
        Returns:
            Батч данных для нейросети
        """
        batch_results = [self.process_for_nn(text, return_attention_mask) for text in texts]
        
        # Собираем батч
        batch = {
            'lemmas': [r['lemmas'] for r in batch_results],
            'sequence_lengths': [r['sequence_length'] for r in batch_results],
            'original_texts': [r['original_text'] for r in batch_results]
        }
        
        if return_attention_mask:
            batch['attention_masks'] = np.stack([r['attention_mask'] for r in batch_results])
        
        if self.keep_original_forms:
            batch['token_lemma_pairs'] = [r['token_lemma_pairs'] for r in batch_results]
        
        return batch

    def create_vocab_from_texts(
        self,
        texts: List[str],
        min_freq: int = 2
    ) -> Dict[str, int]:
        """
        Создание словаря из текстов для векторизации.
        
        Args:
            texts: Список текстов для обучения
            min_freq: Минимальная частота для включения в словарь
            
        Returns:
            Словарь {токен: индекс}
        """
        token_freq = {}
        
        for text in texts:
            result = self.process_for_nn(text)
            for lemma in result['lemmas']:
                if lemma != '[PAD]':
                    token_freq[lemma] = token_freq.get(lemma, 0) + 1
        
        # Специальные токены
        vocab = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[CLS]': 2,
            '[SEP]': 3,
            '[MASK]': 4
        }
        
        # Добавляем токены с частотой >= min_freq
        idx = len(vocab)
        for token, freq in sorted(token_freq.items(), key=lambda x: x[1], reverse=True):
            if freq >= min_freq:
                vocab[token] = idx
                idx += 1
        
        return vocab

    def encode_tokens(
        self,
        tokens: List[str],
        vocab: Dict[str, int]
    ) -> List[int]:
        """
        Кодирование токенов в индексы.
        
        Args:
            tokens: Список токенов
            vocab: Словарь токенов
            
        Returns:
            Список индексов
        """
        unk_idx = vocab.get('[UNK]', 1)
        return [vocab.get(token, unk_idx) for token in tokens]

    def prepare_for_embedding(
        self,
        text: str,
        vocab: Dict[str, int]
    ) -> Dict[str, np.ndarray]:
        """
        Подготовка текста для embedding слоя нейросети.
        
        Args:
            text: Входной текст
            vocab: Словарь токенов
            
        Returns:
            Словарь с закодированными данными
        """
        processed = self.process_for_nn(text)
        encoded = self.encode_tokens(processed['lemmas'], vocab)
        
        return {
            'input_ids': np.array(encoded),
            'attention_mask': processed['attention_mask'],
            'length': processed['sequence_length']
        }

class FastProfanityPreprocessor:
    """
    Упрощенная и быстрая версия препроцессора для real-time обработки.
    """

    def __init__(self, use_cache: bool = True):
        """
        Args:
            use_cache: Использовать ли кеш для лемматизации
        """
        self.use_cache = use_cache
        self.lemma_cache = {} if use_cache else None
        self.morph = morph if PYMORPHY_AVAILABLE else None
        
        # Быстрые замены для обфускации
        self.quick_replacements = str.maketrans(
            '@4630175980aeopcyxkmhtbв',
            'ачбзоитпдоваеорсухкмнтвв'
        )

    def quick_process(self, text: str) -> List[str]:
        """
        Быстрая обработка текста для предсказания.
        
        Args:
            text: Входной текст
            
        Returns:
            Список лемм
        """
        # Быстрая нормализация
        text = text.lower().translate(self.quick_replacements)
        text = re.sub(r'[\-_\.]', '', text)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # Простая токенизация
        tokens = re.findall(r'\b\w+\b', text)
        
        # Лемматизация с кешем
        lemmas = []
        for token in tokens:
            if self.use_cache and token in self.lemma_cache:
                lemmas.append(self.lemma_cache[token])
            elif self.morph and token.isalpha():
                lemma = self.morph.parse(token)[0].normal_form
                if self.use_cache:
                    self.lemma_cache[token] = lemma
                lemmas.append(lemma)
            else:
                lemmas.append(token)
        
        return lemmas