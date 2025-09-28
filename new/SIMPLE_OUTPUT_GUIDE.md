# Simple Output Format Guide

## 🎯 Two-Column Output Format

Your toxicity detector now supports a clean, simple output format with just two columns:

- **`id`**: The unique identifier from your input data
- **`label`**: Binary classification (1 = toxic, 0 = non-toxic)

## 🚀 Usage Options

### Option 1: Using the Simple Classifier Script

```bash
python simple_toxicity_classifier.py input.csv output.csv
```

**Features:**
- Always outputs in `id,label` format
- Automatic preprocessing and optimization
- Simplified statistics
- Clean command-line interface

### Option 2: Using the Full Processor with Simple Output Flag

```bash
python improved_csv_processor.py input.csv -o output.csv --simple-output
```

**Features:**
- Full control over all parameters
- Detailed statistics and method tracking
- Can switch between simple and full output formats

## 📊 Example Output

**Input CSV:**
```csv
id,text,source
1,"Спасибо за помощь, очень полезная информация!",test
2,Ты дурак и идиот!,test
3,"Хорошая работа, продолжайте в том же духе",test
4,"Сука блядь, что за херня?!",test
```

**Output CSV (Simple Format):**
```csv
id,label
1,0
2,1
3,0
4,1
```

## 🔧 Command Line Options

### Simple Classifier
```bash
python simple_toxicity_classifier.py [options] input_file output_file

Options:
  --text-column TEXT      Column name with text data (default: text)
  --id-column ID          Column name with ID data (default: id)
  --model MODEL          Path to model file (default: model_tf.h5)
  --tokenizer TOKENIZER  Path to tokenizer file (default: tokenizer_tf.pkl)
  --batch-size N         Batch size for processing (default: 50)
  --no-progress          Disable progress bar
```

### Full Processor with Simple Output
```bash
python improved_csv_processor.py [options] input_file

Options:
  -o, --output OUTPUT    Output CSV file
  --simple-output        Enable simple id,label format
  --id-column ID         Column name with ID data (default: id)
  -c, --column TEXT      Column name with text data (default: text)
  --batch-size N         Batch size for processing (default: 50)
  --no-preprocessing     Disable text preprocessing
  --no-lemmatization     Disable lemmatization
  --no-progress          Disable progress bar
```

## 📈 Performance Benefits

- **Clean Output**: Only essential columns (id, label)
- **Fast Processing**: Optimized for bulk classification
- **Memory Efficient**: Minimal output storage
- **Easy Integration**: Standard format for ML pipelines

## 🎮 Quick Examples

### Basic Usage
```bash
# Simple classification
python simple_toxicity_classifier.py data.csv results.csv

# With custom columns
python simple_toxicity_classifier.py data.csv results.csv --text-column "comment" --id-column "post_id"
```

### Advanced Usage
```bash
# Full processor with simple output
python improved_csv_processor.py data.csv -o results.csv --simple-output

# Custom ID column and larger batches
python improved_csv_processor.py data.csv -o results.csv --simple-output --id-column "comment_id" --batch-size 100
```

## 📋 Input Requirements

Your input CSV must have:
1. **Text column** (default: "text") - contains the text to classify
2. **ID column** (default: "id") - unique identifier for each row

## ✅ Output Guarantee

The output will always be a valid CSV with exactly two columns:
- `id`: Preserves your original IDs
- `label`: Binary classification (1 = toxic, 0 = non-toxic)

## 🔍 Label Definitions

- **`1` (Toxic)**: Text contains toxic content including:
  - Profanity and obscenities
  - Insults and personal attacks
  - Hate speech
  - Obfuscated toxic words (е.g., "с.у.к.а", "$ук@")

- **`0` (Non-toxic)**: Text is clean and appropriate:
  - Normal conversation
  - Helpful information
  - Neutral or positive content

## 🛠️ Troubleshooting

**Common Issues:**

1. **Missing ID column**: Ensure your CSV has the specified ID column
2. **Missing text column**: Ensure your CSV has the specified text column
3. **Model files not found**: Check that `model_tf.h5` and `tokenizer_tf.pkl` exist

**Error Messages:**
- `❌ Колонка 'id' не найдена в файле!` - Check your ID column name
- `❌ Колонка 'text' не найдена в файле!` - Check your text column name
- `❌ Файл модели model_tf.h5 не найден!` - Check model file path

## 🔮 Integration Examples

### Python Integration
```python
from improved_csv_processor import ImprovedCSVToxicityProcessor

processor = ImprovedCSVToxicityProcessor()
processor.load_model()

stats = processor.process_csv_file(
    input_file='data.csv',
    output_file='results.csv',
    simple_output=True  # Key parameter for simple format
)
```

### Batch Processing
```bash
# Process multiple files
for file in *.csv; do
    python simple_toxicity_classifier.py "$file" "${file%.csv}_results.csv"
done
```

This simple output format makes it easy to integrate toxicity detection into your data processing pipelines!
