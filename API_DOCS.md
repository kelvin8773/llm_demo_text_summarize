# 🔧 API Documentation - LLM Text Summarization Tool

This document provides detailed API documentation for the core functions of the LLM Text Summarization Tool.

## 📚 Table of Contents

- [Core Summarization Functions](#core-summarization-functions)
- [Document Processing](#document-processing)
- [Keyword Extraction](#keyword-extraction)
- [Visualization](#visualization)
- [Configuration](#configuration)
- [Error Handling](#error-handling)

## 🚀 Core Summarization Functions

### `fast_summarize_text()`

Fast text summarization using transformer models with enhanced error handling.

```python
from utils.fast_summarize import fast_summarize_text

summary = fast_summarize_text(
    text: str,
    max_sentences: int = 3,
    model_name: str = "facebook/bart-large-cnn"
) -> str
```

**Parameters:**
- `text` (str): Input text to summarize (minimum 50 characters)
- `max_sentences` (int): Maximum number of sentences in summary (1-50)
- `model_name` (str): Model identifier from `parameters.py`

**Returns:**
- `str`: Generated summary text

**Raises:**
- `ValueError`: For invalid input parameters
- `Exception`: For model loading or processing errors

**Example:**
```python
text = "Your long document text here..."
summary = fast_summarize_text(text, max_sentences=5, model_name="t5-large")
print(summary)
```

### `enhance_summarize_text()`

Enhanced summarization with advanced parameters and markdown formatting.

```python
from utils.enhance_summarize import enhance_summarize_text

summary = enhance_summarize_text(
    text: str,
    max_sentences: int = 10
) -> str
```

**Parameters:**
- `text` (str): Input text to summarize
- `max_sentences` (int): Maximum number of sentences in summary

**Returns:**
- `str`: Markdown-formatted summary with bullet points

**Features:**
- Advanced sampling parameters (temperature=0.8, top_p=0.9)
- Markdown formatting with bullet points
- Two-pass summarization for long texts
- Sentence-aware chunking

**Example:**
```python
text = "Your document content..."
summary = enhance_summarize_text(text, max_sentences=8)
# Returns markdown with bullet points for multiple key points
```

### `chinese_summarize_text()`

Chinese text summarization with specialized models and processing.

```python
from utils.chinese_summarize import chinese_summarize_text

summary = chinese_summarize_text(
    text: str,
    max_sentences: int = 10
) -> str
```

**Parameters:**
- `text` (str): Chinese text to summarize
- `max_sentences` (int): Maximum number of sentences

**Returns:**
- `str`: Chinese summary with bullet point formatting

**Features:**
- Chinese-specific BART model
- Chinese punctuation-aware sentence splitting
- Optimized tokenization for Chinese text

**Example:**
```python
chinese_text = "您的中文文档内容..."
summary = chinese_summarize_text(chinese_text, max_sentences=6)
```

## 📄 Document Processing

### `load_document()`

Load text content from various document formats.

```python
from utils.ingest import load_document

text = load_document(file) -> str
```

**Parameters:**
- `file`: File object (uploaded file or file-like object)

**Returns:**
- `str`: Extracted text content

**Supported Formats:**
- PDF (`.pdf`)
- Text (`.txt`)
- Word documents (`.docx`)

**File Requirements:**
- Maximum size: 10MB
- Minimum text: 50 characters
- Text-based content (not image-only)

**Raises:**
- `ValueError`: For unsupported formats or empty documents
- `Exception`: For file processing errors

**Example:**
```python
with open("document.pdf", "rb") as file:
    text = load_document(file)
    print(f"Extracted {len(text)} characters")
```

## 🔍 Keyword Extraction

### `extract_keywords()`

Basic keyword extraction using TF-IDF.

```python
from utils.insights import extract_keywords

keywords = extract_keywords(text: str, top_n: int = 10) -> List[str]
```

**Parameters:**
- `text` (str): Input text for keyword extraction
- `top_n` (int): Number of top keywords to return

**Returns:**
- `List[str]`: List of keywords sorted by importance

**Features:**
- TF-IDF vectorization
- English stopwords filtering
- Single-word extraction

### `extract_keywords_phrases()`

Advanced keyword and phrase extraction.

```python
from utils.insights import extract_keywords_phrases

keywords = extract_keywords_phrases(text: str, top_n: int = 10) -> List[str]
```

**Parameters:**
- `text` (str): Input text for extraction
- `top_n` (int): Number of top keywords/phrases

**Returns:**
- `List[str]`: List of keywords and phrases

**Features:**
- Noun chunk extraction using spaCy
- Multi-word phrase detection
- Custom stopwords filtering
- N-gram analysis (1-3 grams)

### `extract_chinese_keywords()`

Chinese keyword extraction with jieba segmentation.

```python
from utils.chinese_insights import extract_chinese_keywords

keywords = extract_chinese_keywords(text: str, top_n: int = 10) -> List[str]
```

**Parameters:**
- `text` (str): Chinese text for extraction
- `top_n` (int): Number of top keywords

**Returns:**
- `List[str]`: Chinese keywords and phrases

**Features:**
- jieba Chinese segmentation
- Chinese stopwords filtering
- Bi-gram phrase extraction
- Custom blocklist support

## 📊 Visualization

### `plot_keywords()`

Create keyword importance visualization for English text.

```python
from utils.insights import plot_keywords

fig = plot_keywords(keywords: List[str]) -> matplotlib.figure.Figure
```

**Parameters:**
- `keywords` (List[str]): List of keywords to visualize

**Returns:**
- `matplotlib.figure.Figure`: Horizontal bar chart

**Features:**
- Horizontal bar chart
- Ranked by importance
- Matplotlib integration

### `plot_chinese_keywords()`

Create keyword visualization for Chinese text.

```python
from utils.chinese_insights import plot_chinese_keywords

fig = plot_chinese_keywords(keywords: List[str]) -> matplotlib.figure.Figure
```

**Parameters:**
- `keywords` (List[str]): Chinese keywords to visualize

**Returns:**
- `matplotlib.figure.Figure`: Chinese-compatible chart

**Features:**
- Chinese font support
- Proper character rendering
- Downloadable font fallback

## ⚙️ Configuration

### Model Parameters

```python
from utils.parameters import (
    BART_CNN_MODEL,      # "facebook/bart-large-cnn"
    T5_LARGE_MODEL,      # "t5-large"
    CHINESE_MODEL        # "uer/bart-base-chinese-cluecorpussmall"
)
```

### Customization Options

#### Adding Custom Stopwords

```python
# English stopwords
CUSTOM_STOPWORDS = list(
    ENGLISH_STOP_WORDS.union(
        {"your", "custom", "stopwords"}
    )
)

# Chinese stopwords
CUSTOM_BLOCKLIST = set(["公司", "数据", "业务"])
```

#### Model Configuration

```python
# Custom model parameters
MAX_INPUT_TOKENS = 1024
MAX_LENGTH = 150
MIN_LENGTH = 30
TEMPERATURE = 0.8
TOP_P = 0.9
```

## 🚨 Error Handling

### Common Exceptions

#### `ValueError`
- Empty or invalid input text
- Unsupported file formats
- Parameter out of range
- File size too large

#### `Exception`
- Model loading failures
- Network connectivity issues
- Memory constraints
- Processing errors

### Error Recovery

```python
try:
    summary = fast_summarize_text(text, max_sentences=5)
except ValueError as e:
    print(f"Input error: {e}")
    # Handle invalid input
except Exception as e:
    print(f"Processing error: {e}")
    # Handle processing failure
```

### Logging

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log processing steps
logger.info("Loading model...")
logger.info("Processing text...")
logger.error("Error occurred: {error}")
```

## 🔧 Advanced Usage

### Batch Processing

```python
def process_multiple_documents(files):
    """Process multiple documents in batch."""
    results = []
    for file in files:
        try:
            text = load_document(file)
            summary = fast_summarize_text(text)
            keywords = extract_keywords(text)
            results.append({
                'file': file.name,
                'summary': summary,
                'keywords': keywords
            })
        except Exception as e:
            logger.error(f"Failed to process {file.name}: {e}")
    return results
```

### Custom Model Integration

```python
def custom_summarize(text, model_name, **kwargs):
    """Custom summarization with additional parameters."""
    from transformers import pipeline
    
    summarizer = pipeline(
        "summarization",
        model=model_name,
        **kwargs
    )
    
    return summarizer(text)[0]["summary_text"]
```

### Performance Optimization

```python
import torch

# GPU acceleration
device = 0 if torch.cuda.is_available() else -1

# Memory optimization
torch.cuda.empty_cache()  # Clear GPU memory

# Batch processing
def chunk_and_process(text, chunk_size=1000):
    """Process large texts in chunks."""
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = []
    for chunk in chunks:
        summary = fast_summarize_text(chunk)
        summaries.append(summary)
    return " ".join(summaries)
```

## 📝 Examples

### Complete Workflow

```python
from utils.ingest import load_document
from utils.fast_summarize import fast_summarize_text
from utils.insights import extract_keywords, plot_keywords
import matplotlib.pyplot as plt

# Load document
with open("document.pdf", "rb") as file:
    text = load_document(file)

# Generate summary
summary = fast_summarize_text(text, max_sentences=5)

# Extract keywords
keywords = extract_keywords(text, top_n=15)

# Create visualization
fig = plot_keywords(keywords)
plt.show()

# Print results
print("Summary:", summary)
print("Keywords:", keywords)
```

### Chinese Document Processing

```python
from utils.chinese_summarize import chinese_summarize_text
from utils.chinese_insights import extract_chinese_keywords, plot_chinese_keywords

# Chinese text
chinese_text = "您的中文文档内容..."

# Process
summary = chinese_summarize_text(chinese_text, max_sentences=8)
keywords = extract_chinese_keywords(chinese_text, top_n=10)

# Visualize
fig = plot_chinese_keywords(keywords)
plt.show()
```

---

**Note**: This API documentation covers the core functionality. For additional features and updates, refer to the [README](README.md) and [User Guide](USER_GUIDE.md).