# 📄 LLM Text Summarization Tool

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A powerful, multilingual text and document summarization tool built with Streamlit and advanced language models. Transform lengthy documents into concise summaries with intelligent keyword extraction and visualization.

## 🌟 Features

### 📝 **Multi-Language Support**
- **English**: Advanced summarization with BART and T5 models
- **Chinese**: Specialized Chinese language processing with BART-based models
- Automatic language detection and appropriate model selection

### 🚀 **Multiple Summarization Modes**
- **Fast Summarizer**: Quick processing using transformer models (BART-CNN, T5-Large)
- **Enhanced Summarizer**: Detailed analysis with advanced parameters and markdown formatting
- **Chinese Mode**: Optimized for Chinese text with specialized models

### 📁 **Flexible Input Methods**
- **File Upload**: Support for PDF, TXT, and DOCX files
- **Direct Text Input**: Paste text directly into the interface
- **Sample Documents**: Built-in sample files for demonstration

### 🔍 **Advanced Analysis**
- **Keyword Extraction**: TF-IDF based keyword extraction with customizable stopwords
- **Phrase Detection**: Multi-word phrase extraction for better context
- **Visualization**: Interactive keyword importance charts
- **Export Options**: Download summaries, keywords, and full reports

### 🎯 **Smart Processing**
- **Chunking**: Intelligent text splitting for large documents
- **Error Handling**: Robust error handling with user-friendly messages
- **Progress Tracking**: Real-time progress indicators
- **Validation**: Input validation and quality checks

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM (recommended for model loading)
- Internet connection (for model downloads)

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kelvin8773/llm_demo_text_summarize.git
   cd llm_demo_text_summarize
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy English model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Run the application:**
   ```bash
   streamlit run main.py
   ```

6. **Open your browser:**
   Navigate to `http://localhost:8501`

### Alternative Installation Methods

#### Using pip (Development)
```bash
pip install streamlit transformers torch scikit-learn nltk matplotlib PyPDF2 python-docx jieba spacy
```

#### Using conda
```bash
conda create -n llm-summarizer python=3.9
conda activate llm-summarizer
pip install -r requirements.txt
```

## 🚀 Usage

### Basic Usage

1. **Launch the application:**
   ```bash
   streamlit run main.py
   ```

2. **Configure settings in the sidebar:**
   - Select language (English/Chinese)
   - Choose summarization mode
   - Select model (for Fast mode)
   - Set maximum sentences

3. **Input your content:**
   - Upload a file (PDF, TXT, DOCX)
   - Or paste text directly

4. **View results:**
   - Summary in markdown format
   - Extracted keywords
   - Visualization charts
   - Export options

### Advanced Usage

#### Command Line Interface
```python
from utils.fast_summarize import fast_summarize_text
from utils.enhance_summarize import enhance_summarize_text
from utils.chinese_summarize import chinese_summarize_text

# Fast summarization
summary = fast_summarize_text("Your text here", max_sentences=5)

# Enhanced summarization
summary = enhance_summarize_text("Your text here", max_sentences=10)

# Chinese summarization
summary = chinese_summarize_text("您的中文文本", max_sentences=8)
```

#### Custom Model Configuration
```python
from utils.parameters import BART_CNN_MODEL, T5_LARGE_MODEL

# Use specific models
summary = fast_summarize_text(text, model_name=T5_LARGE_MODEL)
```

## 📊 Supported Models

### English Models
- **facebook/bart-large-cnn**: High-quality CNN-style summarization
- **t5-large**: Google's T5 model for text-to-text tasks

### Chinese Models
- **uer/bart-base-chinese-cluecorpussmall**: Chinese BART model trained on CLUE corpus

### Model Selection Guide
- **BART-CNN**: Best for news articles and formal documents
- **T5-Large**: Better for diverse content types
- **Chinese BART**: Optimized for Chinese text processing

## 📁 Project Structure

```
llm_demo_text_summarize/
├── main.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Project configuration
├── README.md              # This file
├── data/                  # Sample documents
│   ├── AI_Transformation_Playbook.pdf
│   ├── Chinese_text_China_GovReport_2024.txt
│   ├── English_text_China_RealState.txt
│   ├── sample_keywords.txt
│   └── sample_summary.txt
└── utils/                 # Core functionality modules
    ├── fast_summarize.py      # Fast summarization with transformers
    ├── enhance_summarize.py   # Enhanced summarization with advanced features
    ├── chinese_summarize.py   # Chinese language processing
    ├── insights.py            # English keyword extraction and visualization
    ├── chinese_insights.py    # Chinese keyword extraction
    ├── ingest.py              # Document loading and parsing
    └── parameters.py          # Model configurations
```

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set CUDA device for GPU acceleration
export CUDA_VISIBLE_DEVICES=0

# Optional: Set model cache directory
export TRANSFORMERS_CACHE=/path/to/cache
```

### Customization Options

#### Adding Custom Stopwords
```python
# In utils/insights.py
CUSTOM_STOPWORDS = list(
    ENGLISH_STOP_WORDS.union(
        {"your", "custom", "stopwords", "here"}
    )
)
```

#### Adjusting Model Parameters
```python
# In utils/parameters.py
CUSTOM_MODEL = "your-preferred-model-name"
```

## 🐛 Troubleshooting

### Common Issues

#### Model Loading Errors
```bash
# Clear model cache
rm -rf ~/.cache/huggingface/transformers/

# Reinstall transformers
pip install --upgrade transformers torch
```

#### Memory Issues
- Reduce `max_sentences` parameter
- Use smaller models
- Process documents in smaller chunks

#### Chinese Font Issues
```bash
# Install Chinese fonts
sudo apt-get install fonts-noto-cjk  # Ubuntu/Debian
brew install font-noto-cjk           # macOS
```

### Performance Optimization

#### GPU Acceleration
```python
# In utils modules, change device parameter
device=0  # Use GPU
device=-1 # Use CPU (default)
```

#### Memory Management
- Use smaller batch sizes
- Enable gradient checkpointing
- Use mixed precision training

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/kelvin8773/llm_demo_text_summarize.git
cd llm_demo_text_summarize
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

### Running Tests
```bash
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for the language models
- [Streamlit](https://streamlit.io/) for the web interface
- [spaCy](https://spacy.io/) for NLP processing
- [jieba](https://github.com/fxsjy/jieba) for Chinese text segmentation

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/kelvin8773/llm_demo_text_summarize/issues)
- **Discussions**: [GitHub Discussions](https://github.com/kelvin8773/llm_demo_text_summarize/discussions)
- **Email**: [Your Email]

## 🔄 Changelog

### Version 1.0.0
- Initial release with English and Chinese support
- Multiple summarization modes
- File upload functionality
- Keyword extraction and visualization
- Export capabilities

---

**Made with ❤️ for the NLP community**
