# 📖 User Guide - LLM Text Summarization Tool

Welcome to the comprehensive user guide for the LLM Text Summarization Tool! This guide will help you get the most out of the application.

## 🚀 Quick Start

### First Time Setup

1. **Install the application** following the [Installation Guide](README.md#installation)
2. **Launch the app**:
   ```bash
   streamlit run main.py
   ```
3. **Open your browser** to `http://localhost:8501`
4. **Try the sample document** by keeping "Use built-in sample file" checked

### Your First Summary

1. **Select Language**: Choose "English" or "Chinese" in the sidebar
2. **Choose Mode**: 
   - "Fast Summarizer" for quick results
   - "Enhanced Summarizer" for detailed analysis
3. **Set Parameters**: Adjust max sentences (1-20)
4. **Click Process**: The app will automatically process the sample document
5. **View Results**: Check the Summary, Keywords, and Visualization tabs

## 🎯 Detailed Usage Guide

### Language Selection

#### English Mode
- **Best for**: News articles, reports, academic papers, business documents
- **Models Available**:
  - **BART-CNN**: Excellent for news and formal documents
  - **T5-Large**: Better for diverse content types
- **Features**: Advanced keyword extraction, phrase detection

#### Chinese Mode
- **Best for**: Chinese documents, reports, articles
- **Model**: Specialized Chinese BART model
- **Features**: Chinese-specific keyword extraction with jieba segmentation

### Summarization Modes

#### Fast Summarizer
- **Speed**: ⚡⚡⚡ Very Fast (10-20 seconds)
- **Quality**: ⭐⭐⭐ Good
- **Use Case**: Quick overviews, first-pass analysis
- **Features**:
  - Model selection (BART-CNN or T5-Large)
  - Basic keyword extraction
  - Simple text formatting

#### Enhanced Summarizer
- **Speed**: ⚡⚡ Moderate (20-40 seconds)
- **Quality**: ⭐⭐⭐⭐ Excellent
- **Use Case**: Detailed analysis, final summaries
- **Features**:
  - Advanced parameters (temperature, top-p)
  - Phrase-based keyword extraction
  - Markdown formatting with bullet points
  - Better context understanding

### Input Methods

#### File Upload
**Supported Formats**:
- **PDF**: `.pdf` files (text-based, not scanned images)
- **TXT**: `.txt` files (UTF-8, Latin-1, CP1252 encoding)
- **DOCX**: `.docx` files (Microsoft Word documents)

**File Requirements**:
- Maximum size: 10MB
- Minimum text: 50 characters
- Text-based content (not image-only documents)

**Upload Process**:
1. Click "Upload File" in the main area
2. Select your document
3. Wait for processing confirmation
4. Review the extracted text preview

#### Direct Text Input
1. Select "Paste Text" option
2. Paste your text into the text area
3. Ensure text is at least 50 characters
4. Click process when ready

**Tips for Text Input**:
- Remove unnecessary formatting
- Ensure proper encoding
- Include complete sentences
- Avoid very short texts

### Configuration Options

#### Sidebar Settings

**Language**:
- **English**: Uses English models and processing
- **Chinese**: Uses Chinese-specific models

**Summarize Mode** (English only):
- **Fast Summarizer**: Quick processing
- **Enhanced Summarizer**: Detailed analysis

**Model** (Fast mode only):
- **facebook/bart-large-cnn**: Best for news/articles
- **t5-large**: Better for diverse content

**Max Summary Sentences**:
- Range: 1-20 sentences
- Default: 5 sentences
- Recommendation: 3-7 for most documents

#### Advanced Settings

**Use Built-in Sample File**:
- Check: Process the included sample PDF
- Uncheck: Upload your own file or paste text

## 📊 Understanding Results

### Summary Tab
- **Content**: Generated summary in markdown format
- **Statistics**:
  - Summary Length: Character count
  - Original Length: Source text character count
  - Compression Ratio: Percentage reduction

**Reading Tips**:
- Enhanced mode uses bullet points for multiple key points
- Chinese mode splits sentences at Chinese punctuation (。！？)
- Summary quality improves with longer source texts

### Keywords Tab
- **English**: Individual words and phrases
- **Chinese**: Segmented Chinese terms
- **Display**: Organized in 3 columns
- **Count**: Top 15 keywords by default

**Keyword Types**:
- **Single Words**: Important individual terms
- **Phrases**: Multi-word concepts (Enhanced mode)
- **Filtered**: Stopwords and common terms removed

### Visualization Tab
- **Chart Type**: Horizontal bar chart
- **Y-axis**: Keywords (ranked by importance)
- **X-axis**: Importance score (higher = more important)
- **Chinese**: Uses Chinese fonts for proper display

**Interpreting Charts**:
- Longer bars = more important keywords
- Top keywords = most relevant to document content
- Use for quick content overview

### Original Text Tab
- **Preview**: First 1000 characters of source text
- **Full Length**: Shows total character count
- **Purpose**: Verify input quality and content

## 💾 Export Options

### Available Exports

#### Summary Export
- **Format**: Plain text (.txt)
- **Content**: Generated summary only
- **Use Case**: Share summaries, save for later

#### Keywords Export
- **Format**: Plain text (.txt)
- **Content**: List of keywords (one per line)
- **Use Case**: Further analysis, keyword research

#### Full Report Export
- **Format**: Plain text (.txt)
- **Content**: Complete analysis including:
  - Summary
  - Keywords
  - Statistics
  - Metadata
- **Use Case**: Documentation, comprehensive records

### Export Process
1. Process your document
2. Scroll to "Export Results" section
3. Click desired export button
4. Download will start automatically

## 🔧 Troubleshooting

### Common Issues

#### "Model Loading Failed"
**Causes**:
- Insufficient memory
- Network connectivity issues
- Corrupted model cache

**Solutions**:
```bash
# Clear model cache
rm -rf ~/.cache/huggingface/transformers/

# Restart application
streamlit run main.py
```

#### "Text Too Short" Error
**Causes**:
- Input text less than 50 characters
- Empty or whitespace-only text

**Solutions**:
- Ensure text has sufficient content
- Check for hidden characters
- Try a longer document

#### "No Summary Generated"
**Causes**:
- Model processing failure
- Invalid input format
- Memory constraints

**Solutions**:
- Try a different model
- Reduce max sentences
- Check input text quality
- Restart application

#### Chinese Font Issues
**Symptoms**:
- Squares or question marks in charts
- Missing Chinese characters

**Solutions**:
```bash
# Ubuntu/Debian
sudo apt-get install fonts-noto-cjk

# macOS
brew install font-noto-cjk

# Windows
# Download Noto CJK fonts from Google Fonts
```

### Performance Optimization

#### Speed Improvements
- Use Fast Summarizer mode
- Reduce max sentences
- Process smaller documents
- Close other applications

#### Memory Management
- Restart application periodically
- Use smaller models
- Process documents individually
- Monitor system memory usage

#### Quality Improvements
- Use Enhanced Summarizer for important documents
- Ensure good input text quality
- Use appropriate language settings
- Try different models for different content types

## 📈 Best Practices

### Document Preparation
1. **Clean Text**: Remove unnecessary formatting
2. **Complete Content**: Ensure full sentences
3. **Appropriate Length**: 500+ characters for best results
4. **Language Match**: Use correct language setting

### Model Selection
- **News Articles**: BART-CNN
- **Academic Papers**: Enhanced Summarizer
- **Business Documents**: T5-Large
- **Chinese Content**: Chinese Mode
- **Quick Overviews**: Fast Summarizer

### Parameter Tuning
- **Short Documents**: 3-5 sentences
- **Long Documents**: 5-10 sentences
- **Detailed Analysis**: Enhanced mode
- **Quick Overview**: Fast mode

### Workflow Optimization
1. **Start with samples** to understand the tool
2. **Use Fast mode** for initial exploration
3. **Switch to Enhanced** for final summaries
4. **Export results** for documentation
5. **Save configurations** for repeated use

## 🎓 Advanced Usage

### Batch Processing
For multiple documents:
1. Process documents individually
2. Export each result
3. Combine summaries manually
4. Use consistent parameters

### Custom Analysis
1. Export keywords for further analysis
2. Use visualization for content overview
3. Compare summaries across different models
4. Analyze compression ratios

### Integration Ideas
- Use exported summaries in reports
- Import keywords into other tools
- Combine with other NLP tools
- Create automated workflows

## 📞 Getting Help

### Self-Help Resources
- Check this user guide first
- Review the [README](README.md) for technical details
- Look at [Troubleshooting](README.md#troubleshooting) section

### Community Support
- [GitHub Issues](https://github.com/kelvin8773/llm_demo_text_summarize/issues): Bug reports
- [GitHub Discussions](https://github.com/kelvin8773/llm_demo_text_summarize/discussions): Questions and ideas

### Professional Support
- Email: [Your Email]
- Response time: 24-48 hours
- Include: Error messages, system details, steps to reproduce

---

**Happy Summarizing! 🎉**

This tool is designed to make document analysis faster and more efficient. Experiment with different settings to find what works best for your specific use case.