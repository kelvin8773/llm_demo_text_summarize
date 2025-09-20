# utils/__init__.py - Package initialization and public API

"""
LLM Text Summarization Utils Package

This package provides core functionality for text summarization, keyword extraction,
and document processing using various language models.

Main modules:
- fast_summarize: Fast summarization with transformer models
- enhance_summarize: Enhanced summarization with advanced features
- chinese_summarize: Chinese language processing
- insights: English keyword extraction and visualization
- chinese_insights: Chinese keyword extraction
- ingest: Document loading and parsing
- parameters: Model configurations and constants
"""

from .fast_summarize import fast_summarize_text
from .enhance_summarize import enhance_summarize_text
from .chinese_summarize import chinese_summarize_text
from .insights import extract_keywords, extract_keywords_phrases, plot_keywords
from .chinese_insights import extract_chinese_keywords, plot_chinese_keywords
from .ingest import load_document
from .parameters import (
    BART_CNN_MODEL,
    T5_LARGE_MODEL,
    CHINESE_MODEL,
    get_model_info,
    get_available_models,
    validate_model,
    get_model_display_name,
    DEFAULT_MAX_SENTENCES,
    DEFAULT_KEYWORDS_COUNT
)

# Public API
__all__ = [
    # Summarization functions
    "fast_summarize_text",
    "enhance_summarize_text", 
    "chinese_summarize_text",
    
    # Keyword extraction functions
    "extract_keywords",
    "extract_keywords_phrases",
    "extract_chinese_keywords",
    
    # Visualization functions
    "plot_keywords",
    "plot_chinese_keywords",
    
    # Document processing
    "load_document",
    
    # Model constants
    "BART_CNN_MODEL",
    "T5_LARGE_MODEL", 
    "CHINESE_MODEL",
    
    # Helper functions
    "get_model_info",
    "get_available_models",
    "validate_model",
    "get_model_display_name",
    
    # Default parameters
    "DEFAULT_MAX_SENTENCES",
    "DEFAULT_KEYWORDS_COUNT"
]

# Package metadata
__version__ = "1.0.0"
__author__ = "LLM Text Summarization Tool"
__description__ = "Core utilities for LLM-based text summarization and analysis"