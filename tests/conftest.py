# tests/conftest.py - Pytest configuration and fixtures

"""
Pytest configuration file with shared fixtures and test utilities.
"""

import pytest
import logging
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
from tests.fixtures.sample_texts import (
    ENGLISH_SHORT_TEXT, ENGLISH_MEDIUM_TEXT, ENGLISH_LONG_TEXT,
    CHINESE_SHORT_TEXT, CHINESE_MEDIUM_TEXT, CHINESE_LONG_TEXT,
    SAMPLE_PDF_CONTENT, SAMPLE_TXT_CONTENT, SAMPLE_DOCX_CONTENT
)


# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@pytest.fixture(scope="session")
def test_logger():
    """Provide a test logger."""
    return logging.getLogger("test")


@pytest.fixture
def english_texts():
    """Provide English test texts."""
    return {
        "short": ENGLISH_SHORT_TEXT,
        "medium": ENGLISH_MEDIUM_TEXT,
        "long": ENGLISH_LONG_TEXT
    }


@pytest.fixture
def chinese_texts():
    """Provide Chinese test texts."""
    return {
        "short": CHINESE_SHORT_TEXT,
        "medium": CHINESE_MEDIUM_TEXT,
        "long": CHINESE_LONG_TEXT
    }


@pytest.fixture
def sample_documents():
    """Provide sample document contents."""
    return {
        "pdf": SAMPLE_PDF_CONTENT,
        "txt": SAMPLE_TXT_CONTENT,
        "docx": SAMPLE_DOCX_CONTENT
    }


@pytest.fixture
def mock_file_objects():
    """Provide mock file objects for testing."""
    def create_mock_file(name, size=1024, content=b"test content"):
        mock_file = Mock()
        mock_file.name = name
        mock_file.size = size
        mock_file.read.return_value = content
        return mock_file
    
    return {
        "pdf": create_mock_file("test.pdf", 1024),
        "txt": create_mock_file("test.txt", 1024, SAMPLE_TXT_CONTENT.encode('utf-8')),
        "docx": create_mock_file("test.docx", 1024),
        "large": create_mock_file("large.pdf", 15 * 1024 * 1024),  # 15MB
        "unsupported": create_mock_file("test.xyz", 1024)
    }


@pytest.fixture
def temp_directory():
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_fast_summarize():
    """Mock fast_summarize components with proper function mocking."""
    with patch('utils.fast_summarize._load_tokenizer') as mock_load_tokenizer, \
         patch('utils.fast_summarize._load_summarizer') as mock_load_summarizer, \
         patch('utils.fast_summarize.optimize_text_chunking') as mock_chunking:
        
        # Configure mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.encode.return_value = [1, 2, 3, 4, 5]
        mock_load_tokenizer.return_value = mock_tokenizer_instance
        
        # Configure mock summarizer
        mock_summarizer_instance = Mock()
        mock_summarizer_instance.return_value = [{"summary_text": "Mock summary result."}]
        mock_load_summarizer.return_value = mock_summarizer_instance
        
        # Configure mock chunking
        mock_chunking.return_value = ["This is a test chunk of text for summarization."]
        
        yield {
            "tokenizer": mock_tokenizer_instance,
            "summarizer": mock_summarizer_instance,
            "chunking": mock_chunking
        }


@pytest.fixture
def mock_enhance_summarize():
    """Mock enhance_summarize components with proper function mocking."""
    with patch('utils.enhance_summarize._initialize_models') as mock_init_models, \
         patch('utils.enhance_summarize._tokenizer') as mock_tokenizer, \
         patch('utils.enhance_summarize._summarizer') as mock_summarizer, \
         patch('utils.enhance_summarize._chunk_text') as mock_chunk_text:
        
        # Configure mock tokenizer
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        
        # Configure mock summarizer
        mock_summarizer.return_value = [{"summary_text": "Enhanced summary result."}]
        
        # Configure mock chunking
        mock_chunk_text.return_value = ["This is a test chunk of text for enhanced summarization."]
        
        # Mock model initialization
        mock_init_models.return_value = None
        
        yield {
            "tokenizer": mock_tokenizer,
            "summarizer": mock_summarizer,
            "init_models": mock_init_models,
            "chunking": mock_chunk_text
        }


@pytest.fixture
def mock_transformers():
    """Mock transformers library components."""
    with patch('utils.fast_summarize.AutoTokenizer') as mock_tokenizer, \
         patch('utils.fast_summarize.pipeline') as mock_pipeline, \
         patch('utils.enhance_summarize.AutoTokenizer') as mock_enhance_tokenizer, \
         patch('utils.enhance_summarize.pipeline') as mock_enhance_pipeline, \
         patch('utils.chinese_summarize.AutoTokenizer') as mock_chinese_tokenizer, \
         patch('utils.chinese_summarize.pipeline') as mock_chinese_pipeline:
        
        # Configure mock tokenizers
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer_instance.model_max_length = 1024
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_enhance_tokenizer_instance = Mock()
        mock_enhance_tokenizer_instance.encode.return_value = [1, 2, 3, 4, 5]
        mock_enhance_tokenizer_instance.model_max_length = 1024
        mock_enhance_tokenizer.from_pretrained.return_value = mock_enhance_tokenizer_instance
        
        mock_chinese_tokenizer_instance = Mock()
        mock_chinese_tokenizer_instance.encode.return_value = [1, 2, 3, 4, 5]
        mock_chinese_tokenizer_instance.model_max_length = 1024
        mock_chinese_tokenizer.from_pretrained.return_value = mock_chinese_tokenizer_instance
        
        # Configure mock pipelines
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = [{"summary_text": "Mock summary result."}]
        mock_pipeline.return_value = mock_pipeline_instance
        
        mock_enhance_pipeline_instance = Mock()
        mock_enhance_pipeline_instance.return_value = [{"summary_text": "Mock enhanced summary result."}]
        mock_enhance_pipeline.return_value = mock_enhance_pipeline_instance
        
        mock_chinese_pipeline_instance = Mock()
        mock_chinese_pipeline_instance.return_value = [{"summary_text": "模拟中文摘要结果。"}]
        mock_chinese_pipeline.return_value = mock_chinese_pipeline_instance
        
        yield {
            "tokenizer": mock_tokenizer_instance,
            "pipeline": mock_pipeline_instance,
            "enhance_tokenizer": mock_enhance_tokenizer_instance,
            "enhance_pipeline": mock_enhance_pipeline_instance,
            "chinese_tokenizer": mock_chinese_tokenizer_instance,
            "chinese_pipeline": mock_chinese_pipeline_instance
        }


@pytest.fixture
def mock_spacy():
    """Mock spaCy components."""
    with patch('utils.insights.spacy.load') as mock_load:
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_chunk1 = Mock()
        mock_chunk1.text = "artificial intelligence"
        mock_chunk2 = Mock()
        mock_chunk2.text = "machine learning"
        mock_doc.noun_chunks = [mock_chunk1, mock_chunk2]
        mock_nlp.return_value = mock_doc
        mock_load.return_value = mock_nlp
        
        yield mock_nlp


@pytest.fixture
def mock_jieba():
    """Mock jieba components."""
    with patch('utils.chinese_insights.jieba.cut') as mock_cut:
        mock_cut.return_value = ["人工智能", "机器学习", "数据处理"]
        yield mock_cut


@pytest.fixture
def mock_matplotlib():
    """Mock matplotlib components."""
    with patch('matplotlib.pyplot.subplots') as mock_subplots, \
         patch('matplotlib.pyplot.tight_layout') as mock_tight_layout:
        
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        yield {
            "fig": mock_fig,
            "ax": mock_ax,
            "subplots": mock_subplots,
            "tight_layout": mock_tight_layout
        }


@pytest.fixture
def mock_sklearn():
    """Mock scikit-learn components."""
    with patch('utils.insights.TfidfVectorizer') as mock_vectorizer, \
         patch('utils.chinese_insights.TfidfVectorizer') as mock_chinese_vectorizer:
        
        # Configure mock vectorizers
        mock_vectorizer_instance = Mock()
        mock_vectorizer_instance.fit_transform.return_value = Mock()
        mock_vectorizer_instance.get_feature_names_out.return_value = ["test", "keywords"]
        mock_vectorizer.return_value = mock_vectorizer_instance
        
        mock_chinese_vectorizer_instance = Mock()
        mock_chinese_vectorizer_instance.fit_transform.return_value = Mock()
        mock_chinese_vectorizer_instance.get_feature_names_out.return_value = ["测试", "关键词"]
        mock_chinese_vectorizer.return_value = mock_chinese_vectorizer_instance
        
        yield {
            "vectorizer": mock_vectorizer_instance,
            "chinese_vectorizer": mock_chinese_vectorizer_instance
        }


@pytest.fixture
def mock_document_loaders():
    """Mock document loading components."""
    with patch('utils.ingest.PdfReader') as mock_pdf_reader, \
         patch('utils.ingest.docx.Document') as mock_docx:
        
        # Configure mock PDF reader
        mock_reader = Mock()
        mock_page = Mock()
        mock_page.extract_text.return_value = SAMPLE_PDF_CONTENT
        mock_reader.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader
        
        # Configure mock DOCX document
        mock_doc = Mock()
        mock_para = Mock()
        mock_para.text = SAMPLE_DOCX_CONTENT
        mock_doc.paragraphs = [mock_para]
        mock_docx.return_value = mock_doc
        
        yield {
            "pdf_reader": mock_reader,
            "docx_document": mock_doc
        }


@pytest.fixture
def performance_benchmarks():
    """Provide performance benchmark data."""
    return {
        "max_processing_time": 30.0,  # seconds
        "max_memory_usage": 2048,    # MB
        "min_accuracy": 0.8,         # 80%
        "max_response_time": 5.0     # seconds
    }


# Test markers
pytestmark = [
    pytest.mark.unit,
]


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual functions"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for complete workflows"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take longer to run"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and benchmark tests"
    )
    config.addinivalue_line(
        "markers", "visualization: Tests involving matplotlib plotting"
    )
    config.addinivalue_line(
        "markers", "chinese: Tests specific to Chinese language processing"
    )
    config.addinivalue_line(
        "markers", "english: Tests specific to English language processing"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add language-specific markers
        if "chinese" in item.name.lower():
            item.add_marker(pytest.mark.chinese)
        elif "english" in item.name.lower():
            item.add_marker(pytest.mark.english)
        
        # Add visualization markers
        if "plot" in item.name.lower() or "visualization" in item.name.lower():
            item.add_marker(pytest.mark.visualization)
        
        # Add performance markers
        if "performance" in item.name.lower() or "benchmark" in item.name.lower():
            item.add_marker(pytest.mark.performance)
        
        # Add slow markers for long-running tests
        if "large" in item.name.lower() or "long" in item.name.lower():
            item.add_marker(pytest.mark.slow)


# Test utilities
class TestUtils:
    """Utility functions for tests."""
    
    @staticmethod
    def assert_valid_summary(summary, min_length=10, max_length=None):
        """Assert that a summary is valid."""
        assert isinstance(summary, str)
        assert len(summary) >= min_length
        if max_length:
            assert len(summary) <= max_length
        assert summary.strip() == summary  # No leading/trailing whitespace
    
    @staticmethod
    def assert_valid_keywords(keywords, min_count=1, max_count=None):
        """Assert that keywords are valid."""
        assert isinstance(keywords, list)
        assert len(keywords) >= min_count
        if max_count:
            assert len(keywords) <= max_count
        assert all(isinstance(kw, str) for kw in keywords)
        assert all(len(kw) > 0 for kw in keywords)
    
    @staticmethod
    def assert_valid_error_message(error, expected_keywords):
        """Assert that an error message contains expected keywords."""
        error_msg = str(error).lower()
        for keyword in expected_keywords:
            assert keyword.lower() in error_msg


# Make TestUtils available as a fixture
@pytest.fixture
def test_utils():
    """Provide test utility functions."""
    return TestUtils