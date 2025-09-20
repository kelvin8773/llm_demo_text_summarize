# tests/unit/test_enhance_summarize.py - Unit tests for enhanced summarization

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from utils.enhance_summarize import (
    enhance_summarize_text, 
    _split_sentences, 
    _chunk_text, 
    _format_markdown,
    _validate_input,
    _initialize_models
)
from tests.fixtures.sample_texts import (
    ENGLISH_SHORT_TEXT, ENGLISH_MEDIUM_TEXT, ENGLISH_LONG_TEXT,
    EMPTY_TEXT, WHITESPACE_ONLY_TEXT, VERY_SHORT_TEXT,
    VALID_MAX_SENTENCES, INVALID_MAX_SENTENCES
)

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestSplitSentences:
    """Test cases for _split_sentences function."""
    
    def test_basic_sentence_splitting(self):
        """Test basic sentence splitting functionality."""
        text = "This is sentence one. This is sentence two! This is sentence three?"
        result = _split_sentences(text)
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert "This is sentence one" in result[0]
        assert "This is sentence two" in result[1]
        assert "This is sentence three" in result[2]
    
    def test_empty_text(self):
        """Test splitting empty text."""
        result = _split_sentences("")
        assert result == []
    
    def test_whitespace_only_text(self):
        """Test splitting whitespace-only text."""
        result = _split_sentences("   \n\t   ")
        assert result == []
    
    def test_single_sentence(self):
        """Test splitting single sentence."""
        text = "This is a single sentence."
        result = _split_sentences(text)
        
        assert len(result) == 1
        assert result[0] == "This is a single sentence"
    
    def test_no_punctuation(self):
        """Test text without sentence-ending punctuation."""
        text = "This text has no punctuation"
        result = _split_sentences(text)
        
        assert len(result) == 1
        assert result[0] == "This text has no punctuation"
    
    def test_mixed_punctuation(self):
        """Test text with mixed punctuation."""
        text = "Sentence one. Sentence two! Sentence three? Sentence four."
        result = _split_sentences(text)
        
        assert len(result) == 4
        assert all("Sentence" in sentence for sentence in result)


class TestChunkText:
    """Test cases for _chunk_text function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    
    def test_basic_chunking(self):
        """Test basic text chunking."""
        text = "This is a test sentence. This is another test sentence."
        
        with patch('utils.enhance_summarize._tokenizer', self.mock_tokenizer):
            result = _chunk_text(text)
            
            assert isinstance(result, list)
            assert len(result) > 0
            assert all(isinstance(chunk, str) for chunk in result)
    
    def test_empty_text_chunking(self):
        """Test chunking empty text."""
        with patch('utils.enhance_summarize._tokenizer', self.mock_tokenizer):
            result = _chunk_text("")
            assert result == []
    
    def test_short_text_chunking(self):
        """Test chunking short text."""
        text = "Short text."
        
        with patch('utils.enhance_summarize._tokenizer', self.mock_tokenizer):
            result = _chunk_text(text)
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0] == text
    
    def test_long_text_chunking(self):
        """Test chunking long text."""
        text = ". ".join([f"This is sentence {i}" for i in range(20)])
        
        # Mock tokenizer to return long token sequence
        self.mock_tokenizer.encode.return_value = list(range(2000))
        
        with patch('utils.enhance_summarize._tokenizer', self.mock_tokenizer):
            result = _chunk_text(text, max_tokens=100)
            
            assert isinstance(result, list)
            assert len(result) > 1  # Should be split into multiple chunks
    
    def test_chunking_with_custom_max_tokens(self):
        """Test chunking with custom max_tokens parameter."""
        text = "This is a test sentence. This is another test sentence."
        
        with patch('utils.enhance_summarize._tokenizer', self.mock_tokenizer):
            result = _chunk_text(text, max_tokens=50)
            
            assert isinstance(result, list)
            assert len(result) > 0


class TestFormatMarkdown:
    """Test cases for _format_markdown function."""
    
    def test_single_sentence_formatting(self):
        """Test formatting single sentence."""
        text = "This is a single sentence."
        result = _format_markdown(text)
        
        assert result == "This is a single sentence"
    
    def test_multiple_sentences_formatting(self):
        """Test formatting multiple sentences with bullets."""
        text = "This is sentence one. This is sentence two. This is sentence three."
        result = _format_markdown(text)
        
        assert "- This is sentence one" in result
        assert "- This is sentence two" in result
        assert "- This is sentence three" in result
        assert result.count("-") == 3
    
    def test_empty_text_formatting(self):
        """Test formatting empty text."""
        result = _format_markdown("")
        assert result == ""
    
    def test_whitespace_text_formatting(self):
        """Test formatting whitespace-only text."""
        result = _format_markdown("   \n\t   ")
        assert result == ""


class TestValidateInput:
    """Test cases for _validate_input function."""
    
    def test_valid_input(self):
        """Test validation with valid input."""
        text = ENGLISH_MEDIUM_TEXT
        max_sentences = 5
        
        # Should not raise any exception
        _validate_input(text, max_sentences)
    
    def test_empty_text_validation(self):
        """Test validation with empty text."""
        with pytest.raises(ValueError, match="Input text is empty"):
            _validate_input("", 5)
    
    def test_whitespace_text_validation(self):
        """Test validation with whitespace-only text."""
        with pytest.raises(ValueError, match="Input text is empty"):
            _validate_input("   \n\t   ", 5)
    
    def test_short_text_validation(self):
        """Test validation with too short text."""
        with pytest.raises(ValueError, match="too short for meaningful summarization"):
            _validate_input("Short text", 5)
    
    def test_invalid_max_sentences_validation(self):
        """Test validation with invalid max_sentences."""
        for invalid_value in INVALID_MAX_SENTENCES:
            with pytest.raises(ValueError, match="max_sentences must be between"):
                _validate_input(ENGLISH_MEDIUM_TEXT, invalid_value)


class TestInitializeModels:
    """Test cases for _initialize_models function."""
    
    def test_model_initialization_success(self):
        """Test successful model initialization."""
        with patch('utils.enhance_summarize.AutoTokenizer') as mock_tokenizer, \
             patch('utils.enhance_summarize.pipeline') as mock_pipeline:
            
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.model_max_length = 1024
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            mock_pipeline_instance = Mock()
            mock_pipeline.return_value = mock_pipeline_instance
            
            # Should not raise any exception
            _initialize_models()
    
    def test_model_initialization_failure(self):
        """Test model initialization failure."""
        with patch('utils.enhance_summarize.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer.from_pretrained.side_effect = Exception("Model not found")
            
            with pytest.raises(Exception, match="Model initialization failed"):
                _initialize_models()


class TestEnhanceSummarizeText:
    """Test cases for enhance_summarize_text function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.valid_text = ENGLISH_MEDIUM_TEXT
        self.short_text = ENGLISH_SHORT_TEXT
        self.long_text = ENGLISH_LONG_TEXT
    
    def test_basic_functionality(self):
        """Test basic enhanced summarization functionality."""
        with patch('utils.enhance_summarize._initialize_models'), \
             patch('utils.enhance_summarize._tokenizer') as mock_tokenizer, \
             patch('utils.enhance_summarize._summarizer') as mock_summarizer:
            
            # Mock tokenizer
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
            
            # Mock summarizer
            mock_summarizer.return_value = [{"summary_text": "Enhanced summary result."}]
            
            result = enhance_summarize_text(self.valid_text, max_sentences=5)
            
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_markdown_formatting(self):
        """Test that result is properly formatted as markdown."""
        with patch('utils.enhance_summarize._initialize_models'), \
             patch('utils.enhance_summarize._tokenizer') as mock_tokenizer, \
             patch('utils.enhance_summarize._summarizer') as mock_summarizer:
            
            # Mock tokenizer
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
            
            # Mock summarizer returning multiple sentences
            mock_summarizer.return_value = [{"summary_text": "First point. Second point. Third point."}]
            
            result = enhance_summarize_text(self.valid_text, max_sentences=3)
            
            assert isinstance(result, str)
            # Should contain bullet points for multiple sentences
            if result.count("-") > 0:
                assert "- First point" in result
                assert "- Second point" in result
                assert "- Third point" in result
    
    def test_empty_text_error(self):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="Input text is empty"):
            enhance_summarize_text(EMPTY_TEXT)
    
    def test_short_text_error(self):
        """Test that short text raises ValueError."""
        with pytest.raises(ValueError, match="too short for meaningful summarization"):
            enhance_summarize_text(VERY_SHORT_TEXT)
    
    def test_invalid_max_sentences_error(self):
        """Test that invalid max_sentences raises ValueError."""
        for invalid_value in INVALID_MAX_SENTENCES:
            with pytest.raises(ValueError, match="max_sentences must be between"):
                enhance_summarize_text(self.valid_text, max_sentences=invalid_value)
    
    def test_chunking_failure(self):
        """Test handling of chunking failures."""
        with patch('utils.enhance_summarize._initialize_models'), \
             patch('utils.enhance_summarize._tokenizer') as mock_tokenizer:
            
            # Mock tokenizer that fails
            mock_tokenizer.encode.side_effect = Exception("Tokenization failed")
            
            with pytest.raises(Exception, match="Enhanced summarization failed"):
                enhance_summarize_text(self.valid_text)
    
    def test_summarization_failure(self):
        """Test handling of summarization failures."""
        with patch('utils.enhance_summarize._initialize_models'), \
             patch('utils.enhance_summarize._tokenizer') as mock_tokenizer, \
             patch('utils.enhance_summarize._summarizer') as mock_summarizer:
            
            # Mock tokenizer
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
            
            # Mock summarizer that fails
            mock_summarizer.side_effect = Exception("Summarization failed")
            
            with pytest.raises(Exception, match="Enhanced summarization failed"):
                enhance_summarize_text(self.valid_text)
    
    def test_empty_summary_handling(self):
        """Test handling of empty summary results."""
        with patch('utils.enhance_summarize._initialize_models'), \
             patch('utils.enhance_summarize._tokenizer') as mock_tokenizer, \
             patch('utils.enhance_summarize._summarizer') as mock_summarizer:
            
            # Mock tokenizer
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
            
            # Mock summarizer returning empty results
            mock_summarizer.return_value = [{"summary_text": ""}]
            
            with pytest.raises(Exception, match="No summaries were generated"):
                enhance_summarize_text(self.valid_text)
    
    def test_second_pass_summarization(self):
        """Test second-pass summarization for long combined results."""
        with patch('utils.enhance_summarize._initialize_models'), \
             patch('utils.enhance_summarize._tokenizer') as mock_tokenizer, \
             patch('utils.enhance_summarize._summarizer') as mock_summarizer:
            
            # Mock tokenizer
            mock_tokenizer.encode.side_effect = [
                [1, 2, 3, 4, 5],  # First chunk
                [1, 2, 3, 4, 5],  # Second chunk
                list(range(2000))  # Combined result (long)
            ]
            
            # Mock summarizer
            mock_summarizer.return_value = [{"summary_text": "Final enhanced summary."}]
            
            result = enhance_summarize_text(self.long_text, max_sentences=5)
            
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_sentence_limit_enforcement(self):
        """Test that sentence limit is properly enforced."""
        with patch('utils.enhance_summarize._initialize_models'), \
             patch('utils.enhance_summarize._tokenizer') as mock_tokenizer, \
             patch('utils.enhance_summarize._summarizer') as mock_summarizer:
            
            # Mock tokenizer
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
            
            # Mock summarizer returning long summary
            long_summary = ". ".join([f"This is sentence {i}" for i in range(10)])
            mock_summarizer.return_value = [{"summary_text": long_summary}]
            
            result = enhance_summarize_text(self.valid_text, max_sentences=3)
            
            assert isinstance(result, str)
            # Should be limited to 3 sentences
            sentence_count = result.count('.') + (1 if not result.endswith('.') else 0)
            assert sentence_count <= 3


class TestEnhanceSummarizeIntegration:
    """Integration tests for enhance_summarize_text function."""
    
    def test_end_to_end_processing(self):
        """Test complete end-to-end processing."""
        with patch('utils.enhance_summarize._initialize_models'), \
             patch('utils.enhance_summarize._tokenizer') as mock_tokenizer, \
             patch('utils.enhance_summarize._summarizer') as mock_summarizer:
            
            # Mock tokenizer
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
            
            # Mock summarizer
            mock_summarizer.return_value = [{"summary_text": "Complete enhanced summary with multiple points. It covers all aspects thoroughly. The result is comprehensive and well-formatted."}]
            
            result = enhance_summarize_text(ENGLISH_MEDIUM_TEXT, max_sentences=5)
            
            assert isinstance(result, str)
            assert len(result) > 0
            assert "enhanced summary" in result.lower()
    
    def test_performance_characteristics(self):
        """Test performance characteristics."""
        import time
        
        with patch('utils.enhance_summarize._initialize_models'), \
             patch('utils.enhance_summarize._tokenizer') as mock_tokenizer, \
             patch('utils.enhance_summarize._summarizer') as mock_summarizer:
            
            # Mock tokenizer
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
            
            # Mock summarizer
            mock_summarizer.return_value = [{"summary_text": "Performance test summary."}]
            
            start_time = time.time()
            result = enhance_summarize_text(ENGLISH_LONG_TEXT, max_sentences=10)
            end_time = time.time()
            
            assert isinstance(result, str)
            assert len(result) > 0
            # Should complete within reasonable time (mocked, so should be fast)
            assert (end_time - start_time) < 1.0  # 1 second max for mocked test