# tests/unit/test_fast_summarize.py - Unit tests for fast summarization

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from utils.fast_summarize import fast_summarize_text
from utils.parameters import BART_CNN_MODEL, T5_LARGE_MODEL
from tests.fixtures.sample_texts import (
    ENGLISH_SHORT_TEXT, ENGLISH_MEDIUM_TEXT, ENGLISH_LONG_TEXT,
    EMPTY_TEXT, WHITESPACE_ONLY_TEXT, VERY_SHORT_TEXT,
    VALID_MAX_SENTENCES, INVALID_MAX_SENTENCES
)

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestFastSummarizeText:
    """Test cases for fast_summarize_text function."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.valid_text = ENGLISH_MEDIUM_TEXT
        self.short_text = ENGLISH_SHORT_TEXT
        self.long_text = ENGLISH_LONG_TEXT
        
    def test_valid_input_basic(self):
        """Test basic functionality with valid input."""
        with patch('utils.fast_summarize.AutoTokenizer') as mock_tokenizer, \
             patch('utils.fast_summarize.pipeline') as mock_pipeline:
            
            # Mock tokenizer
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.encode.return_value = [1, 2, 3, 4, 5]
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # Mock pipeline
            mock_pipeline_instance = Mock()
            mock_pipeline_instance.return_value = [{"summary_text": "This is a test summary."}]
            mock_pipeline.return_value = mock_pipeline_instance
            
            result = fast_summarize_text(self.valid_text, max_sentences=3)
            
            assert isinstance(result, str)
            assert len(result) > 0
            assert "test summary" in result.lower()
    
    def test_valid_input_different_models(self):
        """Test with different model configurations."""
        models_to_test = [BART_CNN_MODEL, T5_LARGE_MODEL]
        
        for model in models_to_test:
            with patch('utils.fast_summarize.AutoTokenizer') as mock_tokenizer, \
                 patch('utils.fast_summarize.pipeline') as mock_pipeline:
                
                # Mock tokenizer
                mock_tokenizer_instance = Mock()
                mock_tokenizer_instance.encode.return_value = [1, 2, 3, 4, 5]
                mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
                
                # Mock pipeline
                mock_pipeline_instance = Mock()
                mock_pipeline_instance.return_value = [{"summary_text": f"Summary for {model}"}]
                mock_pipeline.return_value = mock_pipeline_instance
                
                result = fast_summarize_text(self.valid_text, max_sentences=5, model_name=model)
                
                assert isinstance(result, str)
                assert len(result) > 0
                assert model in result or "summary" in result.lower()
    
    def test_valid_max_sentences(self):
        """Test with valid max_sentences values."""
        for max_sentences in VALID_MAX_SENTENCES:
            with patch('utils.fast_summarize.AutoTokenizer') as mock_tokenizer, \
                 patch('utils.fast_summarize.pipeline') as mock_pipeline:
                
                # Mock tokenizer
                mock_tokenizer_instance = Mock()
                mock_tokenizer_instance.encode.return_value = [1, 2, 3, 4, 5]
                mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
                
                # Mock pipeline
                mock_pipeline_instance = Mock()
                mock_pipeline_instance.return_value = [{"summary_text": "This is a test summary with multiple sentences. It contains several key points. The summary is comprehensive and informative."}]
                mock_pipeline.return_value = mock_pipeline_instance
                
                result = fast_summarize_text(self.valid_text, max_sentences=max_sentences)
                
                assert isinstance(result, str)
                assert len(result) > 0
    
    def test_empty_text_raises_error(self):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="Input text is empty"):
            fast_summarize_text(EMPTY_TEXT)
    
    def test_whitespace_only_text_raises_error(self):
        """Test that whitespace-only text raises ValueError."""
        with pytest.raises(ValueError, match="Input text is empty"):
            fast_summarize_text(WHITESPACE_ONLY_TEXT)
    
    def test_very_short_text_raises_error(self):
        """Test that very short text raises ValueError."""
        with pytest.raises(ValueError, match="too short for meaningful summarization"):
            fast_summarize_text(VERY_SHORT_TEXT)
    
    def test_invalid_max_sentences_raises_error(self):
        """Test that invalid max_sentences values raise ValueError."""
        for invalid_value in INVALID_MAX_SENTENCES:
            with pytest.raises(ValueError, match="max_sentences must be between"):
                fast_summarize_text(self.valid_text, max_sentences=invalid_value)
    
    def test_none_model_name_raises_error(self):
        """Test that None model name raises ValueError."""
        with pytest.raises(ValueError, match="Model name is required"):
            fast_summarize_text(self.valid_text, model_name=None)
    
    def test_empty_model_name_raises_error(self):
        """Test that empty model name raises ValueError."""
        with pytest.raises(ValueError, match="Model name is required"):
            fast_summarize_text(self.valid_text, model_name="")
    
    def test_model_loading_failure(self):
        """Test handling of model loading failures."""
        with patch('utils.fast_summarize.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer.from_pretrained.side_effect = Exception("Model not found")
            
            with pytest.raises(Exception, match="Failed to load model"):
                fast_summarize_text(self.valid_text)
    
    def test_tokenizer_encoding_failure(self):
        """Test handling of tokenizer encoding failures."""
        with patch('utils.fast_summarize.AutoTokenizer') as mock_tokenizer, \
             patch('utils.fast_summarize.pipeline') as mock_pipeline:
            
            # Mock tokenizer that fails on encode
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.encode.side_effect = Exception("Encoding failed")
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # Mock pipeline
            mock_pipeline_instance = Mock()
            mock_pipeline_instance.return_value = [{"summary_text": "Test summary"}]
            mock_pipeline.return_value = mock_pipeline_instance
            
            with pytest.raises(Exception, match="Failed to chunk text"):
                fast_summarize_text(self.valid_text)
    
    def test_summarizer_failure(self):
        """Test handling of summarizer failures."""
        with patch('utils.fast_summarize.AutoTokenizer') as mock_tokenizer, \
             patch('utils.fast_summarize.pipeline') as mock_pipeline:
            
            # Mock tokenizer
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.encode.return_value = [1, 2, 3, 4, 5]
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # Mock pipeline that fails
            mock_pipeline_instance = Mock()
            mock_pipeline_instance.side_effect = Exception("Summarization failed")
            mock_pipeline.return_value = mock_pipeline_instance
            
            with pytest.raises(Exception, match="Summarization failed"):
                fast_summarize_text(self.valid_text)
    
    def test_empty_summary_result(self):
        """Test handling of empty summary results."""
        with patch('utils.fast_summarize.AutoTokenizer') as mock_tokenizer, \
             patch('utils.fast_summarize.pipeline') as mock_pipeline:
            
            # Mock tokenizer
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.encode.return_value = [1, 2, 3, 4, 5]
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # Mock pipeline returning empty result
            mock_pipeline_instance = Mock()
            mock_pipeline_instance.return_value = [{"summary_text": ""}]
            mock_pipeline.return_value = mock_pipeline_instance
            
            with pytest.raises(Exception, match="No summaries were generated"):
                fast_summarize_text(self.valid_text)
    
    def test_long_text_chunking(self):
        """Test processing of long text with chunking."""
        with patch('utils.fast_summarize.AutoTokenizer') as mock_tokenizer, \
             patch('utils.fast_summarize.pipeline') as mock_pipeline:
            
            # Mock tokenizer with long token sequence
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.encode.return_value = list(range(2000))  # Long sequence
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # Mock pipeline
            mock_pipeline_instance = Mock()
            mock_pipeline_instance.return_value = [{"summary_text": "Chunked summary result"}]
            mock_pipeline.return_value = mock_pipeline_instance
            
            result = fast_summarize_text(self.long_text, max_sentences=5)
            
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_second_pass_summarization(self):
        """Test second-pass summarization for long combined results."""
        with patch('utils.fast_summarize.AutoTokenizer') as mock_tokenizer, \
             patch('utils.fast_summarize.pipeline') as mock_pipeline:
            
            # Mock tokenizer
            mock_tokenizer_instance = Mock()
            # First call returns short tokens, second call returns long tokens (triggers second pass)
            mock_tokenizer_instance.encode.side_effect = [
                [1, 2, 3, 4, 5],  # First chunk
                [1, 2, 3, 4, 5],  # Second chunk
                list(range(2000))  # Combined result (long)
            ]
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # Mock pipeline
            mock_pipeline_instance = Mock()
            mock_pipeline_instance.return_value = [{"summary_text": "Final summary result"}]
            mock_pipeline.return_value = mock_pipeline_instance
            
            result = fast_summarize_text(self.long_text, max_sentences=3)
            
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_sentence_limit_enforcement(self):
        """Test that sentence limit is properly enforced."""
        with patch('utils.fast_summarize.AutoTokenizer') as mock_tokenizer, \
             patch('utils.fast_summarize.pipeline') as mock_pipeline:
            
            # Mock tokenizer
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.encode.return_value = [1, 2, 3, 4, 5]
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # Mock pipeline returning long summary
            long_summary = ". ".join([f"This is sentence {i}" for i in range(10)])
            mock_pipeline_instance = Mock()
            mock_pipeline_instance.return_value = [{"summary_text": long_summary}]
            mock_pipeline.return_value = mock_pipeline_instance
            
            result = fast_summarize_text(self.valid_text, max_sentences=3)
            
            assert isinstance(result, str)
            # Should be limited to 3 sentences
            sentence_count = result.count('.') + (1 if not result.endswith('.') else 0)
            assert sentence_count <= 3


class TestFastSummarizeIntegration:
    """Integration tests for fast_summarize_text function."""
    
    def test_end_to_end_processing(self):
        """Test complete end-to-end processing."""
        with patch('utils.fast_summarize.AutoTokenizer') as mock_tokenizer, \
             patch('utils.fast_summarize.pipeline') as mock_pipeline:
            
            # Mock tokenizer
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.encode.return_value = [1, 2, 3, 4, 5]
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # Mock pipeline
            mock_pipeline_instance = Mock()
            mock_pipeline_instance.return_value = [{"summary_text": "Complete end-to-end test summary."}]
            mock_pipeline.return_value = mock_pipeline_instance
            
            # Test with different parameters
            result = fast_summarize_text(
                ENGLISH_MEDIUM_TEXT, 
                max_sentences=5, 
                model_name=BART_CNN_MODEL
            )
            
            assert isinstance(result, str)
            assert len(result) > 0
            assert "test summary" in result.lower()
    
    def test_performance_with_large_text(self):
        """Test performance characteristics with large text."""
        import time
        
        with patch('utils.fast_summarize.AutoTokenizer') as mock_tokenizer, \
             patch('utils.fast_summarize.pipeline') as mock_pipeline:
            
            # Mock tokenizer
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.encode.return_value = [1, 2, 3, 4, 5]
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # Mock pipeline with slight delay
            mock_pipeline_instance = Mock()
            mock_pipeline_instance.return_value = [{"summary_text": "Performance test summary."}]
            mock_pipeline.return_value = mock_pipeline_instance
            
            start_time = time.time()
            result = fast_summarize_text(ENGLISH_LONG_TEXT, max_sentences=10)
            end_time = time.time()
            
            assert isinstance(result, str)
            assert len(result) > 0
            # Should complete within reasonable time (mocked, so should be fast)
            assert (end_time - start_time) < 1.0  # 1 second max for mocked test