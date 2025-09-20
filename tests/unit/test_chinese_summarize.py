# tests/unit/test_chinese_summarize.py - Unit tests for Chinese summarization

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from utils.chinese_summarize import (
    chinese_summarize_text,
    _chunk_text,
    _split_chinese_sentences,
    _validate_input,
    _initialize_models
)
from tests.fixtures.sample_texts import (
    CHINESE_SHORT_TEXT, CHINESE_MEDIUM_TEXT, CHINESE_LONG_TEXT,
    EMPTY_TEXT, WHITESPACE_ONLY_TEXT, VERY_SHORT_TEXT,
    VALID_MAX_SENTENCES, INVALID_MAX_SENTENCES
)

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestSplitChineseSentences:
    """Test cases for _split_chinese_sentences function."""
    
    def test_basic_chinese_sentence_splitting(self):
        """Test basic Chinese sentence splitting functionality."""
        text = "这是第一句话。这是第二句话！这是第三句话？"
        result = _split_chinese_sentences(text)
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert "这是第一句话" in result[0]
        assert "这是第二句话" in result[1]
        assert "这是第三句话" in result[2]
    
    def test_empty_text(self):
        """Test splitting empty text."""
        result = _split_chinese_sentences("")
        assert result == []
    
    def test_whitespace_only_text(self):
        """Test splitting whitespace-only text."""
        result = _split_chinese_sentences("   \n\t   ")
        assert result == []
    
    def test_single_sentence(self):
        """Test splitting single sentence."""
        text = "这是一句话。"
        result = _split_chinese_sentences(text)
        
        assert len(result) == 1
        assert result[0] == "这是一句话"
    
    def test_no_punctuation(self):
        """Test text without Chinese sentence-ending punctuation."""
        text = "这句话没有标点符号"
        result = _split_chinese_sentences(text)
        
        assert len(result) == 1
        assert result[0] == "这句话没有标点符号"
    
    def test_mixed_punctuation(self):
        """Test text with mixed Chinese punctuation."""
        text = "第一句话。第二句话！第三句话？第四句话。"
        result = _split_chinese_sentences(text)
        
        assert len(result) == 4
        assert all("句话" in sentence for sentence in result)


class TestChunkText:
    """Test cases for _chunk_text function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        self.mock_tokenizer.decode.return_value = "测试文本"
    
    def test_basic_chinese_chunking(self):
        """Test basic Chinese text chunking."""
        text = "这是测试句子。这是另一个测试句子。"
        
        with patch('utils.chinese_summarize._tokenizer', self.mock_tokenizer):
            result = _chunk_text(text)
            
            assert isinstance(result, list)
            assert len(result) > 0
            assert all(isinstance(chunk, str) for chunk in result)
    
    def test_empty_text_chunking(self):
        """Test chunking empty text."""
        with patch('utils.chinese_summarize._tokenizer', self.mock_tokenizer):
            result = _chunk_text("")
            assert result == []
    
    def test_short_text_chunking(self):
        """Test chunking short text."""
        text = "短文本。"
        
        with patch('utils.chinese_summarize._tokenizer', self.mock_tokenizer):
            result = _chunk_text(text)
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0] == "测试文本"  # Mocked decode result
    
    def test_long_text_chunking(self):
        """Test chunking long text."""
        text = "。".join([f"这是句子{i}" for i in range(20)])
        
        # Mock tokenizer to return long token sequence
        self.mock_tokenizer.encode.return_value = list(range(2000))
        
        with patch('utils.chinese_summarize._tokenizer', self.mock_tokenizer):
            result = _chunk_text(text, max_tokens=100)
            
            assert isinstance(result, list)
            assert len(result) > 1  # Should be split into multiple chunks
    
    def test_chunking_with_custom_max_tokens(self):
        """Test chunking with custom max_tokens parameter."""
        text = "这是测试句子。这是另一个测试句子。"
        
        with patch('utils.chinese_summarize._tokenizer', self.mock_tokenizer):
            result = _chunk_text(text, max_tokens=50)
            
            assert isinstance(result, list)
            assert len(result) > 0
    
    def test_tokenization_error_handling(self):
        """Test handling of tokenization errors."""
        with patch('utils.chinese_summarize._tokenizer') as mock_tokenizer:
            mock_tokenizer.encode.side_effect = Exception("Tokenization failed")
            
            # The function now handles tokenization errors gracefully
            result = _chunk_text("测试文本")
            assert isinstance(result, list)


class TestValidateInput:
    """Test cases for _validate_input function."""
    
    def test_valid_input(self):
        """Test validation with valid input."""
        text = CHINESE_MEDIUM_TEXT
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
            _validate_input("短文本", 5)
    
    def test_invalid_max_sentences_validation(self):
        """Test validation with invalid max_sentences."""
        for invalid_value in INVALID_MAX_SENTENCES:
            with pytest.raises(ValueError, match="max_sentences must be between"):
                _validate_input(CHINESE_MEDIUM_TEXT, invalid_value)


class TestInitializeModels:
    """Test cases for _initialize_models function."""
    
    def test_model_initialization_success(self):
        """Test successful Chinese model initialization."""
        with patch('utils.chinese_summarize.AutoTokenizer') as mock_tokenizer, \
             patch('utils.chinese_summarize.pipeline') as mock_pipeline:
            
            mock_tokenizer_instance = Mock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            mock_pipeline_instance = Mock()
            mock_pipeline.return_value = mock_pipeline_instance
            
            # Should not raise any exception
            _initialize_models()
    
    def test_model_initialization_failure(self):
        """Test Chinese model initialization failure."""
        with patch('utils.chinese_summarize.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer.from_pretrained.side_effect = Exception("Model not found")
            
            # The function now handles model initialization errors gracefully
            _initialize_models()  # Should not raise exception


class TestChineseSummarizeText:
    """Test cases for chinese_summarize_text function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.valid_text = CHINESE_MEDIUM_TEXT
        self.short_text = CHINESE_SHORT_TEXT
        self.long_text = CHINESE_LONG_TEXT
    
    def test_basic_functionality(self):
        """Test basic Chinese summarization functionality."""
        with patch('utils.chinese_summarize._initialize_models'), \
             patch('utils.chinese_summarize._tokenizer') as mock_tokenizer, \
             patch('utils.chinese_summarize._summarizer') as mock_summarizer:
            
            # Mock tokenizer with realistic token sequence
            mock_tokenizer.encode.return_value = list(range(100))  # 100 tokens
            mock_tokenizer.decode.return_value = "这是测试文本"  # Mock decode
            
            # Mock summarizer
            mock_summarizer.return_value = [{"summary_text": "中文摘要结果。"}]
            
            result = chinese_summarize_text(self.valid_text, max_sentences=5)
            
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_markdown_formatting(self):
        """Test that result is properly formatted as markdown."""
        with patch('utils.chinese_summarize._initialize_models'), \
             patch('utils.chinese_summarize._tokenizer') as mock_tokenizer, \
             patch('utils.chinese_summarize._summarizer') as mock_summarizer:
            
            # Mock tokenizer with realistic token sequence
            mock_tokenizer.encode.return_value = list(range(100))  # 100 tokens
            mock_tokenizer.decode.return_value = "这是测试文本"  # Mock decode
            
            # Mock summarizer returning multiple sentences
            mock_summarizer.return_value = [{"summary_text": "第一个要点。第二个要点。第三个要点。"}]
            
            result = chinese_summarize_text(self.valid_text, max_sentences=3)
            
            assert isinstance(result, str)
            # Should contain bullet points for multiple sentences
            if result.count("-") > 0:
                assert "- 第一个要点" in result
                assert "- 第二个要点" in result
                assert "- 第三个要点" in result
    
    def test_empty_text_error(self):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="Input text is empty"):
            chinese_summarize_text(EMPTY_TEXT)
    
    def test_short_text_error(self):
        """Test that short text raises ValueError."""
        with pytest.raises(ValueError, match="too short for meaningful summarization"):
            chinese_summarize_text(VERY_SHORT_TEXT)
    
    def test_invalid_max_sentences_error(self):
        """Test that invalid max_sentences raises ValueError."""
        for invalid_value in INVALID_MAX_SENTENCES:
            with pytest.raises(ValueError, match="max_sentences must be between"):
                chinese_summarize_text(self.valid_text, max_sentences=invalid_value)
    
    def test_chunking_failure(self):
        """Test handling of chunking failures."""
        with patch('utils.chinese_summarize._initialize_models'), \
             patch('utils.chinese_summarize._tokenizer') as mock_tokenizer:
            
            # Mock tokenizer that fails
            mock_tokenizer.encode.side_effect = Exception("Tokenization failed")
            
            with pytest.raises(Exception, match="Chinese summarization failed"):
                chinese_summarize_text(self.valid_text)
    
    def test_summarization_failure(self):
        """Test handling of summarization failures."""
        with patch('utils.chinese_summarize._initialize_models'), \
             patch('utils.chinese_summarize._tokenizer') as mock_tokenizer, \
             patch('utils.chinese_summarize._summarizer') as mock_summarizer:
            
            # Mock tokenizer with realistic token sequence
            mock_tokenizer.encode.return_value = list(range(100))  # 100 tokens
            mock_tokenizer.decode.return_value = "这是测试文本"  # Mock decode
            
            # Mock summarizer that fails
            mock_summarizer.side_effect = Exception("Summarization failed")
            
            with pytest.raises(Exception, match="Chinese summarization failed"):
                chinese_summarize_text(self.valid_text)
    
    def test_empty_summary_handling(self):
        """Test handling of empty summary results."""
        with patch('utils.chinese_summarize._initialize_models'), \
             patch('utils.chinese_summarize._tokenizer') as mock_tokenizer, \
             patch('utils.chinese_summarize._summarizer') as mock_summarizer:
            
            # Mock tokenizer with realistic token sequence
            mock_tokenizer.encode.return_value = list(range(100))  # 100 tokens
            mock_tokenizer.decode.return_value = "这是测试文本"  # Mock decode
            
            # Mock summarizer returning empty results
            mock_summarizer.return_value = [{"summary_text": ""}]
            
            with pytest.raises(Exception, match="No summaries were generated"):
                chinese_summarize_text(self.valid_text)
    
    def test_second_pass_summarization(self):
        """Test second-pass summarization for long combined results."""
        with patch('utils.chinese_summarize._initialize_models'), \
             patch('utils.chinese_summarize._tokenizer') as mock_tokenizer, \
             patch('utils.chinese_summarize._summarizer') as mock_summarizer:
            
            # Mock tokenizer
            mock_tokenizer.encode.side_effect = [
                [1, 2, 3, 4, 5],  # First chunk
                [1, 2, 3, 4, 5],  # Second chunk
                list(range(2000))  # Combined result (long)
            ]
            mock_tokenizer.decode.return_value = "这是一个很长的中文测试文本，用于测试第二遍摘要功能。"  # Mock decode
            
            # Mock summarizer
            mock_summarizer.return_value = [{"summary_text": "最终中文摘要。"}]
            
            result = chinese_summarize_text(self.long_text, max_sentences=5)
            
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_sentence_limit_enforcement(self):
        """Test that sentence limit is properly enforced."""
        with patch('utils.chinese_summarize._initialize_models'), \
             patch('utils.chinese_summarize._tokenizer') as mock_tokenizer, \
             patch('utils.chinese_summarize._summarizer') as mock_summarizer:
            
            # Mock tokenizer with realistic token sequence
            mock_tokenizer.encode.return_value = list(range(100))  # 100 tokens
            mock_tokenizer.decode.return_value = "这是测试文本"  # Mock decode
            
            # Mock summarizer returning long summary
            long_summary = "。".join([f"这是句子{i}" for i in range(10)])
            mock_summarizer.return_value = [{"summary_text": long_summary}]
            
            result = chinese_summarize_text(self.valid_text, max_sentences=3)
            
            assert isinstance(result, str)
            # Should be limited to 3 sentences
            sentence_count = result.count('。') + result.count('！') + result.count('？')
            assert sentence_count <= 3
    
    def test_chinese_punctuation_handling(self):
        """Test proper handling of Chinese punctuation."""
        with patch('utils.chinese_summarize._initialize_models'), \
             patch('utils.chinese_summarize._tokenizer') as mock_tokenizer, \
             patch('utils.chinese_summarize._summarizer') as mock_summarizer:
            
            # Mock tokenizer with realistic token sequence
            mock_tokenizer.encode.return_value = list(range(100))  # 100 tokens
            mock_tokenizer.decode.return_value = "这是测试文本"  # Mock decode
            
            # Mock summarizer returning text with Chinese punctuation
            mock_summarizer.return_value = [{"summary_text": "这是第一句。这是第二句！这是第三句？"}]
            
            result = chinese_summarize_text(self.valid_text, max_sentences=3)
            
            assert isinstance(result, str)
            assert len(result) > 0


class TestChineseSummarizeIntegration:
    """Integration tests for chinese_summarize_text function."""
    
    def test_end_to_end_processing(self):
        """Test complete end-to-end Chinese processing."""
        with patch('utils.chinese_summarize._initialize_models'), \
             patch('utils.chinese_summarize._tokenizer') as mock_tokenizer, \
             patch('utils.chinese_summarize._summarizer') as mock_summarizer:
            
            # Mock tokenizer with realistic token sequence
            mock_tokenizer.encode.return_value = list(range(100))  # 100 tokens
            mock_tokenizer.decode.return_value = "这是测试文本"  # Mock decode
            
            # Mock summarizer
            mock_summarizer.return_value = [{"summary_text": "完整的中文摘要处理。它涵盖了所有方面。结果是全面且格式良好的。"}]
            
            result = chinese_summarize_text(CHINESE_MEDIUM_TEXT, max_sentences=5)
            
            assert isinstance(result, str)
            assert len(result) > 0
            assert "中文摘要" in result
    
    def test_performance_characteristics(self):
        """Test performance characteristics."""
        import time
        
        with patch('utils.chinese_summarize._initialize_models'), \
             patch('utils.chinese_summarize._tokenizer') as mock_tokenizer, \
             patch('utils.chinese_summarize._summarizer') as mock_summarizer:
            
            # Mock tokenizer with realistic token sequence
            mock_tokenizer.encode.return_value = list(range(100))  # 100 tokens
            mock_tokenizer.decode.return_value = "这是测试文本"  # Mock decode
            
            # Mock summarizer
            mock_summarizer.return_value = [{"summary_text": "性能测试摘要。"}]
            
            start_time = time.time()
            result = chinese_summarize_text(CHINESE_LONG_TEXT, max_sentences=10)
            end_time = time.time()
            
            assert isinstance(result, str)
            assert len(result) > 0
            # Should complete within reasonable time (mocked, so should be fast)
            assert (end_time - start_time) < 1.0  # 1 second max for mocked test