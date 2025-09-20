# tests/unit/test_chinese_insights.py - Unit tests for Chinese insights

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
import matplotlib.pyplot as plt
from utils.chinese_insights import (
    extract_chinese_keywords,
    plot_chinese_keywords,
    _jieba_tokenizer,
    _validate_input,
    _initialize_chinese_font
)
from tests.fixtures.sample_texts import (
    CHINESE_SHORT_TEXT, CHINESE_MEDIUM_TEXT, CHINESE_LONG_TEXT,
    EMPTY_TEXT, WHITESPACE_ONLY_TEXT, VERY_SHORT_TEXT,
    VALID_TOP_N, INVALID_TOP_N, EXPECTED_CHINESE_KEYWORDS
)

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestJiebaTokenizer:
    """Test cases for _jieba_tokenizer function."""
    
    def test_basic_chinese_tokenization(self):
        """Test basic Chinese tokenization functionality."""
        text = "人工智能正在改变世界。机器学习算法处理数据。"
        
        with patch('utils.chinese_insights.jieba.cut') as mock_cut:
            mock_cut.return_value = ["人工智能", "正在", "改变", "世界", "机器学习", "算法", "处理", "数据"]
            
            result = _jieba_tokenizer(text)
            
            assert isinstance(result, list)
            assert "人工智能" in result
            assert "机器学习" in result
            assert "数据" in result
            # Stopwords should be filtered out
            assert "正在" not in result
    
    def test_empty_text_tokenization(self):
        """Test tokenization with empty text."""
        result = _jieba_tokenizer("")
        assert result == []
    
    def test_whitespace_text_tokenization(self):
        """Test tokenization with whitespace-only text."""
        result = _jieba_tokenizer("   \n\t   ")
        assert result == []
    
    def test_stopword_filtering(self):
        """Test that stopwords are properly filtered."""
        text = "这是一个测试的句子。"
        
        with patch('utils.chinese_insights.jieba.cut') as mock_cut:
            mock_cut.return_value = ["这", "是", "一个", "测试", "的", "句子"]
            
            result = _jieba_tokenizer(text)
            
            # Stopwords should be filtered out
            assert "这" not in result
            assert "是" not in result
            assert "的" not in result
            # Non-stopwords should remain
            assert "测试" in result
            assert "句子" in result
    
    def test_blocklist_filtering(self):
        """Test that blocklist words are filtered."""
        text = "公司使用数据业务系统。"
        
        with patch('utils.chinese_insights.jieba.cut') as mock_cut:
            mock_cut.return_value = ["公司", "使用", "数据", "业务", "系统"]
            
            result = _jieba_tokenizer(text)
            
            # Blocklist words should be filtered out
            assert "公司" not in result
            assert "数据" not in result
            assert "业务" not in result
            assert "使用" not in result
            assert "系统" not in result
    
    def test_single_character_filtering(self):
        """Test that single characters are filtered (except alphanumeric)."""
        text = "A 1 中 文 测试"
        
        with patch('utils.chinese_insights.jieba.cut') as mock_cut:
            mock_cut.return_value = ["A", "1", "中", "文", "测试"]
            
            result = _jieba_tokenizer(text)
            
            # Single alphanumeric characters should remain
            assert "A" in result
            assert "1" in result
            # Single Chinese characters should be filtered
            assert "中" not in result
            assert "文" not in result
            # Multi-character words should remain
            assert "测试" in result
    
    def test_tokenization_error_handling(self):
        """Test handling of tokenization errors."""
        with patch('utils.chinese_insights.jieba.cut') as mock_cut:
            mock_cut.side_effect = Exception("Tokenization failed")
            
            result = _jieba_tokenizer("测试文本")
            
            assert result == []


class TestValidateInput:
    """Test cases for _validate_input function."""
    
    def test_valid_input(self):
        """Test validation with valid input."""
        text = CHINESE_MEDIUM_TEXT
        top_n = 10
        
        # Should not raise any exception
        _validate_input(text, top_n)
    
    def test_empty_text_validation(self):
        """Test validation with empty text."""
        with pytest.raises(ValueError, match="Input text is empty"):
            _validate_input("", 10)
    
    def test_whitespace_text_validation(self):
        """Test validation with whitespace-only text."""
        with pytest.raises(ValueError, match="Input text is empty"):
            _validate_input("   \n\t   ", 10)
    
    def test_short_text_validation(self):
        """Test validation with too short text."""
        with pytest.raises(ValueError, match="too short for meaningful keyword extraction"):
            _validate_input("短文本", 10)
    
    def test_invalid_top_n_validation(self):
        """Test validation with invalid top_n values."""
        for invalid_value in INVALID_TOP_N:
            with pytest.raises(ValueError, match="top_n must be between"):
                _validate_input(CHINESE_MEDIUM_TEXT, invalid_value)


class TestExtractChineseKeywords:
    """Test cases for extract_chinese_keywords function."""
    
    def test_basic_chinese_keyword_extraction(self):
        """Test basic Chinese keyword extraction functionality."""
        result = extract_chinese_keywords(CHINESE_MEDIUM_TEXT, top_n=10)
        
        assert isinstance(result, list)
        assert len(result) <= 10
        assert all(isinstance(keyword, str) for keyword in result)
        assert len(result) > 0
    
    def test_chinese_keyword_extraction_with_different_top_n(self):
        """Test Chinese keyword extraction with different top_n values."""
        for top_n in VALID_TOP_N:
            result = extract_chinese_keywords(CHINESE_MEDIUM_TEXT, top_n=top_n)
            
            assert isinstance(result, list)
            assert len(result) <= top_n
            assert all(isinstance(keyword, str) for keyword in result)
    
    def test_empty_text_error(self):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="Input text is empty"):
            extract_chinese_keywords(EMPTY_TEXT)
    
    def test_short_text_error(self):
        """Test that short text raises ValueError."""
        with pytest.raises(ValueError, match="too short for meaningful keyword extraction"):
            extract_chinese_keywords(VERY_SHORT_TEXT)
    
    def test_invalid_top_n_error(self):
        """Test that invalid top_n raises ValueError."""
        for invalid_value in INVALID_TOP_N:
            with pytest.raises(ValueError, match="top_n must be between"):
                extract_chinese_keywords(CHINESE_MEDIUM_TEXT, top_n=invalid_value)
    
    def test_tfidf_vectorization_error(self):
        """Test handling of TF-IDF vectorization errors."""
        with patch('utils.chinese_insights.TfidfVectorizer') as mock_vectorizer:
            mock_vectorizer.side_effect = Exception("Vectorization failed")
            
            with pytest.raises(Exception, match="Chinese keyword extraction failed"):
                extract_chinese_keywords(CHINESE_MEDIUM_TEXT)
    
    def test_jieba_tokenization_error(self):
        """Test handling of jieba tokenization errors."""
        with patch('utils.chinese_insights._jieba_tokenizer') as mock_tokenizer:
            mock_tokenizer.side_effect = Exception("Tokenization failed")
            
            with pytest.raises(Exception, match="Chinese keyword extraction failed"):
                extract_chinese_keywords(CHINESE_MEDIUM_TEXT)


class TestPlotChineseKeywords:
    """Test cases for plot_chinese_keywords function."""
    
    def test_basic_chinese_plotting(self):
        """Test basic Chinese keyword plotting functionality."""
        keywords = ["人工智能", "机器学习", "数据处理"]
        
        with patch('utils.chinese_insights._initialize_chinese_font'), \
             patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.tight_layout'):
            
            mock_fig = Mock()
            mock_ax = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            result = plot_chinese_keywords(keywords)
            
            assert result == mock_fig
            mock_subplots.assert_called_once()
    
    def test_empty_keywords_error(self):
        """Test that empty keywords list raises ValueError."""
        with pytest.raises(ValueError, match="Keywords list is empty"):
            plot_chinese_keywords([])
    
    def test_chinese_plotting_error_handling(self):
        """Test handling of Chinese plotting errors."""
        keywords = ["人工智能", "机器学习"]
        
        with patch('utils.chinese_insights._initialize_chinese_font'), \
             patch('matplotlib.pyplot.subplots') as mock_subplots:
            
            mock_subplots.side_effect = Exception("Plotting failed")
            
            with pytest.raises(Exception, match="Chinese plotting failed"):
                plot_chinese_keywords(keywords)
    
    def test_chinese_plot_with_different_keyword_counts(self):
        """Test Chinese plotting with different numbers of keywords."""
        for count in [1, 5, 10, 20]:
            keywords = [f"关键词{i}" for i in range(count)]
            
            with patch('utils.chinese_insights._initialize_chinese_font'), \
                 patch('matplotlib.pyplot.subplots') as mock_subplots, \
                 patch('matplotlib.pyplot.tight_layout'):
                
                mock_fig = Mock()
                mock_ax = Mock()
                mock_subplots.return_value = (mock_fig, mock_ax)
                
                result = plot_chinese_keywords(keywords)
                
                assert result == mock_fig
                mock_subplots.assert_called_once()


class TestInitializeChineseFont:
    """Test cases for _initialize_chinese_font function."""
    
    def test_chinese_font_initialization_success(self):
        """Test successful Chinese font initialization."""
        with patch('utils.chinese_insights.os.path.exists') as mock_exists, \
             patch('utils.chinese_insights.requests.get') as mock_get, \
             patch('builtins.open', mock_open()), \
             patch('utils.chinese_insights.fm.FontProperties') as mock_font:
            
            mock_exists.return_value = True  # Font already exists
            mock_font_instance = Mock()
            mock_font.return_value = mock_font_instance
            
            # Should not raise any exception
            _initialize_chinese_font()
            
            mock_font.assert_called_once()
    
    def test_chinese_font_download(self):
        """Test Chinese font download when not cached."""
        with patch('utils.chinese_insights.os.path.exists') as mock_exists, \
             patch('utils.chinese_insights.requests.get') as mock_get, \
             patch('builtins.open', mock_open()), \
             patch('utils.chinese_insights.fm.FontProperties') as mock_font:
            
            mock_exists.return_value = False  # Font doesn't exist
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.content = b"font content"
            mock_get.return_value = mock_response
            
            mock_font_instance = Mock()
            mock_font.return_value = mock_font_instance
            
            # Should not raise any exception
            _initialize_chinese_font()
            
            mock_get.assert_called_once()
            mock_font.assert_called_once()
    
    def test_chinese_font_download_failure(self):
        """Test Chinese font download failure."""
        with patch('utils.chinese_insights.os.path.exists') as mock_exists, \
             patch('utils.chinese_insights.requests.get') as mock_get, \
             patch('utils.chinese_insights.fm.FontProperties') as mock_font:
            
            mock_exists.return_value = False  # Font doesn't exist
            mock_get.side_effect = Exception("Download failed")
            
            mock_font_instance = Mock()
            mock_font.return_value = mock_font_instance
            
            # Should not raise exception, should use fallback
            _initialize_chinese_font()
            
            mock_font.assert_called_once()


def mock_open():
    """Mock open function for file operations."""
    from unittest.mock import mock_open as _mock_open
    return _mock_open()


class TestChineseInsightsIntegration:
    """Integration tests for Chinese insights functions."""
    
    def test_end_to_end_chinese_keyword_extraction(self):
        """Test complete end-to-end Chinese keyword extraction."""
        result = extract_chinese_keywords(CHINESE_MEDIUM_TEXT, top_n=15)
        
        assert isinstance(result, list)
        assert len(result) <= 15
        assert len(result) > 0
        assert all(isinstance(keyword, str) for keyword in result)
        assert all(len(keyword) > 0 for keyword in result)
    
    def test_end_to_end_chinese_plotting(self):
        """Test complete end-to-end Chinese plotting."""
        keywords = extract_chinese_keywords(CHINESE_MEDIUM_TEXT, top_n=10)
        
        with patch('utils.chinese_insights._initialize_chinese_font'), \
             patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.tight_layout'):
            
            mock_fig = Mock()
            mock_ax = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            result = plot_chinese_keywords(keywords)
            
            assert result == mock_fig
    
    def test_performance_with_large_chinese_text(self):
        """Test performance characteristics with large Chinese text."""
        import time
        
        start_time = time.time()
        result = extract_chinese_keywords(CHINESE_LONG_TEXT, top_n=20)
        end_time = time.time()
        
        assert isinstance(result, list)
        assert len(result) <= 20
        assert len(result) > 0
        # Should complete within reasonable time
        assert (end_time - start_time) < 5.0  # 5 seconds max
    
    def test_chinese_keyword_quality(self):
        """Test quality of extracted Chinese keywords."""
        result = extract_chinese_keywords(CHINESE_MEDIUM_TEXT, top_n=10)
        
        # Keywords should be meaningful Chinese words/phrases
        assert all(len(keyword) > 0 for keyword in result)
        
        # Should contain some expected keywords
        text_lower = CHINESE_MEDIUM_TEXT.lower()
        found_keywords = sum(1 for keyword in result if keyword in CHINESE_MEDIUM_TEXT)
        assert found_keywords > 0  # At least some keywords should be found in text
    
    def test_chinese_tokenization_quality(self):
        """Test quality of Chinese tokenization."""
        text = "人工智能正在改变世界。机器学习算法处理大量数据。"
        
        with patch('utils.chinese_insights.jieba.cut') as mock_cut:
            mock_cut.return_value = ["人工智能", "正在", "改变", "世界", "机器学习", "算法", "处理", "大量", "数据"]
            
            result = _jieba_tokenizer(text)
            
            # Should contain meaningful terms
            assert "人工智能" in result
            assert "机器学习" in result
            assert "数据" in result
            
            # Should filter out stopwords
            assert "正在" not in result
    
    def test_chinese_font_handling(self):
        """Test Chinese font handling in plotting."""
        keywords = ["人工智能", "机器学习", "深度学习"]
        
        # Test with successful font initialization
        with patch('utils.chinese_insights._initialize_chinese_font') as mock_init, \
             patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.tight_layout'):
            
            mock_fig = Mock()
            mock_ax = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            result = plot_chinese_keywords(keywords)
            
            assert result == mock_fig
            mock_init.assert_called_once()
        
        # Test with font initialization failure (should still work with fallback)
        with patch('utils.chinese_insights._initialize_chinese_font') as mock_init, \
             patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.tight_layout'):
            
            mock_init.side_effect = Exception("Font initialization failed")
            
            mock_fig = Mock()
            mock_ax = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            result = plot_chinese_keywords(keywords)
            
            assert result == mock_fig