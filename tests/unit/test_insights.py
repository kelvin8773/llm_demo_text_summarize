# tests/unit/test_insights.py - Unit tests for English insights

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
import matplotlib.pyplot as plt
from utils.insights import (
    extract_keywords,
    extract_keywords_phrases,
    plot_keywords,
    _extract_noun_chunks,
    _validate_input,
    _initialize_spacy
)
from tests.fixtures.sample_texts import (
    ENGLISH_SHORT_TEXT, ENGLISH_MEDIUM_TEXT, ENGLISH_LONG_TEXT,
    EMPTY_TEXT, WHITESPACE_ONLY_TEXT, VERY_SHORT_TEXT,
    VALID_TOP_N, INVALID_TOP_N, EXPECTED_ENGLISH_KEYWORDS
)

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestExtractNounChunks:
    """Test cases for _extract_noun_chunks function."""
    
    def test_basic_noun_chunk_extraction(self):
        """Test basic noun chunk extraction."""
        text = "The artificial intelligence system processes data efficiently."
        
        with patch('utils.insights._initialize_spacy'), \
             patch('utils.insights._nlp') as mock_nlp:
            
            # Mock spaCy document
            mock_doc = Mock()
            mock_chunk1 = Mock()
            mock_chunk1.text = "artificial intelligence system"
            mock_chunk2 = Mock()
            mock_chunk2.text = "data"
            
            mock_doc.noun_chunks = [mock_chunk1, mock_chunk2]
            mock_nlp.return_value = mock_doc
            
            result = _extract_noun_chunks(text)
            
            assert isinstance(result, set)
            assert "artificial intelligence system" in result
            assert "data" in result
    
    def test_empty_text_noun_chunks(self):
        """Test noun chunk extraction with empty text."""
        with patch('utils.insights._initialize_spacy'), \
             patch('utils.insights._nlp') as mock_nlp:
            
            mock_doc = Mock()
            mock_doc.noun_chunks = []
            mock_nlp.return_value = mock_doc
            
            result = _extract_noun_chunks("")
            
            assert isinstance(result, set)
            assert len(result) == 0
    
    def test_spacy_initialization_error(self):
        """Test handling of spaCy initialization errors."""
        with patch('utils.insights._initialize_spacy') as mock_init:
            mock_init.side_effect = Exception("spaCy initialization failed")
            
            result = _extract_noun_chunks("test text")
            
            assert isinstance(result, set)
            assert len(result) == 0


class TestValidateInput:
    """Test cases for _validate_input function."""
    
    def test_valid_input(self):
        """Test validation with valid input."""
        text = ENGLISH_MEDIUM_TEXT
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
            _validate_input("Short text", 10)
    
    def test_invalid_top_n_validation(self):
        """Test validation with invalid top_n values."""
        for invalid_value in INVALID_TOP_N:
            with pytest.raises(ValueError, match="top_n must be between"):
                _validate_input(ENGLISH_MEDIUM_TEXT, invalid_value)


class TestExtractKeywords:
    """Test cases for extract_keywords function."""
    
    def test_basic_keyword_extraction(self):
        """Test basic keyword extraction functionality."""
        result = extract_keywords(ENGLISH_MEDIUM_TEXT, top_n=10)
        
        assert isinstance(result, list)
        assert len(result) <= 10
        assert all(isinstance(keyword, str) for keyword in result)
        assert len(result) > 0
    
    def test_keyword_extraction_with_different_top_n(self):
        """Test keyword extraction with different top_n values."""
        for top_n in VALID_TOP_N:
            result = extract_keywords(ENGLISH_MEDIUM_TEXT, top_n=top_n)
            
            assert isinstance(result, list)
            assert len(result) <= top_n
            assert all(isinstance(keyword, str) for keyword in result)
    
    def test_empty_text_error(self):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="Input text is empty"):
            extract_keywords(EMPTY_TEXT)
    
    def test_short_text_error(self):
        """Test that short text raises ValueError."""
        with pytest.raises(ValueError, match="too short for meaningful keyword extraction"):
            extract_keywords(VERY_SHORT_TEXT)
    
    def test_invalid_top_n_error(self):
        """Test that invalid top_n raises ValueError."""
        for invalid_value in INVALID_TOP_N:
            with pytest.raises(ValueError, match="top_n must be between"):
                extract_keywords(ENGLISH_MEDIUM_TEXT, top_n=invalid_value)
    
    def test_tfidf_vectorization_error(self):
        """Test handling of TF-IDF vectorization errors."""
        with patch('utils.insights.TfidfVectorizer') as mock_vectorizer:
            mock_vectorizer.side_effect = Exception("Vectorization failed")
            
            with pytest.raises(Exception, match="Keyword extraction failed"):
                extract_keywords(ENGLISH_MEDIUM_TEXT)


class TestExtractKeywordsPhrases:
    """Test cases for extract_keywords_phrases function."""
    
    def test_basic_phrase_extraction(self):
        """Test basic phrase extraction functionality."""
        with patch('utils.insights._extract_noun_chunks') as mock_chunks:
            mock_chunks.return_value = {"artificial intelligence", "machine learning", "data processing"}
            
            result = extract_keywords_phrases(ENGLISH_MEDIUM_TEXT, top_n=10)
            
            assert isinstance(result, list)
            assert len(result) <= 10
            assert all(isinstance(phrase, str) for phrase in result)
    
    def test_phrase_extraction_with_different_top_n(self):
        """Test phrase extraction with different top_n values."""
        with patch('utils.insights._extract_noun_chunks') as mock_chunks:
            mock_chunks.return_value = {"artificial intelligence", "machine learning", "data processing"}
            
            for top_n in VALID_TOP_N:
                result = extract_keywords_phrases(ENGLISH_MEDIUM_TEXT, top_n=top_n)
                
                assert isinstance(result, list)
                assert len(result) <= top_n
                assert all(isinstance(phrase, str) for phrase in result)
    
    def test_no_noun_chunks_fallback(self):
        """Test fallback to basic keyword extraction when no noun chunks found."""
        with patch('utils.insights._extract_noun_chunks') as mock_chunks, \
             patch('utils.insights.extract_keywords') as mock_extract:
            
            mock_chunks.return_value = set()  # No noun chunks
            mock_extract.return_value = ["keyword1", "keyword2", "keyword3"]
            
            result = extract_keywords_phrases(ENGLISH_MEDIUM_TEXT, top_n=10)
            
            assert isinstance(result, list)
            assert result == ["keyword1", "keyword2", "keyword3"]
            mock_extract.assert_called_once()
    
    def test_empty_phrase_results_fallback(self):
        """Test fallback when phrase extraction returns empty results."""
        with patch('utils.insights._extract_noun_chunks') as mock_chunks, \
             patch('utils.insights.extract_keywords') as mock_extract, \
             patch('utils.insights.TfidfVectorizer') as mock_vectorizer:
            
            mock_chunks.return_value = {"artificial intelligence"}
            mock_extract.return_value = ["keyword1", "keyword2"]
            
            # Mock vectorizer to return empty results
            mock_vectorizer_instance = Mock()
            mock_vectorizer_instance.fit_transform.return_value = Mock()
            mock_vectorizer_instance.get_feature_names_out.return_value = []
            mock_vectorizer.return_value = mock_vectorizer_instance
            
            result = extract_keywords_phrases(ENGLISH_MEDIUM_TEXT, top_n=10)
            
            assert isinstance(result, list)
            assert result == ["keyword1", "keyword2"]
            mock_extract.assert_called_once()
    
    def test_empty_text_error(self):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="Input text is empty"):
            extract_keywords_phrases(EMPTY_TEXT)
    
    def test_short_text_error(self):
        """Test that short text raises ValueError."""
        with pytest.raises(ValueError, match="too short for meaningful keyword extraction"):
            extract_keywords_phrases(VERY_SHORT_TEXT)
    
    def test_invalid_top_n_error(self):
        """Test that invalid top_n raises ValueError."""
        for invalid_value in INVALID_TOP_N:
            with pytest.raises(ValueError, match="top_n must be between"):
                extract_keywords_phrases(ENGLISH_MEDIUM_TEXT, top_n=invalid_value)
    
    def test_phrase_extraction_error(self):
        """Test handling of phrase extraction errors."""
        with patch('utils.insights._extract_noun_chunks') as mock_chunks, \
             patch('utils.insights.TfidfVectorizer') as mock_vectorizer:
            
            mock_chunks.return_value = {"artificial intelligence"}
            mock_vectorizer.side_effect = Exception("Vectorization failed")
            
            with pytest.raises(Exception, match="Phrase extraction failed"):
                extract_keywords_phrases(ENGLISH_MEDIUM_TEXT)


class TestPlotKeywords:
    """Test cases for plot_keywords function."""
    
    def test_basic_plotting(self):
        """Test basic keyword plotting functionality."""
        keywords = ["artificial intelligence", "machine learning", "data processing"]
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.tight_layout'):
            
            mock_fig = Mock()
            mock_ax = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            result = plot_keywords(keywords)
            
            assert result == mock_fig
            mock_subplots.assert_called_once()
    
    def test_empty_keywords_error(self):
        """Test that empty keywords list raises ValueError."""
        with pytest.raises(ValueError, match="Keywords list is empty"):
            plot_keywords([])
    
    def test_plotting_error_handling(self):
        """Test handling of plotting errors."""
        keywords = ["artificial intelligence", "machine learning"]
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_subplots.side_effect = Exception("Plotting failed")
            
            with pytest.raises(Exception, match="Plotting failed"):
                plot_keywords(keywords)
    
    def test_plot_with_different_keyword_counts(self):
        """Test plotting with different numbers of keywords."""
        for count in [1, 5, 10, 20]:
            keywords = [f"keyword{i}" for i in range(count)]
            
            with patch('matplotlib.pyplot.subplots') as mock_subplots, \
                 patch('matplotlib.pyplot.tight_layout'):
                
                mock_fig = Mock()
                mock_ax = Mock()
                mock_subplots.return_value = (mock_fig, mock_ax)
                
                result = plot_keywords(keywords)
                
                assert result == mock_fig
                mock_subplots.assert_called_once()


class TestInitializeSpacy:
    """Test cases for _initialize_spacy function."""
    
    def test_spacy_initialization_success(self):
        """Test successful spaCy initialization."""
        with patch('utils.insights.spacy.load') as mock_load:
            mock_nlp = Mock()
            mock_load.return_value = mock_nlp
            
            # Should not raise any exception
            _initialize_spacy()
            
            mock_load.assert_called_once_with("en_core_web_sm")
    
    def test_spacy_initialization_failure(self):
        """Test spaCy initialization failure."""
        with patch('utils.insights.spacy.load') as mock_load:
            mock_load.side_effect = OSError("Model not found")
            
            with pytest.raises(Exception, match="spaCy model not found"):
                _initialize_spacy()
    
    def test_spacy_initialization_general_error(self):
        """Test spaCy initialization with general error."""
        with patch('utils.insights.spacy.load') as mock_load:
            mock_load.side_effect = Exception("General error")
            
            with pytest.raises(Exception, match="spaCy initialization failed"):
                _initialize_spacy()


class TestInsightsIntegration:
    """Integration tests for insights functions."""
    
    def test_end_to_end_keyword_extraction(self):
        """Test complete end-to-end keyword extraction."""
        result = extract_keywords(ENGLISH_MEDIUM_TEXT, top_n=15)
        
        assert isinstance(result, list)
        assert len(result) <= 15
        assert len(result) > 0
        assert all(isinstance(keyword, str) for keyword in result)
        assert all(len(keyword) > 0 for keyword in result)
    
    def test_end_to_end_phrase_extraction(self):
        """Test complete end-to-end phrase extraction."""
        result = extract_keywords_phrases(ENGLISH_MEDIUM_TEXT, top_n=15)
        
        assert isinstance(result, list)
        assert len(result) <= 15
        assert len(result) > 0
        assert all(isinstance(phrase, str) for phrase in result)
        assert all(len(phrase) > 0 for phrase in result)
    
    def test_end_to_end_plotting(self):
        """Test complete end-to-end plotting."""
        keywords = extract_keywords(ENGLISH_MEDIUM_TEXT, top_n=10)
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.tight_layout'):
            
            mock_fig = Mock()
            mock_ax = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            result = plot_keywords(keywords)
            
            assert result == mock_fig
    
    def test_performance_with_large_text(self):
        """Test performance characteristics with large text."""
        import time
        
        start_time = time.time()
        result = extract_keywords(ENGLISH_LONG_TEXT, top_n=20)
        end_time = time.time()
        
        assert isinstance(result, list)
        assert len(result) <= 20
        assert len(result) > 0
        # Should complete within reasonable time
        assert (end_time - start_time) < 5.0  # 5 seconds max
    
    def test_keyword_quality(self):
        """Test quality of extracted keywords."""
        result = extract_keywords(ENGLISH_MEDIUM_TEXT, top_n=10)
        
        # Keywords should be meaningful words/phrases
        assert all(len(keyword) > 1 for keyword in result)
        assert all(keyword.isalpha() or ' ' in keyword for keyword in result)
        
        # Should contain some expected keywords
        text_lower = ENGLISH_MEDIUM_TEXT.lower()
        found_keywords = sum(1 for keyword in result if keyword.lower() in text_lower)
        assert found_keywords > 0  # At least some keywords should be found in text