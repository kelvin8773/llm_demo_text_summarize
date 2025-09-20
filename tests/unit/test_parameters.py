# tests/unit/test_parameters.py - Unit tests for parameters module

import pytest
import logging
from utils.parameters import (
    BART_CNN_MODEL,
    T5_LARGE_MODEL,
    CHINESE_MODEL,
    get_model_info,
    get_available_models,
    validate_model,
    get_model_display_name,
    DEFAULT_MAX_SENTENCES,
    DEFAULT_KEYWORDS_COUNT,
    ENGLISH_MODELS,
    CHINESE_MODELS,
    MODEL_INFO
)

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestModelConstants:
    """Test cases for model constants."""
    
    def test_model_constants_exist(self):
        """Test that model constants are defined."""
        assert BART_CNN_MODEL is not None
        assert T5_LARGE_MODEL is not None
        assert CHINESE_MODEL is not None
        
        assert isinstance(BART_CNN_MODEL, str)
        assert isinstance(T5_LARGE_MODEL, str)
        assert isinstance(CHINESE_MODEL, str)
    
    def test_model_constants_values(self):
        """Test that model constants have expected values."""
        assert BART_CNN_MODEL == "facebook/bart-large-cnn"
        assert T5_LARGE_MODEL == "t5-large"
        assert CHINESE_MODEL == "uer/bart-base-chinese-cluecorpussmall"
    
    def test_default_parameters(self):
        """Test default parameter values."""
        assert DEFAULT_MAX_SENTENCES == 5
        assert DEFAULT_KEYWORDS_COUNT == 15
        
        assert isinstance(DEFAULT_MAX_SENTENCES, int)
        assert isinstance(DEFAULT_KEYWORDS_COUNT, int)
        assert DEFAULT_MAX_SENTENCES > 0
        assert DEFAULT_KEYWORDS_COUNT > 0


class TestModelLists:
    """Test cases for model lists."""
    
    def test_english_models_list(self):
        """Test English models list."""
        assert isinstance(ENGLISH_MODELS, list)
        assert len(ENGLISH_MODELS) > 0
        assert BART_CNN_MODEL in ENGLISH_MODELS
        assert T5_LARGE_MODEL in ENGLISH_MODELS
    
    def test_chinese_models_list(self):
        """Test Chinese models list."""
        assert isinstance(CHINESE_MODELS, list)
        assert len(CHINESE_MODELS) > 0
        assert CHINESE_MODEL in CHINESE_MODELS
    
    def test_model_lists_no_overlap(self):
        """Test that model lists don't overlap."""
        english_set = set(ENGLISH_MODELS)
        chinese_set = set(CHINESE_MODELS)
        assert len(english_set.intersection(chinese_set)) == 0


class TestModelInfo:
    """Test cases for model information."""
    
    def test_model_info_structure(self):
        """Test MODEL_INFO structure."""
        assert isinstance(MODEL_INFO, dict)
        assert len(MODEL_INFO) > 0
        
        # Check that all models have required keys
        for model_name, info in MODEL_INFO.items():
            assert isinstance(model_name, str)
            assert isinstance(info, dict)
            assert "name" in info
            assert "description" in info
            assert "best_for" in info
            assert "language" in info
            
            assert isinstance(info["name"], str)
            assert isinstance(info["description"], str)
            assert isinstance(info["best_for"], str)
            assert isinstance(info["language"], str)
    
    def test_model_info_coverage(self):
        """Test that MODEL_INFO covers all working models."""
        working_models = [BART_CNN_MODEL, T5_LARGE_MODEL, CHINESE_MODEL]
        
        for model in working_models:
            assert model in MODEL_INFO
            assert MODEL_INFO[model]["language"] in ["English", "Chinese"]


class TestHelperFunctions:
    """Test cases for helper functions."""
    
    def test_get_model_info_valid_model(self):
        """Test get_model_info with valid model."""
        info = get_model_info(BART_CNN_MODEL)
        
        assert info is not None
        assert isinstance(info, dict)
        assert "name" in info
        assert "description" in info
        assert "best_for" in info
        assert "language" in info
    
    def test_get_model_info_invalid_model(self):
        """Test get_model_info with invalid model."""
        info = get_model_info("invalid-model")
        assert info is None
    
    def test_get_model_info_none_input(self):
        """Test get_model_info with None input."""
        info = get_model_info(None)
        assert info is None
    
    def test_get_available_models_english(self):
        """Test get_available_models for English."""
        models = get_available_models("English")
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert BART_CNN_MODEL in models
        assert T5_LARGE_MODEL in models
        assert CHINESE_MODEL not in models
    
    def test_get_available_models_chinese(self):
        """Test get_available_models for Chinese."""
        models = get_available_models("Chinese")
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert CHINESE_MODEL in models
        assert BART_CNN_MODEL not in models
        assert T5_LARGE_MODEL not in models
    
    def test_get_available_models_case_insensitive(self):
        """Test get_available_models is case insensitive."""
        models_upper = get_available_models("ENGLISH")
        models_lower = get_available_models("english")
        models_mixed = get_available_models("English")
        
        assert models_upper == models_lower == models_mixed
    
    def test_get_available_models_unknown_language(self):
        """Test get_available_models with unknown language."""
        models = get_available_models("Unknown")
        
        # Should default to English models
        assert isinstance(models, list)
        assert len(models) > 0
        assert BART_CNN_MODEL in models
    
    def test_validate_model_valid_models(self):
        """Test validate_model with valid models."""
        valid_models = [BART_CNN_MODEL, T5_LARGE_MODEL, CHINESE_MODEL]
        
        for model in valid_models:
            assert validate_model(model) is True
    
    def test_validate_model_invalid_model(self):
        """Test validate_model with invalid model."""
        assert validate_model("invalid-model") is False
        assert validate_model("") is False
        assert validate_model(None) is False
    
    def test_get_model_display_name_valid_model(self):
        """Test get_model_display_name with valid model."""
        display_name = get_model_display_name(BART_CNN_MODEL)
        
        assert isinstance(display_name, str)
        assert len(display_name) > 0
        assert display_name != BART_CNN_MODEL  # Should be different from model name
    
    def test_get_model_display_name_invalid_model(self):
        """Test get_model_display_name with invalid model."""
        display_name = get_model_display_name("invalid-model")
        
        # Should return the input model name as fallback
        assert display_name == "invalid-model"


class TestParameterValidation:
    """Test cases for parameter validation."""
    
    def test_sentence_limits(self):
        """Test sentence limit constants."""
        from utils.parameters import MIN_SENTENCES, MAX_SENTENCES
        
        assert MIN_SENTENCES == 1
        assert MAX_SENTENCES == 50
        assert MIN_SENTENCES < DEFAULT_MAX_SENTENCES < MAX_SENTENCES
    
    def test_text_length_limits(self):
        """Test text length limit constants."""
        from utils.parameters import MIN_TEXT_LENGTH, MAX_TEXT_LENGTH
        
        assert MIN_TEXT_LENGTH == 50
        assert MAX_TEXT_LENGTH == 100000
        assert MIN_TEXT_LENGTH > 0
        assert MAX_TEXT_LENGTH > MIN_TEXT_LENGTH
    
    def test_file_size_limits(self):
        """Test file size limit constants."""
        from utils.parameters import MAX_FILE_SIZE_MB
        
        assert MAX_FILE_SIZE_MB == 10
        assert MAX_FILE_SIZE_MB > 0
    
    def test_token_limits(self):
        """Test token limit constants."""
        from utils.parameters import DEFAULT_MAX_TOKENS, CHINESE_MAX_TOKENS
        
        assert DEFAULT_MAX_TOKENS == 1024
        assert CHINESE_MAX_TOKENS == 800
        assert DEFAULT_MAX_TOKENS > CHINESE_MAX_TOKENS
        assert DEFAULT_MAX_TOKENS > 0
        assert CHINESE_MAX_TOKENS > 0
    
    def test_keyword_limits(self):
        """Test keyword limit constants."""
        from utils.parameters import MIN_KEYWORDS, MAX_KEYWORDS
        
        assert MIN_KEYWORDS == 1
        assert MAX_KEYWORDS == 100
        assert MIN_KEYWORDS < DEFAULT_KEYWORDS_COUNT < MAX_KEYWORDS


class TestParameterIntegration:
    """Integration tests for parameters module."""
    
    def test_all_models_have_info(self):
        """Test that all working models have information."""
        working_models = [BART_CNN_MODEL, T5_LARGE_MODEL, CHINESE_MODEL]
        
        for model in working_models:
            info = get_model_info(model)
            assert info is not None
            
            # Test that we can get display name
            display_name = get_model_display_name(model)
            assert isinstance(display_name, str)
            assert len(display_name) > 0
            
            # Test that model is valid
            assert validate_model(model) is True
    
    def test_model_language_consistency(self):
        """Test that model language assignments are consistent."""
        english_models = get_available_models("English")
        chinese_models = get_available_models("Chinese")
        
        # Check English models
        for model in english_models:
            info = get_model_info(model)
            assert info["language"] == "English"
        
        # Check Chinese models
        for model in chinese_models:
            info = get_model_info(model)
            assert info["language"] == "Chinese"
    
    def test_default_parameters_consistency(self):
        """Test that default parameters are consistent with limits."""
        from utils.parameters import (
            MIN_SENTENCES, MAX_SENTENCES,
            MIN_KEYWORDS, MAX_KEYWORDS
        )
        
        # Default max sentences should be within limits
        assert MIN_SENTENCES <= DEFAULT_MAX_SENTENCES <= MAX_SENTENCES
        
        # Default keywords count should be within limits
        assert MIN_KEYWORDS <= DEFAULT_KEYWORDS_COUNT <= MAX_KEYWORDS