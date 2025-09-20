# utils/parameters.py - Model configurations and constants

from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# WORKING MODELS - Tested and verified
# =============================================================================

# English Models
BART_CNN_MODEL = "facebook/bart-large-cnn"
T5_LARGE_MODEL = "t5-large"

# Chinese Models
CHINESE_MODEL = "uer/bart-base-chinese-cluecorpussmall"

# =============================================================================
# EXPERIMENTAL MODELS - May require additional configuration
# =============================================================================

# Alternative English Models (not fully tested)
GOOGLE_MODEL = "google/pegasus-large"
FALCONSAI_MODEL = "Falconsai/Text-Summarization"
MRM_MODEL = "mrm8488/t5-base-finetuned-summarize-news"

# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

# Model metadata for UI display
MODEL_INFO: Dict[str, Dict[str, str]] = {
    BART_CNN_MODEL: {
        "name": "BART-CNN",
        "description": "High-quality CNN-style summarization",
        "best_for": "News articles, formal documents",
        "language": "English"
    },
    T5_LARGE_MODEL: {
        "name": "T5-Large",
        "description": "Google's T5 model for text-to-text tasks",
        "best_for": "Diverse content types",
        "language": "English"
    },
    CHINESE_MODEL: {
        "name": "Chinese BART",
        "description": "Chinese BART model trained on CLUE corpus",
        "best_for": "Chinese text processing",
        "language": "Chinese"
    }
}

# Available models by language
ENGLISH_MODELS: List[str] = [BART_CNN_MODEL, T5_LARGE_MODEL]
CHINESE_MODELS: List[str] = [CHINESE_MODEL]

# =============================================================================
# PROCESSING PARAMETERS
# =============================================================================

# Summarization parameters
DEFAULT_MAX_SENTENCES = 5
MIN_SENTENCES = 1
MAX_SENTENCES = 50

# Text processing limits
MIN_TEXT_LENGTH = 50
MAX_TEXT_LENGTH = 100000  # 100K characters
MAX_FILE_SIZE_MB = 10

# Token limits
DEFAULT_MAX_TOKENS = 1024
CHINESE_MAX_TOKENS = 800

# Keyword extraction
DEFAULT_KEYWORDS_COUNT = 15
MIN_KEYWORDS = 1
MAX_KEYWORDS = 100

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_model_info(model_name: str) -> Optional[Dict[str, str]]:
    """Get model information by name."""
    return MODEL_INFO.get(model_name)

def get_available_models(language: str = "English") -> List[str]:
    """Get available models for a specific language."""
    if language.lower() == "english":
        return ENGLISH_MODELS
    elif language.lower() == "chinese":
        return CHINESE_MODELS
    else:
        logger.warning(f"Unknown language: {language}")
        return ENGLISH_MODELS

def validate_model(model_name: str) -> bool:
    """Validate if a model is supported."""
    return model_name in MODEL_INFO

def get_model_display_name(model_name: str) -> str:
    """Get display name for a model."""
    info = get_model_info(model_name)
    return info.get("name", model_name) if info else model_name

# =============================================================================
# DEPRECATION WARNINGS
# =============================================================================

def _warn_experimental_models():
    """Warn about experimental models."""
    experimental_models = [GOOGLE_MODEL, FALCONSAI_MODEL, MRM_MODEL]
    logger.warning("The following models are experimental and may not work correctly:")
    for model in experimental_models:
        logger.warning(f"  - {model}")
    logger.warning("Use BART_CNN_MODEL or T5_LARGE_MODEL for reliable results.")
