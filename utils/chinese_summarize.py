from transformers import pipeline, AutoTokenizer
import re
import logging
from typing import List, Optional
from .parameters import CHINESE_MODEL

logger = logging.getLogger(__name__)

# Configuration constants
MODEL_NAME = CHINESE_MODEL
DEFAULT_MAX_TOKENS = 800
DEFAULT_MAX_SENTENCES = 10
MIN_TEXT_LENGTH = 50

# Global model components (lazy loading)
_tokenizer: Optional[AutoTokenizer] = None
_summarizer: Optional[pipeline] = None


def _initialize_models() -> None:
    """Initialize Chinese tokenizer and summarizer models."""
    global _tokenizer, _summarizer
    
    if _tokenizer is None or _summarizer is None:
        try:
            logger.info(f"Initializing Chinese summarization model: {MODEL_NAME}")
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
            
            _summarizer = pipeline(
                "summarization",
                model=MODEL_NAME,
                tokenizer=_tokenizer,
                device=-1,  # use CPU; change to 0 for GPU
            )
            logger.info("Chinese summarization model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Chinese summarization model: {e}")
            raise Exception(f"Chinese model initialization failed: {e}")


def _chunk_text(text: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> List[str]:
    """Split Chinese text into token-safe chunks for processing."""
    if not text or not text.strip():
        return []
    
    try:
        # Directly tokenize raw text
        token_ids = _tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            return []
        
        chunks = []
        for i in range(0, len(token_ids), max_tokens):
            chunk_ids = token_ids[i : i + max_tokens]
            chunk_text = _tokenizer.decode(chunk_ids, skip_special_tokens=True)
            if chunk_text and chunk_text.strip():
                chunks.append(chunk_text.strip())
        
        return chunks
    except Exception as e:
        logger.error(f"Error chunking Chinese text: {e}")
        raise Exception(f"Failed to chunk Chinese text: {e}")


def _split_chinese_sentences(text: str) -> List[str]:
    """Split Chinese text into sentences using Chinese punctuation."""
    if not text or not text.strip():
        return []
    
    # Split on Chinese sentence endings
    sentences = re.split(r"[。！？]", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _validate_input(text: str, max_sentences: int) -> None:
    """Validate input parameters for Chinese summarization."""
    if not text or not text.strip():
        raise ValueError("Input text is empty or contains only whitespace")
    
    if len(text.strip()) < MIN_TEXT_LENGTH:
        raise ValueError(f"Input text is too short for meaningful summarization (minimum {MIN_TEXT_LENGTH} characters)")
    
    if max_sentences < 1 or max_sentences > 50:
        raise ValueError("max_sentences must be between 1 and 50")


def chinese_summarize_text(text: str, max_sentences: int = DEFAULT_MAX_SENTENCES) -> str:
    """
    Chinese text summarization with specialized models and processing.
    
    Args:
        text: Chinese text to summarize
        max_sentences: Maximum number of sentences in summary
        
    Returns:
        Chinese summary with bullet point formatting
        
    Raises:
        ValueError: For invalid input parameters
        Exception: For processing errors
    """
    # Validate input
    _validate_input(text, max_sentences)
    
    # Initialize models if needed
    _initialize_models()
    
    try:
        # Step 1: Split into safe token-sized chunks
        chunks = _chunk_text(text)
        if not chunks:
            raise ValueError("Text could not be processed into chunks")
        
        partial_summaries = []
        
        # Step 2: Summarize each chunk separately
        for i, chunk in enumerate(chunks):
            try:
                if len(chunk.strip()) < 10:
                    logger.warning(f"Skipping Chinese chunk {i+1} - too short")
                    continue
                
                result = _summarizer(
                    chunk, 
                    max_length=150, 
                    min_length=30, 
                    truncation=True
                )
                
                if result and len(result) > 0 and "summary_text" in result[0]:
                    summary_text = result[0]["summary_text"].strip()
                    if summary_text:
                        partial_summaries.append(summary_text)
                    else:
                        logger.warning(f"Empty summary for Chinese chunk {i+1}")
                else:
                    logger.warning(f"No summary generated for Chinese chunk {i+1}")
                    
            except Exception as e:
                logger.error(f"Error summarizing Chinese chunk {i+1}: {e}")
                continue
        
        if not partial_summaries:
            raise Exception("No summaries were generated from any Chinese chunks")
        
        # Step 3: Combine summaries
        combined_summary = " ".join(partial_summaries)
        
        # Step 4: Optional second-pass summary if too long
        token_len = len(_tokenizer.encode(combined_summary, add_special_tokens=False))
        if token_len > DEFAULT_MAX_TOKENS:
            logger.info("Performing second-pass Chinese summarization")
            try:
                result = _summarizer(
                    combined_summary, 
                    max_length=150, 
                    min_length=30, 
                    truncation=True
                )
                
                if result and len(result) > 0 and "summary_text" in result[0]:
                    combined_summary = result[0]["summary_text"].strip()
                else:
                    logger.warning("Second-pass Chinese summarization failed, using original combined summary")
                    
            except Exception as e:
                logger.warning(f"Second-pass Chinese summarization failed: {e}")
        
        # Step 5: Format into Markdown bullet points (Chinese sentence split)
        sentences = _split_chinese_sentences(combined_summary)
        if not sentences:
            raise Exception("No sentences found in Chinese summary")
        
        if len(sentences) > 1:
            formatted_sentences = sentences[:max_sentences]
            md_output = "\n".join([f"- {s}" for s in formatted_sentences])
        else:
            md_output = sentences[0]
        
        return md_output
        
    except Exception as e:
        logger.error(f"Error in chinese_summarize_text: {e}")
        raise Exception(f"Chinese summarization failed: {e}")
