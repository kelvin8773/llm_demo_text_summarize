from transformers import pipeline, AutoTokenizer
import re
import logging
from typing import List, Optional
from .parameters import BART_CNN_MODEL
from .performance import (
    cached_model_loader, 
    performance_timer, 
    memory_aware,
    optimize_text_chunking,
    model_cache,
    performance_monitor
)

logger = logging.getLogger(__name__)

# Configuration constants
MODEL_NAME = BART_CNN_MODEL
DEFAULT_MAX_TOKENS = 1024
DEFAULT_MAX_SENTENCES = 10
MIN_TEXT_LENGTH = 50

# Global model components (lazy loading)
_tokenizer: Optional[AutoTokenizer] = None
_summarizer: Optional[pipeline] = None
MAX_INPUT_TOKENS = DEFAULT_MAX_TOKENS


@cached_model_loader(lambda: "enhance_tokenizer")
@performance_timer("initialize_enhance_tokenizer")
def _initialize_tokenizer() -> AutoTokenizer:
    """Initialize tokenizer with caching."""
    logger.info(f"Initializing enhanced tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    global MAX_INPUT_TOKENS
    MAX_INPUT_TOKENS = min(tokenizer.model_max_length, DEFAULT_MAX_TOKENS)
    return tokenizer


@cached_model_loader(lambda: "enhance_summarizer")
@performance_timer("initialize_enhance_summarizer")
def _initialize_summarizer() -> pipeline:
    """Initialize summarizer with caching."""
    logger.info(f"Initializing enhanced summarizer: {MODEL_NAME}")
    tokenizer = _initialize_tokenizer()
    return pipeline(
        "summarization", 
        model=MODEL_NAME, 
        tokenizer=tokenizer, 
        device=-1
    )


def _initialize_models() -> None:
    """Initialize tokenizer and summarizer models."""
    global _tokenizer, _summarizer
    
    if _tokenizer is None:
        _tokenizer = _initialize_tokenizer()
    
    if _summarizer is None:
        _summarizer = _initialize_summarizer()


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using regex patterns."""
    if not text or not text.strip():
        return []
    
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


@performance_timer("chunk_text")
def _chunk_text(text: str, max_tokens: int = None) -> List[str]:
    """Split text into token-safe chunks for processing."""
    if max_tokens is None:
        max_tokens = MAX_INPUT_TOKENS - 32
    
    if not text or not text.strip():
        return []
    
    # Use optimized chunking if tokenizer is available
    if _tokenizer is not None:
        return optimize_text_chunking(text, max_tokens, _tokenizer)
    
    # Fallback to sentence-based chunking
    sentences = _split_sentences(text)
    if not sentences:
        return []
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        try:
            # Rough estimate for token length
            token_length = len(sentence.split()) * 1.3  # Rough estimate
            
            if current_length + token_length > max_tokens and current_chunk:
                # Start new chunk
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = token_length
            else:
                current_chunk.append(sentence)
                current_length += token_length
                
        except Exception as e:
            logger.warning(f"Error processing sentence: {e}")
            continue
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def _format_markdown(summary_text: str) -> str:
    """
    Convert summary into Markdown-friendly output.
    - Break into sentences
    - If multiple key points found, format as bullet points
    """
    if not summary_text or not summary_text.strip():
        return ""
    
    sentences = _split_sentences(summary_text)
    if not sentences:
        return ""
    
    if len(sentences) > 1:
        # Bullet list format
        return "\n".join([f"- {s}" for s in sentences])
    else:
        return sentences[0]


def _validate_input(text: str, max_sentences: int) -> None:
    """Validate input parameters."""
    if not text or not text.strip():
        raise ValueError("Input text is empty or contains only whitespace")
    
    if len(text.strip()) < MIN_TEXT_LENGTH:
        raise ValueError(f"Input text is too short for meaningful summarization (minimum {MIN_TEXT_LENGTH} characters)")
    
    if max_sentences < 1 or max_sentences > 50:
        raise ValueError("max_sentences must be between 1 and 50")


@performance_timer("enhance_summarize_text")
@memory_aware
def enhance_summarize_text(text: str, max_sentences: int = DEFAULT_MAX_SENTENCES) -> str:
    """
    Enhanced text summarization with advanced parameters and markdown formatting.
    
    Args:
        text: Input text to summarize
        max_sentences: Maximum number of sentences in summary
        
    Returns:
        Markdown-formatted summary
        
    Raises:
        ValueError: For invalid input parameters
        Exception: For processing errors
    """
    # Validate input
    _validate_input(text, max_sentences)
    
    # Initialize models if needed
    _initialize_models()
    
    try:
        # Step 1: Chunk text and summarize each chunk
        chunks = _chunk_text(text)
        if not chunks:
            raise ValueError("Text could not be processed into chunks")
        
        partial_summaries = []
        for i, chunk in enumerate(chunks):
            try:
                token_length = len(_tokenizer.encode(chunk, add_special_tokens=False))
                if token_length < 10:
                    logger.warning(f"Skipping chunk {i+1} - too short")
                    continue
                
                result = _summarizer(
                    chunk,
                    max_length=150,
                    min_length=40,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.8,
                )
                
                if result and len(result) > 0 and "summary_text" in result[0]:
                    summary_text = result[0]["summary_text"].strip()
                    if summary_text:
                        partial_summaries.append(summary_text)
                    else:
                        logger.warning(f"Empty summary for chunk {i+1}")
                else:
                    logger.warning(f"No summary generated for chunk {i+1}")
                    
            except Exception as e:
                logger.error(f"Error summarizing chunk {i+1}: {e}")
                continue
        
        if not partial_summaries:
            raise Exception("No summaries were generated from any chunks")
        
        # Step 2: Combine summaries and optionally re-summarize
        combined = " ".join(partial_summaries)
        
        # Check if combined summary needs further processing
        combined_token_length = len(_tokenizer.encode(combined, add_special_tokens=False))
        if combined_token_length > MAX_INPUT_TOKENS:
            logger.info("Performing second-pass summarization")
            try:
                result = _summarizer(
                    combined,
                    max_length=150,
                    min_length=50,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.8,
                )
                
                if result and len(result) > 0 and "summary_text" in result[0]:
                    combined = result[0]["summary_text"].strip()
                else:
                    logger.warning("Second-pass summarization failed, using original combined summary")
                    
            except Exception as e:
                logger.warning(f"Second-pass summarization failed: {e}")
        
        # Step 3: Trim to target length
        sentences = _split_sentences(combined)
        if not sentences:
            raise Exception("No sentences found in combined summary")
        
        formatted_summary = " ".join(sentences[:max_sentences])
        
        # Step 4: Format as Markdown with bullets if applicable
        return _format_markdown(formatted_summary)
        
    except Exception as e:
        logger.error(f"Error in enhance_summarize_text: {e}")
        raise Exception(f"Enhanced summarization failed: {e}")
