from transformers import pipeline, AutoTokenizer
import logging
from .parameters import BART_CNN_MODEL

logger = logging.getLogger(__name__)


def fast_summarize_text(text, max_sentences=3, model_name=BART_CNN_MODEL):
    """
    Fast text summarization using transformer models with enhanced error handling.
    
    Args:
        text (str): Input text to summarize
        max_sentences (int): Maximum number of sentences in summary
        model_name (str): Name of the model to use
        
    Returns:
        str: Generated summary
        
    Raises:
        ValueError: For invalid input parameters
        Exception: For model loading or processing errors
    """
    # Input validation
    if not text or not text.strip():
        raise ValueError("Input text is empty or contains only whitespace")
    
    if len(text.strip()) < 50:
        raise ValueError("Input text is too short for meaningful summarization (minimum 50 characters)")
    
    if max_sentences < 1 or max_sentences > 50:
        raise ValueError("max_sentences must be between 1 and 50")
    
    if not model_name:
        raise ValueError("Model name is required")
    
    try:
        # Init tokenizer + pipeline with error handling
        logger.info(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        summarizer = pipeline("summarization", model=model_name)
        
        # Validate model loaded successfully
        if not tokenizer or not summarizer:
            raise Exception("Failed to load model components")
        
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        raise Exception(f"Failed to load model '{model_name}': {str(e)}")

    try:
        # 1. Token-safe chunking with validation
        def chunk_text(txt, max_tokens=900):
            if not txt or not txt.strip():
                return []
            
            try:
                token_ids = tokenizer.encode(txt, add_special_tokens=False)
                if not token_ids:
                    return []
                
                chunks = []
                for i in range(0, len(token_ids), max_tokens):
                    chunk_ids = token_ids[i : i + max_tokens]
                    chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
                    if chunk_text and chunk_text.strip():
                        chunks.append(chunk_text.strip())
                
                return chunks
            except Exception as e:
                logger.error(f"Error chunking text: {str(e)}")
                raise Exception(f"Failed to chunk text: {str(e)}")

        # 2. Summarize each chunk individually with error handling
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("Text could not be processed into chunks")
        
        partials = []
        for i, ch in enumerate(chunks):
            try:
                if len(ch.strip()) < 10:
                    logger.warning(f"Skipping chunk {i+1} - too short")
                    continue
                
                result = summarizer(
                    ch, 
                    max_length=150, 
                    min_length=30, 
                    truncation=True
                )
                
                if result and len(result) > 0 and "summary_text" in result[0]:
                    summary_text = result[0]["summary_text"].strip()
                    if summary_text:
                        partials.append(summary_text)
                    else:
                        logger.warning(f"Empty summary for chunk {i+1}")
                else:
                    logger.warning(f"No summary generated for chunk {i+1}")
                    
            except Exception as e:
                logger.error(f"Error summarizing chunk {i+1}: {str(e)}")
                # Continue with other chunks rather than failing completely
                continue

        if not partials:
            raise Exception("No summaries were generated from any chunks")

        # 3. Combine summaries
        combined = " ".join(partials)
        if not combined or not combined.strip():
            raise Exception("Failed to combine summaries")

        # 4. Second-pass summarization if still too long
        try:
            if len(tokenizer.encode(combined)) > 900:
                logger.info("Performing second-pass summarization")
                combined_chunks = chunk_text(combined)
                refined_parts = []
                
                for ch in combined_chunks:
                    try:
                        out = summarizer(ch, max_length=150, min_length=30, truncation=True)[0]["summary_text"].strip()
                        if out:
                            refined_parts.append(out)
                    except Exception as e:
                        logger.warning(f"Error in second-pass summarization: {str(e)}")
                        continue
                
                if refined_parts:
                    combined = " ".join(refined_parts)
                else:
                    logger.warning("Second-pass summarization failed, using original combined summary")
        except Exception as e:
            logger.warning(f"Second-pass summarization failed: {str(e)}")

        # 5. Optional: shorten to desired sentence count
        if max_sentences and max_sentences > 0:
            try:
                sentences = combined.split(".")
                sentences = [s.strip() for s in sentences if s.strip()]
                
                if len(sentences) > max_sentences:
                    combined = ". ".join(sentences[:max_sentences]).strip()
                    if not combined.endswith("."):
                        combined += "."
                elif len(sentences) > 0:
                    combined = ". ".join(sentences).strip()
                    if not combined.endswith("."):
                        combined += "."
            except Exception as e:
                logger.warning(f"Error formatting sentences: {str(e)}")

        # Final validation
        if not combined or len(combined.strip()) < 10:
            raise Exception("Generated summary is too short or empty")

        logger.info(f"Successfully generated summary ({len(combined)} characters)")
        return combined.strip()

    except Exception as e:
        logger.error(f"Error in fast_summarize_text: {str(e)}")
        raise Exception(f"Summarization failed: {str(e)}")
