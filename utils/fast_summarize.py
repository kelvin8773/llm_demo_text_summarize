from transformers import pipeline, AutoTokenizer
from .parameters import BART_CNN_MODEL

def fast_summarize_text(text, max_sentences=3, model_name=BART_CNN_MODEL):
    """
    Summarize text safely by chunking to avoid position embedding overflows.
    Handles Chinese/English length quirks and performs optional second-pass summarization.
    """
    # Init tokenizer + pipeline once per session
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    summarizer = pipeline("summarization", model=model_name)

    # 1. Token-safe chunking
    def chunk_text(txt, max_tokens=900):
        token_ids = tokenizer.encode(txt, add_special_tokens=False)
        chunks = []
        for i in range(0, len(token_ids), max_tokens):
            chunk_ids = token_ids[i:i+max_tokens]
            chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
        return chunks

    # 2. Summarize each chunk individually
    chunks = chunk_text(text)
    partials = []
    for ch in chunks:
        result = summarizer(
            ch,
            max_length=150,      # keep well within limit
            min_length=30,
            truncation=True
        )[0]["summary_text"].strip()
        partials.append(result)

    # 3. Combine summaries
    combined = " ".join(partials)

    # 4. Second-pass summarization if still too long
    if len(tokenizer.encode(combined)) > 900:
        combined_chunks = chunk_text(combined)
        refined_parts = []
        for ch in combined_chunks:
            out = summarizer(
                ch,
                max_length=150,
                min_length=30,
                truncation=True
            )[0]["summary_text"].strip()
            refined_parts.append(out)
        combined = " ".join(refined_parts)

    # 5. Optional: shorten to desired sentence count
    if max_sentences:
        sentences = combined.split(".")
        combined = ".".join(sentences[:max_sentences]).strip() + "."

    return combined

