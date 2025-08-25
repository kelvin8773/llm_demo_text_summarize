from transformers import pipeline, AutoTokenizer
import re
from .parameters import BART_CNN_MODEL

MODEL_NAME = BART_CNN_MODEL

_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
MAX_INPUT_TOKENS = min(_tokenizer.model_max_length, 1024)

_summarizer = pipeline(
    "summarization", model=MODEL_NAME, tokenizer=_tokenizer, device=-1
)


def _split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _chunk_text(text, max_tokens=MAX_INPUT_TOKENS - 32):
    sentences = _split_sentences(text)
    chunks, current, cur_len = [], [], 0
    for s in sentences:
        tok_len = len(_tokenizer.encode(s, add_special_tokens=False))
        if cur_len + tok_len > max_tokens and current:
            chunks.append(" ".join(current))
            current, cur_len = [s], tok_len
        else:
            current.append(s)
            cur_len += tok_len
    if current:
        chunks.append(" ".join(current))
    return chunks


def _format_markdown(summary_text):
    """
    Convert summary into Markdown-friendly output.
    - Break into sentences
    - If multiple key points found, format as bullet points
    """
    sentences = _split_sentences(summary_text)
    if len(sentences) > 1:
        # Bullet list format
        md_output = "\n".join([f"- {s}" for s in sentences])
    else:
        md_output = sentences[0]
    return md_output


def enhance_summarize_text(text, max_sentences=10):
    if not text or not text.strip():
        return ""

    # Step 1: chunk + summarize each
    chunks = _chunk_text(text)
    partial_summaries = []
    for ch in chunks:
        if len(_tokenizer.encode(ch, add_special_tokens=False)) < 10:
            continue
        part = _summarizer(
            ch,
            max_length=150,
            min_length=40,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
        )[0]["summary_text"].strip()
        partial_summaries.append(part)

    # Step 2: combine summaries and optionally re-summarize
    combined = " ".join(partial_summaries)
    if len(_tokenizer.encode(combined, add_special_tokens=False)) > MAX_INPUT_TOKENS:
        combined = _summarizer(
            combined,
            max_length=150,
            min_length=50,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
        )[0]["summary_text"].strip()

    # Step 3: trim to target length
    sentences = _split_sentences(combined)
    formatted_summary = " ".join(sentences[:max_sentences])

    # Step 4: format as Markdown with bullets if applicable
    return _format_markdown(formatted_summary)
