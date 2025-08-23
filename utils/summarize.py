from typing import List
import re
from transformers import pipeline, AutoTokenizer

MODEL_NAME = "google/pegasus-xsum"

_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
MAX_IN_TOKENS = min(_tokenizer.model_max_length, 1024)
_summarizer = pipeline(
    "summarization", model=MODEL_NAME, tokenizer=_tokenizer, device=-1
)


def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[\.\!\?])\s+", text.strip())
    return [s for s in sentences if s.strip()]


def _chunk_by_tokens(text: str, max_tokens: int) -> List[str]:
    sents = _split_sentences(text)
    chunks, current, cur_len = [], [], 0
    for s in sents:
        tok_len = len(_tokenizer.encode(s, add_special_tokens=False))
        if cur_len + tok_len > max_tokens and current:
            chunks.append(" ".join(current))
            current, cur_len = [s], tok_len
        else:
            current.append(s)
            cur_len += tok_len
    if current:
        chunks.append(" ".join(current))
    return [c.strip() for c in chunks if c.strip()]


def summarize_text(
    text: str,
    target_words: int = 60,
    mode: str = "deterministic_brief",
) -> str:
    if not text or not text.strip():
        return ""

    # Decoding settings
    if mode == "concise_natural":
        gen_kwargs = dict(
            max_length=min(4 * target_words, 120),
            min_length=max(int(1.2 * target_words), 20),
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            repetition_penalty=1.2,
            num_beams=1,
        )
    else:
        gen_kwargs = dict(
            max_length=min(4 * target_words, 120),
            min_length=max(int(1.2 * target_words), 20),
            do_sample=False,
            num_beams=4,
            length_penalty=1.0,
            repetition_penalty=1.1,
        )

    # Chunk input
    chunks = _chunk_by_tokens(text, max_tokens=MAX_IN_TOKENS - 32)
    partials = []

    for ch in chunks:
        token_len = len(_tokenizer.encode(ch, add_special_tokens=False))
        if token_len < 5:  # ignore tiny fragments
            continue
        # Adjust min_length down if chunk is short
        chunk_kwargs = dict(gen_kwargs)
        chunk_kwargs["min_length"] = min(
            chunk_kwargs["min_length"], max(token_len - 1, 5)
        )
        out = _summarizer(ch, **chunk_kwargs)[0]["summary_text"].strip()
        if out:
            partials.append(out)

    if not partials:
        return ""

    if len(partials) == 1:
        final = partials[0]
    else:
        combined = " ".join(partials).strip()
        if not combined:
            return ""
        token_len = len(_tokenizer.encode(combined, add_special_tokens=False))
        final_kwargs = dict(gen_kwargs)
        final_kwargs["min_length"] = min(
            final_kwargs["min_length"], max(token_len - 1, 5)
        )
        final = _summarizer(combined, **final_kwargs)[0]["summary_text"].strip()

    return re.sub(r"\s+", " ", final)
