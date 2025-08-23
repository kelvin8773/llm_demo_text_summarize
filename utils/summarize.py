# utils/summarize.py
from typing import List
import re
from transformers import pipeline, AutoTokenizer

# Choose one model:
# - "google/pegasus-xsum"  -> ultra-concise, headline-like, very natural phrasing
# - "facebook/bart-large-cnn" -> balanced, slightly longer, robust
MODEL_NAME = "google/pegasus-xsum"

_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
MAX_IN_TOKENS = min(_tokenizer.model_max_length, 1024)  # many summarizers cap at 1024
_summarizer = pipeline(
    "summarization", model=MODEL_NAME, tokenizer=_tokenizer, device=-1
)


def _split_sentences(text: str) -> List[str]:
    # Lightweight sentence splitter to avoid NLTK downloads
    sentences = re.split(r"(?<=[\.\!\?])\s+", text.strip())
    return [s for s in sentences if s]


def _chunk_by_tokens(text: str, max_tokens: int) -> List[str]:
    # Pack sentences into chunks that respect the model's max input length
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
    return chunks


def summarize_text(
    text: str,
    target_words: int = 60,
    mode: str = "deterministic_brief",  # "deterministic_brief" or "concise_natural"
) -> str:
    if not text or not text.strip():
        return ""

    # Prepare decoding presets
    if mode == "concise_natural":
        gen_kwargs = dict(
            max_length=min(4 * target_words, 120),  # rough token heuristic
            min_length=max(int(1.2 * target_words), 20),
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            repetition_penalty=1.2,
            num_beams=1,
        )
    else:  # deterministic_brief
        gen_kwargs = dict(
            max_length=min(4 * target_words, 120),
            min_length=max(int(1.2 * target_words), 20),
            do_sample=False,
            num_beams=4,
            length_penalty=1.0,
            repetition_penalty=1.1,
        )

    # Chunk the input and summarize each chunk in one pass
    chunks = _chunk_by_tokens(
        text, max_tokens=MAX_IN_TOKENS - 32
    )  # margin for specials
    partials = []
    for ch in chunks:
        out = _summarizer(ch, **gen_kwargs)[0]["summary_text"].strip()
        partials.append(out)

    # If there are multiple partials, do a short "collate" pass with the SAME model
    # Still a single pipeline; this just fuses chunk summaries into a final brief summary.
    combined = " ".join(partials)
    final = _summarizer(
        combined,
        max_length=min(4 * target_words, 120),
        min_length=max(int(1.0 * target_words), 20),
        do_sample=gen_kwargs.get("do_sample", False),
        top_p=gen_kwargs.get("top_p", None),
        temperature=gen_kwargs.get("temperature", None),
        num_beams=gen_kwargs.get("num_beams", 4),
        length_penalty=1.0,
        repetition_penalty=1.1,
    )[0]["summary_text"].strip()

    # Light cleanup so it reads cleanly
    final = re.sub(r"\s+", " ", final)
    return final
