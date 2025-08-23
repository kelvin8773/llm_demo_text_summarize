from transformers import pipeline
from .parameters import BART_CNN_MODEL


def summarize_text(text, max_sentences=10, model_name=BART_CNN_MODEL):
    summarizer = pipeline("summarization", model=model_name)
    summary = summarizer(text, max_length=300, min_length=60, do_sample=True)[0][
        "summary_text"
    ]

    # Split into sentences and trim to target
    sentences = [s.strip() for s in summary.split(". ") if s.strip()]
    formatted = ".\n".join(sentences[:max_sentences]).strip()

    if not formatted.endswith("."):
        formatted += "."
    return formatted
