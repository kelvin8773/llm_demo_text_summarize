from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, max_sentences=10):
    summary = summarizer(
        text, max_length=500, min_length=50, do_sample=False  # increased token limits
    )[0]["summary_text"]

    # Split into sentences and trim to target
    sentences = [s.strip() for s in summary.split(". ") if s.strip()]
    formatted = ".\n".join(sentences[:max_sentences]).strip()

    if not formatted.endswith("."):
        formatted += "."
    return formatted
