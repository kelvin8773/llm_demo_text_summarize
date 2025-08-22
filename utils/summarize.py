# utils/summarize.py
from transformers import pipeline

# Load once at module import
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)

def summarize_text(text, max_chunk=1000):
    # Simple chunking for long docs
    chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
    summaries = [summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
                 for chunk in chunks]
    return " ".join(summaries)
