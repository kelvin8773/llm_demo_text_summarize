from transformers import pipeline, AutoTokenizer
import re

# Load model + tokenizer
CHINESE_MODEL = "uer/bart-base-chinese-cluecorpussmall"  # example
tokenizer = AutoTokenizer.from_pretrained(CHINESE_MODEL, use_fast=False)
summarizer = pipeline(
    "summarization",
    model=CHINESE_MODEL,
    tokenizer=tokenizer,
    device=-1,  # use CPU; change to 0 for GPU
)


def chunk_text(text, tokenizer, max_tokens=800):
    # Directly tokenize raw text
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(token_ids), max_tokens):
        chunk_ids = token_ids[i : i + max_tokens]
        chunks.append(tokenizer.decode(chunk_ids))
    return chunks


def chinese_summarize_text(text, max_sentences=10):
    if not text or not text.strip():
        return ""

    # Step 1: Split into safe token-sized chunks
    chunks = chunk_text(text, tokenizer, max_tokens=800)
    partial_summaries = []

    # Step 2: Summarize each chunk separately
    for ch in chunks:
        out = summarizer(ch, max_length=150, min_length=30, truncation=True)[0][
            "summary_text"
        ].strip()
        partial_summaries.append(out)

    # Step 3: Combine summaries
    combined_summary = " ".join(partial_summaries)

    # Step 4: Optional second-pass summary if too long
    token_len = len(tokenizer.encode(combined_summary, add_special_tokens=False))
    if token_len > 800:
        combined_summary = summarizer(
            combined_summary, max_length=150, min_length=30, truncation=True
        )[0]["summary_text"].strip()

    # Step 5: Format into Markdown bullet points (Chinese sentence split)
    sentences = re.split(r"[。！？]", combined_summary)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) > 1:
        md_output = "\n".join([f"- {s}" for s in sentences[:max_sentences]])
    else:
        md_output = sentences[0]

    return md_output
