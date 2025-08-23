# main.py (only the relevant additions shown)
import streamlit as st
from pathlib import Path
from utils.ingest import load_document
from utils.fast_summarize import summarize_text as fast_summarize_text
from utils.enhance_summarize import enhance_summarize_text
from utils.insights import (
    extract_keywords,
    extract_keywords_phrases,
    plot_keywords,
)

st.title("📄 Documents Summarizer & Insight Extractor")

# Controls for one-pass summarizer

col1, col2 = st.columns(2)
with col1:
    max_sentences = st.slider(
        "Max Summary Sentences", min_value=5, max_value=30, value=10, step=1
    )
with col2:
    mode = st.selectbox("Summarize Mode", ["Fast Summarizer", "Enhanced Summarizer"])

use_sample = st.sidebar.checkbox("Use built-in sample file", value=True)

if use_sample:
    sample_path = Path("data/AI_Transformation_Playbook.pdf")
    raw_text = load_document(sample_path.open("rb"))

    with st.spinner("Summarizing..."):
        if mode == "Fast Summarizer":
            summary = fast_summarize_text(raw_text, max_sentences)
            keywords = extract_keywords(raw_text, top_n=15)
        else:
            summary = enhance_summarize_text(raw_text, max_sentences)
            keywords = extract_keywords_phrases(raw_text, top_n=15)

    # fall back to your precomputed cache if you prefer instant display:
    # summary = Path("data/sample_summary.txt").read_text()
    # keywords = Path("data/sample_keywords.txt").read_text().splitlines()
    st.info("Showing built-in sample. Upload a file to process live.")
else:
    uploaded_file = st.file_uploader(
        "Upload a PDF/Text file", type=["pdf", "txt", "docx"]
    )
    if not uploaded_file:
        st.stop()
    raw_text = load_document(uploaded_file)

    with st.spinner("Summarizing..."):
        if mode == "Fast Summarizer":
            summary = fast_summarize_text(raw_text, max_sentences)
            keywords = extract_keywords(raw_text, top_n=15)
        else:
            summary = enhance_summarize_text(raw_text, max_sentences)
            keywords = extract_keywords_phrases(raw_text, top_n=15)

# Display
st.subheader("Original Text (preview)")
st.write(raw_text[:800] + "...")

st.subheader("Summary")
st.write(summary)

st.subheader("Top Keywords")
st.write(keywords)

fig = plot_keywords(keywords)
st.pyplot(fig)
