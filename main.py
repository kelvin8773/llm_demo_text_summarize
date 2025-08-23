# main.py (only the relevant additions shown)
import streamlit as st
from pathlib import Path
from utils.ingest import load_document
from utils.fast_summarize import summarize_text as fast_summarize_text
from utils.enhance_summarize import enhance_summarize_text
from utils.chinese_summarize import chinese_summarize_text
from utils.insights import (
    extract_keywords,
    extract_keywords_phrases,
    plot_keywords,
    plot_chinese_keywords
)
from utils.chinese_insights import extract_chinese_keywords
from utils.parameters import (
    BART_CNN_MODEL,
    MRM_MODEL,
    T5_LARGE_MODEL,
)

st.title("📄 Documents Summarizer & Insight Extractor")

# Controls for one-pass summarizer

col1, col2, col3, col4 = st.columns(4)
with col1:
    max_sentences = st.slider(
        "Max Summary Sentences", min_value=5, max_value=30, value=10, step=1
    )
with col2:
    language = st.selectbox("Language", ["English", "Chinese"])

with col3:
    if language == "Chinese":
        mode = None
    else:
        mode = st.selectbox("Summarize Mode", ["Fast Summarizer", "Enhanced Summarizer"])

with col4:
    if mode == "Fast Summarizer":
        model = st.selectbox(
            "Model",
            [BART_CNN_MODEL, T5_LARGE_MODEL, MRM_MODEL],
        )
    else:
        model = None


use_sample = st.sidebar.checkbox("Use built-in sample file", value=True)

if use_sample:
    sample_path = Path("data/AI_Transformation_Playbook.pdf")
    raw_text = load_document(sample_path.open("rb"))

    with st.spinner("Summarizing..."):
        if language == "Chinese":
            summary = chinese_summarize_text(raw_text, max_sentences)
            keywords = extract_chinese_keywords(raw_text, top_n=15)
        else:    
            if mode == "Fast Summarizer":
                summary = fast_summarize_text(raw_text, max_sentences, model_name=model)
                keywords = extract_keywords(raw_text, top_n=15)
            else:
                summary = enhance_summarize_text(raw_text, max_sentences)
                keywords = extract_keywords_phrases(raw_text, top_n=15)
    
    st.info("Showing built-in sample. Upload a file to process live.")
else:
    input_mode = st.radio("Input Method", ["Upload File", "Paste Text"], horizontal=True)

    if input_mode == "Paste Text":
        raw_text = st.text_area("Paste your document text here:", height=300)
        if not raw_text.strip():
            st.stop()
    else:
        uploaded_file = st.file_uploader(
            "Upload a PDF/Text file", type=["pdf", "txt", "docx"]
        )
        if not uploaded_file:
            st.stop()
        raw_text = load_document(uploaded_file)

    with st.spinner("Summarizing..."):
        if language == "Chinese":
            summary = chinese_summarize_text(raw_text, max_sentences)
            keywords = extract_chinese_keywords(raw_text, top_n=15)
        else:
            if mode == "Fast Summarizer":
                summary = fast_summarize_text(raw_text, max_sentences, model_name=model)
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

if language == "Chinese":
    fig = plot_chinese_keywords(keywords)
else:
    fig = plot_keywords(keywords)

st.pyplot(fig)
