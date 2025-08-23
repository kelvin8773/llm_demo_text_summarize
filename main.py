# main.py (only the relevant additions shown)
import streamlit as st
from pathlib import Path
from utils.ingest import load_document
from utils.summarize import summarize_text
from utils.insights import extract_keywords, cluster_keywords, plot_keywords

st.title("📄 Local Text Summarizer & Insight Extractor")

# Controls for one-pass summarizer
col1, col2 = st.columns(2)
with col1:
    target_words = st.slider(
        "Target summary length (words)", min_value=30, max_value=150, value=60, step=10
    )
with col2:
    mode = st.selectbox("Tone mode", ["deterministic_brief", "concise_natural"])

use_sample = st.sidebar.checkbox("Use built-in sample file", value=True)

if use_sample:
    sample_path = Path("data/AI_Transformation_Playbook.pdf")
    raw_text = load_document(sample_path.open("rb"))
    # Optionally re-run one-pass summarizer live on sample to reflect your controls:
    summary = summarize_text(raw_text, target_words=target_words, mode=mode)
    # Or fall back to your precomputed cache if you prefer instant display:
    # summary = Path("data/sample_summary.txt").read_text()
    keywords = Path("data/sample_keywords.txt").read_text().splitlines()
    st.info("Showing built-in sample. Upload a file to process live.")
else:
    uploaded_file = st.file_uploader(
        "Upload a PDF/Text file", type=["pdf", "txt", "docx"]
    )
    if not uploaded_file:
        st.stop()
    raw_text = load_document(uploaded_file)
    with st.spinner("Summarizing..."):
        summary = summarize_text(raw_text, target_words=target_words, mode=mode)
    keywords = extract_keywords(raw_text, top_n=15)

# Display
st.subheader("Original Text (preview)")
st.write(raw_text[:500] + "...")

st.subheader("Summary")
st.write(summary)

st.subheader("Top Keywords")
st.write(keywords)

clusters = cluster_keywords(keywords, n_clusters=3)
st.subheader("Keyword Clusters")
st.write(clusters)

fig = plot_keywords(keywords)
st.pyplot(fig)


preset = st.radio("Quick presets", ["Balanced", "Ultra-concise"])
if preset == "Ultra-concise":
    target_words = 40
    mode = "deterministic_brief"  # or "concise_natural" if you're okay with mild randomness
