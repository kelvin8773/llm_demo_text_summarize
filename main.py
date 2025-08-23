# main.py
import streamlit as st
from utils.ingest import load_document
from utils.summarize import summarize_text
from utils.new_summarize import summarize_text as new_summarize_text
from utils.insights import extract_keywords, cluster_keywords, plot_keywords

st.title("📄 Local Text Summarizer & Insight Extractor")

uploaded_file = st.file_uploader("Upload a PDF/Text file", type=["pdf", "txt", "docx"])

if uploaded_file:
    # 1. Ingest
    raw_text = load_document(uploaded_file)
    st.subheader("Original Text (preview)")
    st.write(raw_text[:500] + "...")

    # 2. Summarize
    # old_summary = summarize_text(raw_text, max_chunk=500)
    # st.subheader("Summary (Old Method)")
    # st.write(old_summary)

    # New summarization method
    new_summary = new_summarize_text(raw_text, max_sentences=8)
    st.subheader("Summary (New Method)")
    st.write(new_summary)

    # 3. Insights
    keywords = extract_keywords(raw_text, top_n=10)
    st.subheader("Top Keywords")
    st.write(keywords)

    clusters = cluster_keywords(keywords, n_clusters=3)
    st.subheader("Keyword Clusters")
    st.write(clusters)

    # 4. Visualization
    fig = plot_keywords(keywords)
    st.pyplot(fig)
