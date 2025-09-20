# main.py - Enhanced LLM Text Summarization Tool
import streamlit as st
import traceback
from pathlib import Path
from utils.ingest import load_document
from utils.fast_summarize import fast_summarize_text
from utils.enhance_summarize import enhance_summarize_text
from utils.chinese_summarize import chinese_summarize_text
from utils.insights import (
    extract_keywords,
    extract_keywords_phrases,
    plot_keywords,
)
from utils.chinese_insights import extract_chinese_keywords, plot_chinese_keywords
from utils.parameters import (
    BART_CNN_MODEL,
    T5_LARGE_MODEL,
)

st.set_page_config(
    page_title="LLM Text Summarizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 Documents Summarizer & Insight Extractor")
st.markdown("---")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Language selection
    language = st.selectbox("Language", ["English", "Chinese"], help="Select the language of your text")
    
    # Mode selection (only for English)
    if language == "Chinese":
        mode = None
        st.info("Chinese mode uses specialized Chinese language models")
    else:
        mode = st.selectbox(
            "Summarize Mode", 
            ["Fast Summarizer", "Enhanced Summarizer"],
            help="Fast: Quick summarization. Enhanced: More detailed analysis"
        )
    
    # Model selection (only for Fast mode)
    if mode == "Fast Summarizer":
        model = st.selectbox(
            "Model",
            [BART_CNN_MODEL, T5_LARGE_MODEL],
            help="Choose the underlying language model"
        )
    else:
        model = None
    
    # Summary length
    max_sentences = st.slider(
        "Max Summary Sentences", 
        min_value=1, 
        max_value=20, 
        value=5, 
        step=1,
        help="Maximum number of sentences in the summary"
    )
    
    st.markdown("---")
    
    # Sample file option
    use_sample = st.checkbox("Use built-in sample file", value=True, help="Use a sample PDF for demonstration")

# Main content area
col1, col2, col3, col4 = st.columns(4)
# Status indicators
with col1:
    st.metric("Language", language)
with col2:
    st.metric("Mode", mode or "Chinese Mode")
with col3:
    st.metric("Model", model or "Auto")
with col4:
    st.metric("Max Sentences", max_sentences)

# Main processing section
if use_sample:
    st.subheader("📁 Sample Document Processing")
    
    try:
        sample_path = Path("data/AI_Transformation_Playbook.pdf")
        if not sample_path.exists():
            st.error("❌ Sample file not found. Please upload a file instead.")
            st.stop()
            
        with st.spinner("📖 Loading sample document..."):
            raw_text = load_document(sample_path.open("rb"))
            
        if not raw_text or len(raw_text.strip()) < 50:
            st.error("❌ Sample document appears to be empty or corrupted.")
            st.stop()
            
        st.success(f"✅ Loaded sample document ({len(raw_text)} characters)")
        
        # Processing with progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🤖 Initializing language model...")
            progress_bar.progress(20)
            
            status_text.text("📝 Generating summary...")
            progress_bar.progress(50)
            
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
            
            status_text.text("🔍 Extracting keywords...")
            progress_bar.progress(80)
            
            status_text.text("✅ Processing complete!")
            progress_bar.progress(100)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")
            st.error("Please try a different model or check your input.")
            if st.checkbox("Show technical details"):
                st.code(traceback.format_exc())
            st.stop()
            
    except Exception as e:
        st.error(f"❌ Error loading sample file: {str(e)}")
        st.info("Please upload a file instead or check if the sample file exists.")
        st.stop()
        
    st.info("ℹ️ Showing built-in sample. Upload a file to process live.")
else:
    st.subheader("📝 Custom Document Processing")
    
    input_mode = st.radio(
        "Input Method",
        [
            "Paste Text",
            "Upload File",
        ],
        horizontal=True,
    )
    
    raw_text = ""
    
    if input_mode == "Paste Text":
        st.markdown("**Paste your document text below:**")
        raw_text = st.text_area(
            "Document Text", 
            height=300,
            placeholder="Paste your text here...",
            help="Enter the text you want to summarize"
        )
        
        if not raw_text.strip():
            st.warning("⚠️ Please enter some text to summarize.")
            st.stop()
            
        if len(raw_text.strip()) < 50:
            st.warning("⚠️ Text seems too short for meaningful summarization.")
            
    else:
        st.markdown("**Upload a document file:**")
        uploaded_file = st.file_uploader(
            "Choose a file", 
            type=["pdf", "txt", "docx"],
            help="Supported formats: PDF, TXT, DOCX"
        )
        
        if not uploaded_file:
            st.info("📁 Please upload a file to get started.")
            st.stop()
            
        try:
            with st.spinner("📖 Loading document..."):
                raw_text = load_document(uploaded_file)
                
            if not raw_text or len(raw_text.strip()) < 50:
                st.error("❌ Document appears to be empty or could not be processed.")
                st.stop()
                
            st.success(f"✅ Successfully loaded {uploaded_file.name} ({len(raw_text)} characters)")
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.error("Please try a different file or check the file format.")
            if st.checkbox("Show technical details"):
                st.code(traceback.format_exc())
            st.stop()
    
    # Process the text
    if raw_text:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🤖 Initializing language model...")
            progress_bar.progress(20)
            
            status_text.text("📝 Generating summary...")
            progress_bar.progress(50)
            
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
            
            status_text.text("🔍 Extracting keywords...")
            progress_bar.progress(80)
            
            status_text.text("✅ Processing complete!")
            progress_bar.progress(100)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")
            st.error("Please try a different model or check your input.")
            if st.checkbox("Show technical details"):
                st.code(traceback.format_exc())
            st.stop()

# Results Display
if 'raw_text' in locals() and 'summary' in locals():
    st.markdown("---")
    
    # Create tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Summary", "🔍 Keywords", "📊 Visualization", "📝 Original Text"])
    
    with tab1:
        st.subheader("📄 Generated Summary")
        if summary:
            st.markdown(summary)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Summary Length", f"{len(summary)} chars")
            with col2:
                st.metric("Original Length", f"{len(raw_text)} chars")
            with col3:
                compression_ratio = len(summary) / len(raw_text) * 100
                st.metric("Compression Ratio", f"{compression_ratio:.1f}%")
        else:
            st.error("❌ No summary generated. Please check your input.")
    
    with tab2:
        st.subheader("🔍 Extracted Keywords")
        if keywords:
            if isinstance(keywords, list):
                # Display keywords as a nice list
                cols = st.columns(3)
                for i, keyword in enumerate(keywords):
                    with cols[i % 3]:
                        st.markdown(f"• **{keyword}**")
            else:
                st.write(keywords)
        else:
            st.warning("⚠️ No keywords extracted.")
    
    with tab3:
        st.subheader("📊 Keywords Visualization")
        try:
            if keywords:
                if language == "Chinese":
                    fig = plot_chinese_keywords(keywords)
                else:
                    fig = plot_keywords(keywords)
                st.pyplot(fig)
            else:
                st.info("No keywords available for visualization.")
        except Exception as e:
            st.error(f"❌ Error generating visualization: {str(e)}")
            st.info("Please try again or check your data.")
    
    with tab4:
        st.subheader("📝 Original Text Preview")
        preview_length = 1000
        if len(raw_text) > preview_length:
            st.markdown(f"**Showing first {preview_length} characters:**")
            st.text_area("Original Text", raw_text[:preview_length] + "...", height=300, disabled=True)
            st.info(f"Full text length: {len(raw_text)} characters")
        else:
            st.text_area("Original Text", raw_text, height=300, disabled=True)
    
    # Export functionality (placeholder for now)
    st.markdown("---")
    st.subheader("💾 Export Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export Summary as TXT"):
            st.download_button(
                label="Download Summary",
                data=summary,
                file_name="summary.txt",
                mime="text/plain"
            )
    
    with col2:
        if st.button("🔍 Export Keywords as TXT"):
            keywords_text = "\n".join(keywords) if isinstance(keywords, list) else str(keywords)
            st.download_button(
                label="Download Keywords",
                data=keywords_text,
                file_name="keywords.txt",
                mime="text/plain"
            )
    
    with col3:
        if st.button("📊 Export Full Report"):
            report = f"""SUMMARY REPORT
================

Summary:
{summary}

Keywords:
{keywords_text if 'keywords_text' in locals() else keywords}

Original Text Length: {len(raw_text)} characters
Summary Length: {len(summary)} characters
Compression Ratio: {len(summary) / len(raw_text) * 100:.1f}%
"""
            st.download_button(
                label="Download Report",
                data=report,
                file_name="summary_report.txt",
                mime="text/plain"
            )
else:
    st.info("👆 Please process a document to see results.")
