# main_unified_dark.py - Unified Dark Mode LLM Text Summarization Tool
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
from utils.performance_dashboard import (
    render_performance_dashboard,
    render_performance_widget,
    render_performance_alerts,
    track_operation_time
)
from utils.performance import get_cache_stats, cleanup_resources

# Unified Dark Mode Configuration
st.set_page_config(
    page_title="LLM Text Summarizer Pro",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/kelvin8773/llm_demo_text_summarize',
        'Report a bug': 'https://github.com/kelvin8773/llm_demo_text_summarize/issues',
        'About': "AI-powered text summarization tool with multilingual support"
    }
)

# Unified Dark Mode CSS - Clean and Professional
st.markdown("""
<style>
    /* Reset and Base Styles */
    * {
        box-sizing: border-box;
    }
    
    /* Main App Dark Theme */
    .main {
        background: #0f0f0f;
        color: #ffffff;
    }
    
    .stApp {
        background: #0f0f0f;
        color: #ffffff;
    }
    
    /* Sidebar Dark Theme */
    .css-1d391kg {
        background: #1a1a1a;
        border-right: 1px solid #333333;
    }
    
    /* Sidebar Text - High Contrast */
    .css-1d391kg h1, 
    .css-1d391kg h2, 
    .css-1d391kg h3, 
    .css-1d391kg h4 {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    .css-1d391kg .stMarkdown {
        color: #ffffff !important;
    }
    
    .css-1d391kg .stMarkdown p {
        color: #ffffff !important;
    }
    
    /* Sidebar Form Controls */
    .css-1d391kg .stSelectbox label {
        color: #ffffff !important;
        font-weight: 500;
    }
    
    .css-1d391kg .stSelectbox > div > div {
        background-color: #2a2a2a;
        border: 1px solid #444444;
        color: #ffffff;
        border-radius: 6px;
    }
    
    .css-1d391kg .stSelectbox > div > div:hover {
        border-color: #666666;
    }
    
    .css-1d391kg .stSlider label {
        color: #ffffff !important;
        font-weight: 500;
    }
    
    .css-1d391kg .stSlider > div > div {
        background-color: #2a2a2a;
        border-radius: 6px;
        padding: 1rem;
    }
    
    .css-1d391kg .stCheckbox label {
        color: #ffffff !important;
        font-weight: 500;
    }
    
    .css-1d391kg .stCheckbox > div > div {
        background-color: #2a2a2a;
        border-radius: 6px;
        padding: 0.5rem;
    }
    
    .css-1d391kg .stRadio label {
        color: #ffffff !important;
        font-weight: 500;
    }
    
    .css-1d391kg .stRadio > div {
        background-color: #2a2a2a;
        border-radius: 6px;
        padding: 1rem;
    }
    
    /* Main Content Area */
    .main .block-container {
        background: #1a1a1a;
        border-radius: 12px;
        padding: 2rem;
        border: 1px solid #333333;
        margin: 1rem;
    }
    
    /* Dark Theme Text */
    .main .block-container h1,
    .main .block-container h2,
    .main .block-container h3,
    .main .block-container h4,
    .main .block-container p {
        color: #ffffff;
    }
    
    /* Enhanced Metric Cards - Dark Theme */
    .metric-card {
        background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
        color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #333333;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 1.2rem;
        font-weight: 600;
        color: #ffffff;
    }
    
    .metric-card p {
        margin: 0.5rem 0 0 0;
        font-size: 0.9rem;
        color: #cccccc;
    }
    
    /* Enhanced Buttons - Dark Theme */
    .stButton > button {
        background: linear-gradient(135deg, #4a4a4a 0%, #2a2a2a 100%);
        color: #ffffff;
        border: 1px solid #555555;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #5a5a5a 0%, #3a3a3a 100%);
        border-color: #666666;
        transform: translateY(-1px);
    }
    
    /* Enhanced Tabs - Dark Theme */
    .stTabs [data-baseweb="tab-list"] {
        background: #2a2a2a;
        border-radius: 8px;
        padding: 0.5rem;
        border: 1px solid #333333;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        color: #cccccc;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: #4a4a4a;
        color: #ffffff;
        border: 1px solid #555555;
    }
    
    /* Enhanced File Upload - Dark Theme */
    .stFileUploader > div {
        background: #2a2a2a;
        border-radius: 8px;
        border: 2px dashed #555555;
        padding: 2rem;
        text-align: center;
        color: #ffffff;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: #777777;
        background: #3a3a3a;
    }
    
    /* Enhanced Text Areas - Dark Theme */
    .stTextArea > div > div > textarea {
        background: #2a2a2a;
        border-radius: 8px;
        border: 1px solid #444444;
        padding: 1rem;
        font-size: 1rem;
        line-height: 1.5;
        color: #ffffff;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #666666;
        box-shadow: 0 0 0 2px rgba(102, 102, 102, 0.2);
    }
    
    .stTextArea > div > div > textarea::placeholder {
        color: #888888;
    }
    
    /* Enhanced Progress Bars - Dark Theme */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #4a4a4a 0%, #666666 100%);
        border-radius: 10px;
    }
    
    /* Enhanced Alerts - Dark Theme */
    .stAlert {
        border-radius: 8px;
        border: 1px solid #333333;
        background: #2a2a2a;
        color: #ffffff;
    }
    
    .stAlert [data-testid="stAlert"] {
        background: #2a2a2a;
        color: #ffffff;
    }
    
    /* Enhanced Radio Buttons - Dark Theme */
    .stRadio > div {
        background: #2a2a2a;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #333333;
    }
    
    .stRadio label {
        color: #ffffff !important;
    }
    
    /* Enhanced Selectbox - Dark Theme */
    .stSelectbox > div > div {
        background: #2a2a2a;
        border-radius: 8px;
        border: 1px solid #444444;
        color: #ffffff;
    }
    
    .stSelectbox label {
        color: #ffffff !important;
    }
    
    /* Enhanced Slider - Dark Theme */
    .stSlider > div > div {
        background: #2a2a2a;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #333333;
    }
    
    .stSlider label {
        color: #ffffff !important;
    }
    
    /* Enhanced Checkbox - Dark Theme */
    .stCheckbox > div > div {
        background: #2a2a2a;
        border-radius: 8px;
        padding: 0.5rem;
        border: 1px solid #333333;
    }
    
    .stCheckbox label {
        color: #ffffff !important;
    }
    
    /* Hero Section - Dark Theme */
    .hero-section {
        background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
        padding: 3rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        color: #ffffff;
        border: 1px solid #333333;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    
    .hero-section h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        color: #ffffff;
    }
    
    .hero-section p {
        font-size: 1.1rem;
        margin: 1rem 0 0 0;
        color: #cccccc;
    }
    
    /* Keyword Tags - Dark Theme */
    .keyword-tag {
        background: linear-gradient(135deg, #4a4a4a 0%, #2a2a2a 100%);
        color: #ffffff;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        text-align: center;
        margin: 0.25rem;
        font-weight: 500;
        border: 1px solid #555555;
        display: inline-block;
    }
    
    /* Info Cards - Dark Theme */
    .info-card {
        background: #2a2a2a;
        border: 1px solid #333333;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #ffffff;
    }
    
    .success-card {
        background: #1a3a1a;
        border: 1px solid #2a5a2a;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #ffffff;
    }
    
    .warning-card {
        background: #3a3a1a;
        border: 1px solid #5a5a2a;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #ffffff;
    }
    
    .error-card {
        background: #3a1a1a;
        border: 1px solid #5a2a2a;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #ffffff;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
            margin: 0.5rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
        
        .hero-section {
            padding: 2rem 1rem;
        }
        
        .hero-section h1 {
            font-size: 2rem;
        }
    }
    
    /* Loading Animation */
    .loading-spinner {
        width: 40px;
        height: 40px;
        border: 4px solid #333333;
        border-top: 4px solid #666666;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Smooth Transitions */
    * {
        transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section">
    <h1>📄 AI Text Summarizer Pro</h1>
    <p>Transform lengthy documents into concise summaries with intelligent keyword extraction</p>
</div>
""", unsafe_allow_html=True)

# Performance alerts
render_performance_alerts()

# Enhanced sidebar with dark theme
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Language selection
    language = st.selectbox(
        "🌍 Language", 
        ["English", "Chinese"], 
        help="Choose the language of your text for optimal processing"
    )
    
    st.markdown("---")
    
    # Mode selection
    if language == "Chinese":
        mode = None
        st.markdown("### 🇨🇳 Chinese Mode")
        st.info("Chinese mode uses specialized Chinese language models")
    else:
        st.markdown("### ⚡ Summarization Mode")
        mode = st.selectbox(
            "Choose Mode", 
            ["Fast Summarizer", "Enhanced Summarizer"],
            help="Fast: Quick processing. Enhanced: Detailed analysis"
        )
    
    st.markdown("---")
    
    # Model selection (only for Fast mode)
    if mode == "Fast Summarizer":
        st.markdown("### 🤖 Model Selection")
        model = st.selectbox(
            "Choose Model",
            [BART_CNN_MODEL, T5_LARGE_MODEL],
            help="Select the underlying language model"
        )
    else:
        model = None
    
    st.markdown("---")
    
    # Summary length
    st.markdown("### 📏 Summary Length")
    max_sentences = st.slider(
        "Maximum Sentences", 
        min_value=1, 
        max_value=20, 
        value=5, 
        step=1,
        help="Control the length of your summary"
    )
    
    st.markdown("---")
    
    # Sample file option
    st.markdown("### 📁 Input Method")
    use_sample = st.checkbox(
        "Use Sample Document", 
        value=True, 
        help="Process the built-in sample PDF for demonstration"
    )
    
    st.markdown("---")
    
    # Performance widget
    render_performance_widget()

# Current configuration display
st.markdown("### 📊 Current Configuration")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>🌍 Language</h3>
        <p>{}</p>
    </div>
    """.format(language), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3>⚡ Mode</h3>
        <p>{}</p>
    </div>
    """.format(mode or "Chinese Mode"), unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3>🤖 Model</h3>
        <p>{}</p>
    </div>
    """.format(model or "Auto"), unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h3>📏 Max Sentences</h3>
        <p>{}</p>
    </div>
    """.format(max_sentences), unsafe_allow_html=True)

# Main processing section
if use_sample:
    st.markdown("### 📁 Sample Document Processing")
    
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
                summary = track_operation_time("chinese_summarization")(chinese_summarize_text)(raw_text, max_sentences)
                keywords = track_operation_time("chinese_keywords")(extract_chinese_keywords)(raw_text, top_n=15)
            else:
                if mode == "Fast Summarizer":
                    summary = track_operation_time("fast_summarization")(fast_summarize_text)(raw_text, max_sentences, model_name=model)
                    keywords = track_operation_time("english_keywords")(extract_keywords)(raw_text, top_n=15)
                else:
                    summary = track_operation_time("enhanced_summarization")(enhance_summarize_text)(raw_text, max_sentences)
                    keywords = track_operation_time("english_phrases")(extract_keywords_phrases)(raw_text, top_n=15)
            
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
    st.markdown("### 📝 Custom Document Processing")
    
    # Input method selection
    input_mode = st.radio(
        "Choose Input Method",
        ["Paste Text", "Upload File"],
        horizontal=True
    )
    
    raw_text = ""
    
    if input_mode == "Paste Text":
        st.markdown("### 📝 Text Input")
        raw_text = st.text_area(
            "Enter your document text", 
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
        st.markdown("### 📁 File Upload")
        uploaded_file = st.file_uploader(
            "Choose a file", 
            type=["pdf", "txt", "docx"],
            help="Supported formats: PDF, TXT, DOCX (max 10MB)"
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
                summary = track_operation_time("chinese_summarization")(chinese_summarize_text)(raw_text, max_sentences)
                keywords = track_operation_time("chinese_keywords")(extract_chinese_keywords)(raw_text, top_n=15)
            else:
                if mode == "Fast Summarizer":
                    summary = track_operation_time("fast_summarization")(fast_summarize_text)(raw_text, max_sentences, model_name=model)
                    keywords = track_operation_time("english_keywords")(extract_keywords)(raw_text, top_n=15)
                else:
                    summary = track_operation_time("enhanced_summarization")(enhance_summarize_text)(raw_text, max_sentences)
                    keywords = track_operation_time("english_phrases")(extract_keywords_phrases)(raw_text, top_n=15)
            
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

# Results display
if 'raw_text' in locals() and 'summary' in locals():
    st.markdown("---")
    
    # Enhanced tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Summary", 
        "🔍 Keywords", 
        "📊 Visualization", 
        "📝 Original Text", 
        "⚡ Performance"
    ])
    
    with tab1:
        st.markdown("### 📄 Generated Summary")
        
        if summary:
            # Summary display with dark theme
            st.markdown("""
            <div class="info-card">
            """, unsafe_allow_html=True)
            st.markdown(summary)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h3>📏 Summary Length</h3>
                    <p>{} characters</p>
                </div>
                """.format(len(summary)), unsafe_allow_html=True)
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h3>📄 Original Length</h3>
                    <p>{} characters</p>
                </div>
                """.format(len(raw_text)), unsafe_allow_html=True)
            with col3:
                compression_ratio = len(summary) / len(raw_text) * 100
                st.markdown("""
                <div class="metric-card">
                    <h3>📊 Compression Ratio</h3>
                    <p>{:.1f}%</p>
                </div>
                """.format(compression_ratio), unsafe_allow_html=True)
        else:
            st.error("❌ No summary generated. Please check your input.")
    
    with tab2:
        st.markdown("### 🔍 Extracted Keywords")
        
        if keywords:
            if isinstance(keywords, list):
                st.markdown("### 🏷️ Keywords")
                cols = st.columns(3)
                for i, keyword in enumerate(keywords):
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div class="keyword-tag">
                            {keyword}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.write(keywords)
        else:
            st.warning("⚠️ No keywords extracted.")
    
    with tab3:
        st.markdown("### 📊 Keywords Visualization")
        
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
        st.markdown("### 📝 Original Text Preview")
        
        preview_length = 1000
        if len(raw_text) > preview_length:
            st.markdown(f"**Showing first {preview_length} characters:**")
            st.text_area("Original Text", raw_text[:preview_length] + "...", height=300, disabled=True)
            st.info(f"Full text length: {len(raw_text)} characters")
        else:
            st.text_area("Original Text", raw_text, height=300, disabled=True)
    
    with tab5:
        render_performance_dashboard()
    
    # Export functionality
    st.markdown("---")
    st.markdown("### 💾 Export Your Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export Summary"):
            st.download_button(
                label="Download Summary",
                data=summary,
                file_name="summary.txt",
                mime="text/plain"
            )
    
    with col2:
        if st.button("🔍 Export Keywords"):
            keywords_text = "\n".join(keywords) if isinstance(keywords, list) else str(keywords)
            st.download_button(
                label="Download Keywords",
                data=keywords_text,
                file_name="keywords.txt",
                mime="text/plain"
            )
    
    with col3:
        if st.button("📊 Export Full Report"):
            keywords_text = "\n".join(keywords) if isinstance(keywords, list) else str(keywords)
            report = f"""SUMMARY REPORT
================

Summary:
{summary}

Keywords:
{keywords_text}

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
    st.markdown("""
    <div class="info-card">
        <h3>👆 Please process a document to see results</h3>
        <p>Upload a file or paste text to get started with summarization</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="
    text-align: center;
    color: #888888;
    padding: 2rem 0;
    border-top: 1px solid #333333;
    margin-top: 3rem;
">
    <p style="margin: 0; font-size: 1rem;">
        🤖 <strong>AI Text Summarizer Pro</strong> - Powered by Advanced Language Models
    </p>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">
        Built with ❤️ using Streamlit, Transformers, and modern AI technology
    </p>
</div>
""", unsafe_allow_html=True)