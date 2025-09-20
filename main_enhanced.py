# main_enhanced.py - Enhanced UI for LLM Text Summarization Tool
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

# Enhanced page configuration
st.set_page_config(
    page_title="AI Text Summarizer Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/kelvin8773/llm_demo_text_summarize',
        'Report a bug': "https://github.com/kelvin8773/llm_demo_text_summarize/issues",
        'About': "# AI Text Summarizer Pro\nPowered by advanced language models"
    }
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --warning-color: #ff7f0e;
        --error-color: #d62728;
        --background-color: #f8f9fa;
        --card-background: #ffffff;
        --text-color: #2c3e50;
        --border-color: #e9ecef;
    }
    
    /* Global styles */
    .main {
        font-family: 'Inter', sans-serif;
        background-color: var(--background-color);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 0;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Card styling */
    .card {
        background: var(--card-background);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border: 1px solid var(--border-color);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.12);
    }
    
    /* Status indicators */
    .status-indicator {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 500;
        text-align: center;
        margin: 0.25rem;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: var(--card-background);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Metric cards */
    .metric-card {
        background: var(--card-background);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid var(--primary-color);
    }
    
    /* Alert styling */
    .alert-success {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        color: #155724;
    }
    
    .alert-error {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        padding: 1rem;
        color: #721c24;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        color: #856404;
    }
    
    .alert-info {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 1px solid #bee5eb;
        border-radius: 8px;
        padding: 1rem;
        color: #0c5460;
    }
    
    /* File upload area */
    .stFileUploader > div {
        border: 2px dashed var(--border-color);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: var(--background-color);
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: var(--primary-color);
        background: rgba(31, 119, 180, 0.05);
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 1px solid var(--border-color);
        transition: border-color 0.3s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 2px rgba(31, 119, 180, 0.1);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 1px solid var(--border-color);
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .card {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Enhanced header
st.markdown("""
<div class="main-header fade-in-up">
    <h1>🤖 AI Text Summarizer Pro</h1>
    <p>Transform lengthy documents into concise summaries with intelligent keyword extraction</p>
</div>
""", unsafe_allow_html=True)

# Performance alerts
render_performance_alerts()

# Enhanced sidebar with better organization
with st.sidebar:
    st.markdown("""
    <div class="card">
        <h3 style="margin-top: 0; color: var(--primary-color);">⚙️ Configuration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Language selection with enhanced styling
    st.markdown("### 🌐 Language Selection")
    language = st.selectbox(
        "Select Language", 
        ["English", "Chinese"], 
        help="Choose the language of your input text",
        key="language_select"
    )
    
    # Mode selection with conditional display
    if language == "Chinese":
        mode = None
        st.markdown("""
        <div class="alert-info">
            <strong>🇨🇳 Chinese Mode</strong><br>
            Uses specialized Chinese language models for optimal results
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("### 🚀 Summarization Mode")
        mode = st.selectbox(
            "Choose Mode", 
            ["Fast Summarizer", "Enhanced Summarizer"],
            help="Fast: Quick processing. Enhanced: Detailed analysis with advanced features",
            key="mode_select"
        )
    
    # Model selection with enhanced descriptions
    if mode == "Fast Summarizer":
        st.markdown("### 🧠 AI Model Selection")
        model_options = {
            BART_CNN_MODEL: "📰 BART-CNN (Best for news & formal docs)",
            T5_LARGE_MODEL: "🔬 T5-Large (Versatile for diverse content)"
        }
        model_display = st.selectbox(
            "Select Model",
            list(model_options.keys()),
            format_func=lambda x: model_options[x],
            help="Choose the underlying language model",
            key="model_select"
        )
        model = model_display
    else:
        model = None
    
    # Enhanced slider with better styling
    st.markdown("### 📏 Summary Length")
    max_sentences = st.slider(
        "Maximum Sentences", 
        min_value=1, 
        max_value=20, 
        value=5, 
        step=1,
        help="Control the length of your summary",
        key="max_sentences_slider"
    )
    
    st.markdown("---")
    
    # Sample file option with enhanced styling
    st.markdown("### 📁 Input Options")
    use_sample = st.checkbox(
        "🎯 Use Sample Document", 
        value=True, 
        help="Process the built-in AI Transformation Playbook PDF",
        key="sample_checkbox"
    )
    
    # Performance widget
    render_performance_widget()

# Enhanced status indicators
st.markdown("### 📊 Current Configuration")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="status-indicator">
        <strong>🌐 Language</strong><br>
        {language}
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="status-indicator">
        <strong>⚡ Mode</strong><br>
        {mode or "Chinese Mode"}
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="status-indicator">
        <strong>🧠 Model</strong><br>
        {model or "Auto"}
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="status-indicator">
        <strong>📏 Length</strong><br>
        {max_sentences} sentences
    </div>
    """, unsafe_allow_html=True)

# Enhanced main processing section
if use_sample:
    st.markdown("""
    <div class="card">
        <h3 style="margin-top: 0; color: var(--primary-color);">📁 Sample Document Processing</h3>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        sample_path = Path("data/AI_Transformation_Playbook.pdf")
        if not sample_path.exists():
            st.markdown("""
            <div class="alert-error">
                <strong>❌ Sample file not found</strong><br>
                Please upload a file instead or check if the sample file exists.
            </div>
            """, unsafe_allow_html=True)
            st.stop()
            
        with st.spinner("📖 Loading sample document..."):
            raw_text = load_document(sample_path.open("rb"))
            
        if not raw_text or len(raw_text.strip()) < 50:
            st.markdown("""
            <div class="alert-error">
                <strong>❌ Document Error</strong><br>
                Sample document appears to be empty or corrupted.
            </div>
            """, unsafe_allow_html=True)
            st.stop()
            
        st.markdown(f"""
        <div class="alert-success">
            <strong>✅ Document Loaded Successfully</strong><br>
            Sample document loaded with {len(raw_text):,} characters
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced progress tracking
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create a custom progress section
            progress_col1, progress_col2, progress_col3 = st.columns(3)
            
        try:
            with progress_col1:
                st.markdown("**🤖 Initializing Model**")
            status_text.text("Loading language model...")
            progress_bar.progress(20)
            
            with progress_col2:
                st.markdown("**📝 Generating Summary**")
            status_text.text("Creating summary...")
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
            
            with progress_col3:
                st.markdown("**🔍 Extracting Keywords**")
            status_text.text("Analyzing keywords...")
            progress_bar.progress(80)
            
            status_text.text("✅ Processing complete!")
            progress_bar.progress(100)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            progress_container.empty()
            
        except Exception as e:
            st.markdown(f"""
            <div class="alert-error">
                <strong>❌ Processing Error</strong><br>
                {str(e)}<br>
                Please try a different model or check your input.
            </div>
            """, unsafe_allow_html=True)
            if st.checkbox("🔧 Show Technical Details"):
                st.code(traceback.format_exc())
            st.stop()
            
    except Exception as e:
        st.markdown(f"""
        <div class="alert-error">
            <strong>❌ File Loading Error</strong><br>
            {str(e)}<br>
            Please upload a file instead or check if the sample file exists.
        </div>
        """, unsafe_allow_html=True)
        st.stop()
        
    st.markdown("""
    <div class="alert-info">
        <strong>ℹ️ Demo Mode</strong><br>
        Showing built-in sample. Upload a file to process your own documents.
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="card">
        <h3 style="margin-top: 0; color: var(--primary-color);">📝 Custom Document Processing</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced input method selection
    st.markdown("### 📋 Choose Input Method")
    input_mode = st.radio(
        "Input Method",
        [
            "📝 Paste Text Directly",
            "📁 Upload Document File",
        ],
        horizontal=True,
        key="input_mode_radio"
    )
    
    raw_text = ""
    
    if input_mode == "📝 Paste Text Directly":
        st.markdown("### ✍️ Text Input")
        raw_text = st.text_area(
            "Document Text", 
            height=300,
            placeholder="Paste your text here...\n\nSupports up to 100,000 characters for optimal processing.",
            help="Enter the text you want to summarize",
            key="text_input_area"
        )
        
        if not raw_text.strip():
            st.markdown("""
            <div class="alert-warning">
                <strong>⚠️ Input Required</strong><br>
                Please enter some text to summarize.
            </div>
            """, unsafe_allow_html=True)
            st.stop()
            
        if len(raw_text.strip()) < 50:
            st.markdown("""
            <div class="alert-warning">
                <strong>⚠️ Text Too Short</strong><br>
                Text seems too short for meaningful summarization. Consider adding more content.
            </div>
            """, unsafe_allow_html=True)
            
    else:
        st.markdown("### 📁 File Upload")
        uploaded_file = st.file_uploader(
            "Choose a document file", 
            type=["pdf", "txt", "docx"],
            help="Supported formats: PDF, TXT, DOCX (Max 10MB)",
            key="file_uploader"
        )
        
        if not uploaded_file:
            st.markdown("""
            <div class="alert-info">
                <strong>📁 Ready to Upload</strong><br>
                Please upload a file to get started with summarization.
            </div>
            """, unsafe_allow_html=True)
            st.stop()
            
        try:
            with st.spinner("📖 Loading document..."):
                raw_text = load_document(uploaded_file)
                
            if not raw_text or len(raw_text.strip()) < 50:
                st.markdown("""
                <div class="alert-error">
                    <strong>❌ Document Processing Failed</strong><br>
                    Document appears to be empty or could not be processed.
                </div>
                """, unsafe_allow_html=True)
                st.stop()
                
            st.markdown(f"""
            <div class="alert-success">
                <strong>✅ File Loaded Successfully</strong><br>
                {uploaded_file.name} loaded with {len(raw_text):,} characters
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.markdown(f"""
            <div class="alert-error">
                <strong>❌ File Loading Error</strong><br>
                {str(e)}<br>
                Please try a different file or check the file format.
            </div>
            """, unsafe_allow_html=True)
            if st.checkbox("🔧 Show Technical Details"):
                st.code(traceback.format_exc())
            st.stop()
    
    # Process the text with enhanced UI
    if raw_text:
        st.markdown("### ⚡ Processing Status")
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Enhanced progress display
            progress_col1, progress_col2, progress_col3 = st.columns(3)
            
        try:
            with progress_col1:
                st.markdown("**🤖 Model Initialization**")
            status_text.text("Loading language model...")
            progress_bar.progress(20)
            
            with progress_col2:
                st.markdown("**📝 Summary Generation**")
            status_text.text("Creating summary...")
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
            
            with progress_col3:
                st.markdown("**🔍 Keyword Analysis**")
            status_text.text("Extracting keywords...")
            progress_bar.progress(80)
            
            status_text.text("✅ Processing complete!")
            progress_bar.progress(100)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            progress_container.empty()
            
        except Exception as e:
            st.markdown(f"""
            <div class="alert-error">
                <strong>❌ Processing Error</strong><br>
                {str(e)}<br>
                Please try a different model or check your input.
            </div>
            """, unsafe_allow_html=True)
            if st.checkbox("🔧 Show Technical Details"):
                st.code(traceback.format_exc())
            st.stop()

# Enhanced Results Display
if 'raw_text' in locals() and 'summary' in locals():
    st.markdown("---")
    
    # Enhanced tabs with better organization
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Summary", 
        "🔍 Keywords", 
        "📊 Visualization", 
        "📝 Original Text", 
        "⚡ Performance"
    ])
    
    with tab1:
        st.markdown("""
        <div class="card">
            <h3 style="margin-top: 0; color: var(--primary-color);">📄 Generated Summary</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if summary:
            # Enhanced summary display
            st.markdown(summary)
            
            # Enhanced statistics with better styling
            st.markdown("### 📊 Summary Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin: 0; color: var(--primary-color);">Summary Length</h4>
                    <h2 style="margin: 0.5rem 0; color: var(--text-color);">{len(summary):,} chars</h2>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin: 0; color: var(--primary-color);">Original Length</h4>
                    <h2 style="margin: 0.5rem 0; color: var(--text-color);">{len(raw_text):,} chars</h2>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                compression_ratio = len(summary) / len(raw_text) * 100
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin: 0; color: var(--primary-color);">Compression Ratio</h4>
                    <h2 style="margin: 0.5rem 0; color: var(--text-color);">{compression_ratio:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-error">
                <strong>❌ Summary Generation Failed</strong><br>
                No summary generated. Please check your input and try again.
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <div class="card">
            <h3 style="margin-top: 0; color: var(--primary-color);">🔍 Extracted Keywords</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if keywords:
            if isinstance(keywords, list):
                # Enhanced keyword display with tags
                st.markdown("### 🏷️ Top Keywords")
                keyword_cols = st.columns(3)
                for i, keyword in enumerate(keywords):
                    with keyword_cols[i % 3]:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    color: white; padding: 0.5rem 1rem; border-radius: 20px; 
                                    text-align: center; margin: 0.25rem 0; font-weight: 500;">
                            {keyword}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.write(keywords)
        else:
            st.markdown("""
            <div class="alert-warning">
                <strong>⚠️ No Keywords Found</strong><br>
                No keywords were extracted from the text.
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("""
        <div class="card">
            <h3 style="margin-top: 0; color: var(--primary-color);">📊 Keywords Visualization</h3>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            if keywords:
                if language == "Chinese":
                    fig = plot_chinese_keywords(keywords)
                else:
                    fig = plot_keywords(keywords)
                st.pyplot(fig)
            else:
                st.markdown("""
                <div class="alert-info">
                    <strong>📊 No Data Available</strong><br>
                    No keywords available for visualization.
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f"""
            <div class="alert-error">
                <strong>❌ Visualization Error</strong><br>
                {str(e)}<br>
                Please try again or check your data.
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("""
        <div class="card">
            <h3 style="margin-top: 0; color: var(--primary-color);">📝 Original Text Preview</h3>
        </div>
        """, unsafe_allow_html=True)
        
        preview_length = 1000
        if len(raw_text) > preview_length:
            st.markdown(f"**Showing first {preview_length:,} characters:**")
            st.text_area("Original Text", raw_text[:preview_length] + "...", height=300, disabled=True)
            st.markdown(f"""
            <div class="alert-info">
                <strong>📏 Full Text Length:</strong> {len(raw_text):,} characters
            </div>
            """, unsafe_allow_html=True)
        else:
            st.text_area("Original Text", raw_text, height=300, disabled=True)
    
    with tab5:
        render_performance_dashboard()
    
    # Enhanced export functionality
    st.markdown("---")
    st.markdown("""
    <div class="card">
        <h3 style="margin-top: 0; color: var(--primary-color);">💾 Export Results</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export Summary", key="export_summary"):
            st.download_button(
                label="Download Summary",
                data=summary,
                file_name="summary.txt",
                mime="text/plain",
                key="download_summary"
            )
    
    with col2:
        if st.button("🔍 Export Keywords", key="export_keywords"):
            keywords_text = "\n".join(keywords) if isinstance(keywords, list) else str(keywords)
            st.download_button(
                label="Download Keywords",
                data=keywords_text,
                file_name="keywords.txt",
                mime="text/plain",
                key="download_keywords"
            )
    
    with col3:
        if st.button("📊 Export Full Report", key="export_report"):
            report = f"""AI TEXT SUMMARIZER PRO - ANALYSIS REPORT
===============================================

SUMMARY:
{summary}

KEYWORDS:
{keywords_text if 'keywords_text' in locals() else keywords}

STATISTICS:
- Original Text Length: {len(raw_text):,} characters
- Summary Length: {len(summary):,} characters
- Compression Ratio: {len(summary) / len(raw_text) * 100:.1f}%
- Language: {language}
- Mode: {mode or "Chinese Mode"}
- Model: {model or "Auto"}
- Max Sentences: {max_sentences}

Generated by AI Text Summarizer Pro
"""
            st.download_button(
                label="Download Report",
                data=report,
                file_name="summary_report.txt",
                mime="text/plain",
                key="download_report"
            )

else:
    st.markdown("""
    <div class="alert-info">
        <strong>👆 Ready to Process</strong><br>
        Please process a document above to see results and analysis.
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>🤖 <strong>AI Text Summarizer Pro</strong> - Powered by Advanced Language Models</p>
    <p>Built with ❤️ using Streamlit, Transformers, and modern AI technology</p>
</div>
""", unsafe_allow_html=True)