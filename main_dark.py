# main_dark.py - Dark Theme UI for LLM Text Summarization Tool
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
from utils.ui_components import UIComponents, ThemeManager, AnimationManager

# Enhanced page configuration
st.set_page_config(
    page_title="AI Text Summarizer Pro - Dark Mode",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/kelvin8773/llm_demo_text_summarize',
        'Report a bug': "https://github.com/kelvin8773/llm_demo_text_summarize/issues",
        'About': "# AI Text Summarizer Pro\nPowered by advanced language models"
    }
)

# Dark theme CSS with modern design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Dark theme CSS variables */
    :root {
        --primary-color: #8b9dc3;
        --secondary-color: #9b59b6;
        --accent-color: #3498db;
        --success-color: #2ecc71;
        --warning-color: #f39c12;
        --error-color: #e74c3c;
        --background-color: #1a1a1a;
        --surface-color: #2d2d2d;
        --card-background: #2d2d2d;
        --text-color: #ffffff;
        --text-secondary: #b0b0b0;
        --border-color: #404040;
        --shadow: 0 4px 20px rgba(0,0,0,0.3);
        --shadow-hover: 0 8px 30px rgba(0,0,0,0.4);
        --border-radius: 12px;
        --transition: all 0.3s ease;
    }
    
    /* Override Streamlit's default dark theme */
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    .main {
        font-family: 'Inter', sans-serif;
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    /* Dark theme hero section */
    .hero-section {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 50%, #2c3e50 100%);
        padding: 3rem 2rem;
        border-radius: var(--border-radius);
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: var(--shadow);
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        margin: 1rem 0 0 0;
        opacity: 0.9;
        position: relative;
        z-index: 1;
    }
    
    /* Dark theme cards */
    .card {
        background: var(--card-background);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow);
        border: 1px solid var(--border-color);
        transition: var(--transition);
        color: var(--text-color);
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-hover);
        border-color: var(--primary-color);
    }
    
    /* Dark theme status indicators */
    .status-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.5rem;
        margin: 1rem 0;
    }
    
    .status-indicator {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        padding: 0.75rem;
        border-radius: 20px;
        font-weight: 500;
        text-align: center;
        font-size: 0.9rem;
        box-shadow: var(--shadow);
    }
    
    .status-indicator .label {
        font-size: 0.8rem;
        opacity: 0.8;
        margin-bottom: 0.25rem;
    }
    
    .status-indicator .value {
        font-size: 1rem;
        font-weight: 600;
    }
    
    /* Dark theme buttons */
    .btn {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: var(--transition);
        box-shadow: var(--shadow);
        cursor: pointer;
        width: 100%;
        margin: 0.25rem 0;
    }
    
    .btn:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-hover);
    }
    
    /* Dark theme alerts */
    .alert {
        border-radius: var(--border-radius);
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid;
    }
    
    .alert-success {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        border-color: #2ecc71;
        color: white;
    }
    
    .alert-error {
        background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%);
        border-color: #e74c3c;
        color: white;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #d68910 0%, #f39c12 100%);
        border-color: #f39c12;
        color: white;
    }
    
    .alert-info {
        background: linear-gradient(135deg, #2980b9 0%, #3498db 100%);
        border-color: #3498db;
        color: white;
    }
    
    /* Dark theme metrics */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: var(--card-background);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        text-align: center;
        box-shadow: var(--shadow);
        border-left: 4px solid var(--primary-color);
        color: var(--text-color);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-color);
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: var(--text-secondary);
        font-weight: 600;
        margin: 0;
    }
    
    /* Dark theme keyword tags */
    .keywords-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 0.5rem;
        margin: 1rem 0;
    }
    
    .keyword-tag {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        text-align: center;
        font-weight: 500;
        font-size: 0.9rem;
        box-shadow: var(--shadow);
        transition: var(--transition);
    }
    
    .keyword-tag:hover {
        transform: scale(1.05);
        box-shadow: var(--shadow-hover);
    }
    
    /* Dark theme file upload */
    .file-upload-area {
        border: 2px dashed var(--border-color);
        border-radius: var(--border-radius);
        padding: 2rem 1rem;
        text-align: center;
        background: var(--surface-color);
        transition: var(--transition);
        margin: 1rem 0;
    }
    
    .file-upload-area:hover {
        border-color: var(--primary-color);
        background: rgba(139, 157, 195, 0.1);
    }
    
    .file-upload-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        color: var(--primary-color);
    }
    
    /* Dark theme progress */
    .progress-section {
        margin: 1rem 0;
    }
    
    .progress-steps {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.5rem;
        margin: 1rem 0;
    }
    
    .progress-step {
        background: var(--card-background);
        border-radius: var(--border-radius);
        padding: 1rem;
        text-align: center;
        box-shadow: var(--shadow);
        transition: var(--transition);
        color: var(--text-color);
    }
    
    .progress-step.active {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
    }
    
    .progress-step.completed {
        background: var(--success-color);
        color: white;
    }
    
    /* Dark theme sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--surface-color) 0%, var(--background-color) 100%);
    }
    
    /* Dark theme form elements */
    .stSelectbox > div > div {
        background-color: var(--card-background);
        border-color: var(--border-color);
        color: var(--text-color);
    }
    
    .stTextArea > div > div > textarea {
        background-color: var(--card-background);
        border-color: var(--border-color);
        color: var(--text-color);
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 2px rgba(139, 157, 195, 0.2);
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    }
    
    /* Dark theme tabs */
    .stTabs [data-baseweb="tab"] {
        background: var(--card-background);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: 1px solid var(--border-color);
        transition: var(--transition);
        color: var(--text-color);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
    }
    
    /* Dark theme progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    }
    
    /* Dark theme checkbox */
    .stCheckbox > div > div {
        background-color: var(--card-background);
    }
    
    /* Dark theme radio */
    .stRadio > div > div {
        background-color: var(--card-background);
    }
    
    /* Dark theme file uploader */
    .stFileUploader > div {
        background-color: var(--card-background);
        border-color: var(--border-color);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom animations for dark theme */
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
    
    @keyframes glow {
        0%, 100% {
            box-shadow: 0 0 5px var(--primary-color);
        }
        50% {
            box-shadow: 0 0 20px var(--primary-color), 0 0 30px var(--primary-color);
        }
    }
    
    .glow {
        animation: glow 2s ease-in-out infinite;
    }
    
    /* Responsive design for dark theme */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2rem;
        }
        
        .hero-subtitle {
            font-size: 1rem;
        }
        
        .card {
            padding: 1rem;
        }
        
        .status-indicator {
            padding: 0.5rem;
            font-size: 0.8rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
        
        .metric-value {
            font-size: 1.5rem;
        }
        
        .file-upload-area {
            padding: 1.5rem 0.75rem;
        }
        
        .file-upload-icon {
            font-size: 2rem;
        }
        
        .keywords-grid {
            grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
        }
        
        .keyword-tag {
            font-size: 0.8rem;
            padding: 0.4rem 0.8rem;
        }
        
        .progress-steps {
            grid-template-columns: 1fr;
        }
    }
    
    /* Loading animation for dark theme */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loading-spinner {
        width: 40px;
        height: 40px;
        border: 4px solid var(--border-color);
        border-top: 4px solid var(--primary-color);
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

# Initialize UI components
ui = UIComponents()
theme_manager = ThemeManager()
animation_manager = AnimationManager()

# Apply dark theme
theme_manager.apply_dark_theme()

# Add animations
animation_manager.add_fade_in_animation()
animation_manager.add_slide_in_animation()
animation_manager.add_pulse_animation()

# Enhanced dark theme header
st.markdown("""
<div class="hero-section fade-in-up">
    <h1 class="hero-title">🌙 AI Text Summarizer Pro</h1>
    <p class="hero-subtitle">Dark Mode - Transform lengthy documents into concise summaries with intelligent keyword extraction</p>
</div>
""", unsafe_allow_html=True)

# Performance alerts
render_performance_alerts()

# Dark theme sidebar
with st.sidebar:
    st.markdown("""
    <div class="card">
        <h3 style="margin-top: 0; color: var(--primary-color);">⚙️ Configuration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Language selection
    st.markdown("### 🌐 Language Selection")
    language = st.selectbox(
        "Select Language", 
        ["English", "Chinese"], 
        help="Choose the language of your input text",
        key="language_select"
    )
    
    # Mode selection
    if language == "Chinese":
        mode = None
        st.markdown("""
        <div class="alert alert-info">
            <strong>🇨🇳 Chinese Mode</strong><br>
            Uses specialized Chinese language models for optimal results
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("### 🚀 Summarization Mode")
        mode = st.selectbox(
            "Choose Mode", 
            ["Fast Summarizer", "Enhanced Summarizer"],
            help="Fast: Quick processing. Enhanced: Detailed analysis",
            key="mode_select"
        )
    
    # Model selection
    if mode == "Fast Summarizer":
        st.markdown("### 🧠 AI Model Selection")
        model_options = {
            BART_CNN_MODEL: "📰 BART-CNN (News & Formal)",
            T5_LARGE_MODEL: "🔬 T5-Large (Versatile)"
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
    
    # Summary length
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
    
    # Input options
    st.markdown("### 📁 Input Options")
    use_sample = st.checkbox(
        "🎯 Use Sample Document", 
        value=True, 
        help="Process the built-in AI Transformation Playbook PDF",
        key="sample_checkbox"
    )
    
    # Theme toggle
    st.markdown("### 🎨 Theme Options")
    if st.button("🌞 Switch to Light Mode", key="theme_toggle"):
        st.experimental_rerun()
    
    # Performance widget
    render_performance_widget()

# Dark theme status indicators
st.markdown("### 📊 Current Configuration")
st.markdown("""
<div class="status-grid">
    <div class="status-indicator">
        <div class="label">🌐 Language</div>
        <div class="value">""" + language + """</div>
    </div>
    <div class="status-indicator">
        <div class="label">⚡ Mode</div>
        <div class="value">""" + (mode or "Chinese Mode") + """</div>
    </div>
    <div class="status-indicator">
        <div class="label">🧠 Model</div>
        <div class="value">""" + (model or "Auto") + """</div>
    </div>
    <div class="status-indicator">
        <div class="label">📏 Length</div>
        <div class="value">""" + str(max_sentences) + """ sentences</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main processing section with dark theme
if use_sample:
    st.markdown("""
    <div class="card fade-in-up">
        <h3 style="margin-top: 0; color: var(--primary-color);">📁 Sample Document Processing</h3>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        sample_path = Path("data/AI_Transformation_Playbook.pdf")
        if not sample_path.exists():
            st.markdown("""
            <div class="alert alert-error">
                <strong>❌ Sample file not found</strong><br>
                Please upload a file instead or check if the sample file exists.
            </div>
            """, unsafe_allow_html=True)
            st.stop()
            
        with st.spinner("📖 Loading sample document..."):
            raw_text = load_document(sample_path.open("rb"))
            
        if not raw_text or len(raw_text.strip()) < 50:
            st.markdown("""
            <div class="alert alert-error">
                <strong>❌ Document Error</strong><br>
                Sample document appears to be empty or corrupted.
            </div>
            """, unsafe_allow_html=True)
            st.stop()
            
        st.markdown(f"""
        <div class="alert alert-success">
            <strong>✅ Document Loaded Successfully</strong><br>
            Sample document loaded with {len(raw_text):,} characters
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced progress tracking
        st.markdown("### ⚡ Processing Status")
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
            st.markdown(f"""
            <div class="alert alert-error">
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
        <div class="alert alert-error">
            <strong>❌ File Loading Error</strong><br>
            {str(e)}<br>
            Please upload a file instead or check if the sample file exists.
        </div>
        """, unsafe_allow_html=True)
        st.stop()
        
    st.markdown("""
    <div class="alert alert-info">
        <strong>ℹ️ Demo Mode</strong><br>
        Showing built-in sample. Upload a file to process your own documents.
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="card fade-in-up">
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
            <div class="alert alert-warning">
                <strong>⚠️ Input Required</strong><br>
                Please enter some text to summarize.
            </div>
            """, unsafe_allow_html=True)
            st.stop()
            
        if len(raw_text.strip()) < 50:
            st.markdown("""
            <div class="alert alert-warning">
                <strong>⚠️ Text Too Short</strong><br>
                Text seems too short for meaningful summarization. Consider adding more content.
            </div>
            """, unsafe_allow_html=True)
            
    else:
        st.markdown("### 📁 File Upload")
        st.markdown("""
        <div class="file-upload-area">
            <div class="file-upload-icon">📁</div>
            <h3 style="color: var(--primary-color); margin: 0;">Drag & Drop Your File Here</h3>
            <p style="color: var(--text-secondary); margin: 0.5rem 0 0 0;">
                Supports PDF, TXT, DOCX files up to 10MB
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a document file", 
            type=["pdf", "txt", "docx"],
            help="Supported formats: PDF, TXT, DOCX (Max 10MB)",
            key="file_uploader"
        )
        
        if not uploaded_file:
            st.markdown("""
            <div class="alert alert-info">
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
                <div class="alert alert-error">
                    <strong>❌ Document Processing Failed</strong><br>
                    Document appears to be empty or could not be processed.
                </div>
                """, unsafe_allow_html=True)
                st.stop()
                
            st.markdown(f"""
            <div class="alert alert-success">
                <strong>✅ File Loaded Successfully</strong><br>
                {uploaded_file.name} loaded with {len(raw_text):,} characters
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.markdown(f"""
            <div class="alert alert-error">
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
            st.markdown(f"""
            <div class="alert alert-error">
                <strong>❌ Processing Error</strong><br>
                {str(e)}<br>
                Please try a different model or check your input.
            </div>
            """, unsafe_allow_html=True)
            if st.checkbox("🔧 Show Technical Details"):
                st.code(traceback.format_exc())
            st.stop()

# Enhanced Results Display with dark theme
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
        <div class="card fade-in-up">
            <h3 style="margin-top: 0; color: var(--primary-color);">📄 Generated Summary</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if summary:
            # Enhanced summary display
            st.markdown(summary)
            
            # Enhanced statistics with dark theme
            st.markdown("### 📊 Summary Statistics")
            st.markdown(f"""
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{len(summary):,}</div>
                    <div class="metric-label">Summary Length (chars)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(raw_text):,}</div>
                    <div class="metric-label">Original Length (chars)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(summary) / len(raw_text) * 100:.1f}%</div>
                    <div class="metric-label">Compression Ratio</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert alert-error">
                <strong>❌ Summary Generation Failed</strong><br>
                No summary generated. Please check your input and try again.
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <div class="card fade-in-up">
            <h3 style="margin-top: 0; color: var(--primary-color);">🔍 Extracted Keywords</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if keywords:
            if isinstance(keywords, list):
                # Enhanced keyword display with dark theme
                st.markdown("### 🏷️ Top Keywords")
                keywords_html = ""
                for keyword in keywords:
                    keywords_html += f'<div class="keyword-tag">{keyword}</div>'
                
                st.markdown(f"""
                <div class="keywords-grid">
                    {keywords_html}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.write(keywords)
        else:
            st.markdown("""
            <div class="alert alert-warning">
                <strong>⚠️ No Keywords Found</strong><br>
                No keywords were extracted from the text.
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("""
        <div class="card fade-in-up">
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
                <div class="alert alert-info">
                    <strong>📊 No Data Available</strong><br>
                    No keywords available for visualization.
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f"""
            <div class="alert alert-error">
                <strong>❌ Visualization Error</strong><br>
                {str(e)}<br>
                Please try again or check your data.
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("""
        <div class="card fade-in-up">
            <h3 style="margin-top: 0; color: var(--primary-color);">📝 Original Text Preview</h3>
        </div>
        """, unsafe_allow_html=True)
        
        preview_length = 1000
        if len(raw_text) > preview_length:
            st.markdown(f"**Showing first {preview_length:,} characters:**")
            st.text_area("Original Text", raw_text[:preview_length] + "...", height=300, disabled=True)
            st.markdown(f"""
            <div class="alert alert-info">
                <strong>📏 Full Text Length:</strong> {len(raw_text):,} characters
            </div>
            """, unsafe_allow_html=True)
        else:
            st.text_area("Original Text", raw_text, height=300, disabled=True)
    
    with tab5:
        render_performance_dashboard()
    
    # Enhanced export functionality
    ui.create_export_section()
    
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
            report = f"""AI TEXT SUMMARIZER PRO - ANALYSIS REPORT (DARK MODE)
===============================================================

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
- Theme: Dark Mode

Generated by AI Text Summarizer Pro
"""
            st.download_button(
                label="Download Report",
                data=report,
                file_name="summary_report_dark.txt",
                mime="text/plain",
                key="download_report"
            )

else:
    st.markdown("""
    <div class="alert alert-info">
        <strong>👆 Ready to Process</strong><br>
        Please process a document above to see results and analysis.
    </div>
    """, unsafe_allow_html=True)

# Enhanced dark theme footer
st.markdown("""
<div style="
    text-align: center;
    color: var(--text-secondary);
    padding: 2rem 0;
    border-top: 1px solid var(--border-color);
    margin-top: 3rem;
">
    <p style="margin: 0; font-size: 1.1rem;">
        🌙 <strong>AI Text Summarizer Pro - Dark Mode</strong> - Powered by Advanced Language Models
    </p>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">
        Built with ❤️ using Streamlit, Transformers, and modern AI technology
    </p>
</div>
""", unsafe_allow_html=True)