# ui_components.py - Enhanced UI Components for Streamlit
import streamlit as st
import time
from typing import List, Dict, Any, Optional
import json

class UIComponents:
    """Enhanced UI components for better user experience"""
    
    @staticmethod
    def create_hero_section(title: str, subtitle: str, icon: str = "🤖"):
        """Create an enhanced hero section"""
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 3rem 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        ">
            <h1 style="font-size: 3rem; font-weight: 700; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">
                {icon} {title}
            </h1>
            <p style="font-size: 1.2rem; margin: 1rem 0 0 0; opacity: 0.9;">
                {subtitle}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_info_card(title: str, content: str, icon: str = "ℹ️", card_type: str = "info"):
        """Create styled information cards"""
        colors = {
            "info": "linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%)",
            "success": "linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%)",
            "warning": "linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%)",
            "error": "linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%)"
        }
        
        text_colors = {
            "info": "#0c5460",
            "success": "#155724",
            "warning": "#856404",
            "error": "#721c24"
        }
        
        st.markdown(f"""
        <div style="
            background: {colors[card_type]};
            border: 1px solid rgba(0,0,0,0.1);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            color: {text_colors[card_type]};
        ">
            <h4 style="margin: 0 0 0.5rem 0; font-weight: 600;">
                {icon} {title}
            </h4>
            <p style="margin: 0; line-height: 1.5;">
                {content}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_metric_card(title: str, value: str, subtitle: str = "", icon: str = "📊"):
        """Create enhanced metric cards"""
        st.markdown(f"""
        <div style="
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            border-left: 4px solid #667eea;
            margin: 0.5rem 0;
        ">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
            <h3 style="margin: 0; color: #2c3e50; font-size: 1.8rem;">{value}</h3>
            <h4 style="margin: 0.5rem 0 0 0; color: #667eea; font-weight: 600;">{title}</h4>
            {f'<p style="margin: 0.5rem 0 0 0; color: #666; font-size: 0.9rem;">{subtitle}</p>' if subtitle else ''}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_progress_section(steps: List[str], current_step: int = 0):
        """Create an enhanced progress section"""
        st.markdown("### ⚡ Processing Progress")
        
        # Create progress columns
        cols = st.columns(len(steps))
        
        for i, step in enumerate(steps):
            with cols[i]:
                if i <= current_step:
                    # Completed or current step
                    color = "#667eea" if i == current_step else "#2ca02c"
                    icon = "🔄" if i == current_step else "✅"
                else:
                    # Future step
                    color = "#e9ecef"
                    icon = "⏳"
                
                st.markdown(f"""
                <div style="
                    background: {color};
                    color: white;
                    padding: 1rem;
                    border-radius: 10px;
                    text-align: center;
                    margin: 0.25rem 0;
                    font-weight: 500;
                ">
                    <div style="font-size: 1.5rem;">{icon}</div>
                    <div style="font-size: 0.9rem;">{step}</div>
                </div>
                """, unsafe_allow_html=True)
    
    @staticmethod
    def create_keyword_tags(keywords: List[str], max_display: int = 15):
        """Create styled keyword tags"""
        if not keywords:
            return
        
        st.markdown("### 🏷️ Extracted Keywords")
        
        # Display keywords in a grid
        cols = st.columns(3)
        for i, keyword in enumerate(keywords[:max_display]):
            with cols[i % 3]:
                # Color gradient based on position
                colors = ["#667eea", "#764ba2", "#f093fb"]
                color = colors[i % 3]
                
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {color} 0%, {color}dd 100%);
                    color: white;
                    padding: 0.5rem 1rem;
                    border-radius: 20px;
                    text-align: center;
                    margin: 0.25rem 0;
                    font-weight: 500;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                ">
                    {keyword}
                </div>
                """, unsafe_allow_html=True)
    
    @staticmethod
    def create_file_upload_area():
        """Create enhanced file upload area"""
        st.markdown("""
        <div style="
            border: 2px dashed #667eea;
            border-radius: 15px;
            padding: 3rem 2rem;
            text-align: center;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            margin: 1rem 0;
            transition: all 0.3s ease;
        ">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📁</div>
            <h3 style="color: #667eea; margin: 0;">Drag & Drop Your File Here</h3>
            <p style="color: #666; margin: 0.5rem 0 0 0;">
                Supports PDF, TXT, DOCX files up to 10MB
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_loading_animation(message: str = "Processing..."):
        """Create enhanced loading animation"""
        st.markdown(f"""
        <div style="
            text-align: center;
            padding: 2rem;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 15px;
            margin: 1rem 0;
        ">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🤖</div>
            <h3 style="color: #667eea; margin: 0;">{message}</h3>
            <div style="margin-top: 1rem;">
                <div style="
                    width: 40px;
                    height: 40px;
                    border: 4px solid #e9ecef;
                    border-top: 4px solid #667eea;
                    border-radius: 50%;
                    animation: spin 1s linear infinite;
                    margin: 0 auto;
                "></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_settings_panel():
        """Create enhanced settings panel"""
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid rgba(0,0,0,0.1);
        ">
            <h3 style="margin-top: 0; color: #667eea; text-align: center;">
                ⚙️ Configuration Panel
            </h3>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_export_section():
        """Create enhanced export section"""
        st.markdown("""
        <div style="
            background: white;
            border-radius: 15px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            border: 1px solid rgba(0,0,0,0.1);
        ">
            <h3 style="margin-top: 0; color: #667eea; text-align: center;">
                💾 Export Your Results
            </h3>
            <p style="text-align: center; color: #666; margin-bottom: 1.5rem;">
                Download your summary, keywords, and full analysis report
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_footer():
        """Create enhanced footer"""
        st.markdown("""
        <div style="
            text-align: center;
            color: #666;
            padding: 2rem 0;
            border-top: 1px solid #e9ecef;
            margin-top: 3rem;
        ">
            <p style="margin: 0; font-size: 1.1rem;">
                🤖 <strong>AI Text Summarizer Pro</strong> - Powered by Advanced Language Models
            </p>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                Built with ❤️ using Streamlit, Transformers, and modern AI technology
            </p>
        </div>
        """, unsafe_allow_html=True)

class ThemeManager:
    """Manage UI themes and styling"""
    
    @staticmethod
    def apply_light_theme():
        """Apply light theme"""
        st.markdown("""
        <style>
        .main {
            background-color: #f8f9fa;
        }
        .stApp {
            background-color: #f8f9fa;
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def apply_dark_theme():
        """Apply dark theme"""
        st.markdown("""
        <style>
        .main {
            background-color: #1a1a1a;
            color: #ffffff;
        }
        .stApp {
            background-color: #1a1a1a;
        }
        .card {
            background-color: #2d2d2d;
            color: #ffffff;
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def get_theme_config():
        """Get theme configuration"""
        return {
            "light": {
                "primary": "#667eea",
                "secondary": "#764ba2",
                "background": "#f8f9fa",
                "card": "#ffffff",
                "text": "#2c3e50"
            },
            "dark": {
                "primary": "#8b9dc3",
                "secondary": "#9b59b6",
                "background": "#1a1a1a",
                "card": "#2d2d2d",
                "text": "#ffffff"
            }
        }

class AnimationManager:
    """Manage UI animations and transitions"""
    
    @staticmethod
    def add_fade_in_animation():
        """Add fade-in animation CSS"""
        st.markdown("""
        <style>
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .fade-in-up {
            animation: fadeInUp 0.8s ease-out;
        }
        
        .fade-in-up-delayed {
            animation: fadeInUp 0.8s ease-out 0.2s both;
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def add_slide_in_animation():
        """Add slide-in animation CSS"""
        st.markdown("""
        <style>
        @keyframes slideInLeft {
            from {
                opacity: 0;
                transform: translateX(-30px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        .slide-in-left {
            animation: slideInLeft 0.6s ease-out;
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def add_pulse_animation():
        """Add pulse animation CSS"""
        st.markdown("""
        <style>
        @keyframes pulse {
            0% {
                transform: scale(1);
            }
            50% {
                transform: scale(1.05);
            }
            100% {
                transform: scale(1);
            }
        }
        
        .pulse {
            animation: pulse 2s infinite;
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def add_spin_animation():
        """Add spin animation CSS"""
        st.markdown("""
        <style>
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .spin {
            animation: spin 1s linear infinite;
        }
        </style>
        """, unsafe_allow_html=True)