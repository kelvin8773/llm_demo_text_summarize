#!/usr/bin/env python3
"""
Enhanced UI Runner for LLM Text Summarizer
This script runs the enhanced UI version with better styling and functionality.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run the enhanced UI version of the LLM Text Summarizer"""
    
    # Get the directory of this script
    script_dir = Path(__file__).parent
    enhanced_ui_file = script_dir / "main_enhanced_ui.py"
    
    # Check if the enhanced UI file exists
    if not enhanced_ui_file.exists():
        print("❌ Enhanced UI file not found!")
        print(f"Expected: {enhanced_ui_file}")
        sys.exit(1)
    
    print("🚀 Starting Enhanced UI LLM Text Summarizer...")
    print("📄 Features:")
    print("   • Enhanced sidebar with better visibility")
    print("   • Improved color scheme and contrast")
    print("   • Dark mode toggle")
    print("   • Better responsive design")
    print("   • Enhanced animations and transitions")
    print("   • Improved text visibility")
    print()
    
    try:
        # Run the enhanced UI
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(enhanced_ui_file),
            "--server.headless", "false",
            "--server.port", "8503",
            "--server.address", "localhost"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running enhanced UI: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Enhanced UI stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()