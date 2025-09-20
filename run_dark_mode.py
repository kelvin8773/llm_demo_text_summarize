#!/usr/bin/env python3
"""
Unified Dark Mode Runner for LLM Text Summarizer
Clean, professional dark theme with enhanced responsive design.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run the unified dark mode version of the LLM Text Summarizer"""
    
    # Get the directory of this script
    script_dir = Path(__file__).parent
    dark_mode_file = script_dir / "main_unified_dark.py"
    
    # Check if the dark mode file exists
    if not dark_mode_file.exists():
        print("❌ Dark mode file not found!")
        print(f"Expected: {dark_mode_file}")
        sys.exit(1)
    
    print("🌙 Starting Unified Dark Mode LLM Text Summarizer...")
    print("🎨 Features:")
    print("   • Clean, professional dark theme")
    print("   • High contrast text for excellent readability")
    print("   • Enhanced responsive design")
    print("   • Unified color scheme throughout")
    print("   • Better sidebar configuration")
    print("   • Improved form controls and interactions")
    print()
    
    try:
        # Run the dark mode version
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dark_mode_file),
            "--server.headless", "false",
            "--server.port", "8504",
            "--server.address", "localhost"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running dark mode UI: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Dark mode UI stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()