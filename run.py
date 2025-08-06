#!/usr/bin/env python3
"""
Quick run script for Advanced Prompting Studio
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if requirements are installed"""
    try:
        import streamlit
        import google.generativeai
        import langfuse
        import pandas
        import plotly
        print("✅ All requirements are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing requirement: {e}")
        return False

def run_app():
    """Run the Streamlit application"""
    if not check_requirements():
        print("\n🔧 Please run setup first:")
        print("   python setup.py")
        sys.exit(1)
    
    print("🚀 Starting Advanced Prompting Studio...")
    print("📍 Navigate to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped")
    except Exception as e:
        print(f"❌ Error running application: {e}")

if __name__ == "__main__":
    run_app()