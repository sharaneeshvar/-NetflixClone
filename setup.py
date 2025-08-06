#!/usr/bin/env python3
"""
Setup script for Advanced Prompting Studio
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version {sys.version.split()[0]} is compatible")

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        sys.exit(1)

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        print("📝 Creating .env file from template...")
        with open(env_example, 'r') as src, open(env_file, 'w') as dst:
            dst.write(src.read())
        print("✅ .env file created. Please edit it with your API keys.")
    elif env_file.exists():
        print("ℹ️  .env file already exists")
    else:
        print("⚠️  .env.example not found")

def check_streamlit():
    """Check if Streamlit is installed and working"""
    print("🔍 Checking Streamlit installation...")
    try:
        import streamlit as st
        print(f"✅ Streamlit {st.__version__} is installed")
    except ImportError:
        print("❌ Streamlit not found")
        sys.exit(1)

def print_next_steps():
    """Print instructions for next steps"""
    print("\n" + "="*50)
    print("🎉 Setup completed successfully!")
    print("="*50)
    print("\n📋 Next steps:")
    print("1. Edit the .env file with your API keys:")
    print("   - Get Gemini API key from: https://makersuite.google.com/app/apikey")
    print("   - Get Langfuse keys from: https://langfuse.com (optional)")
    print("\n2. Run the application:")
    print("   streamlit run app.py")
    print("\n3. Open your browser and navigate to:")
    print("   http://localhost:8501")
    print("\n🔧 Configuration:")
    print("   - Enter your API keys in the sidebar")
    print("   - Enable Langfuse tracking (optional)")
    print("   - Adjust model parameters as needed")
    print("\n📖 For more information, see README.md")

def main():
    """Main setup function"""
    print("🚀 Advanced Prompting Studio Setup")
    print("=" * 40)
    
    # Check Python version
    check_python_version()
    
    # Install requirements
    install_requirements()
    
    # Create .env file
    create_env_file()
    
    # Check Streamlit
    check_streamlit()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()