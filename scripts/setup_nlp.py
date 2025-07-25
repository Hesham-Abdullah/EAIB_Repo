#!/usr/bin/env python3
"""
Setup script to install required NLP models and data
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up NLP models and data for text processing...")
    
    # Install spaCy English model
    success = run_command(
        "python -m spacy download en_core_web_sm",
        "Installing spaCy English model (en_core_web_sm)"
    )
    
    if not success:
        print("⚠️  Failed to install spaCy model. You can install it manually with:")
        print("   python -m spacy download en_core_web_sm")
    
    # Download NLTK data
    print("\n🔄 Downloading NLTK data...")
    try:
        import nltk
        
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        
        print("✅ NLTK data downloaded successfully")
    except Exception as e:
        print(f"❌ NLTK data download failed: {e}")
    
    # Test sentence transformers installation
    print("\n🔄 Testing sentence transformers...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Sentence transformers working correctly")
    except Exception as e:
        print(f"❌ Sentence transformers test failed: {e}")
        print("⚠️  The model will be downloaded on first use")
    
    print("\n🎯 Setup Summary:")
    print("Required dependencies for text processing:")
    print("  - spacy (with en_core_web_sm model)")
    print("  - scikit-learn")
    print("  - sentence-transformers")
    print("  - langdetect")
    print("  - ftfy")
    print("  - nltk (with data)")
    print("  - textstat")
    
    print("\n📚 Usage:")
    print("After setup, you can process documents with:")
    print("  curl -X POST 'http://localhost:8000/process-document/your-document-id'")
    
    print("\n✅ Setup completed!")

if __name__ == "__main__":
    main() 