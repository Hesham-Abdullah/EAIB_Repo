#!/usr/bin/env python3
"""
Script to start the Streamlit interface for the EAIB pipeline
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Start the Streamlit interface"""
    print("🚀 Starting EAIB Streamlit Interface...")
    print("📋 Make sure the API server is running first:")
    print("   python scripts/start_api.py")
    print()
    
    # Get the project root
    project_root = Path(__file__).parent.parent
    streamlit_app_path = project_root / "src" / "streamlit_app.py"
    
    if not streamlit_app_path.exists():
        print(f"❌ Streamlit app not found at: {streamlit_app_path}")
        sys.exit(1)
    
    # Start Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(streamlit_app_path),
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Streamlit interface stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 