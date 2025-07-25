#!/usr/bin/env python3
"""
EAIB Unified Application Launcher
Starts both the API server and Streamlit interface
"""

import subprocess
import sys
import time
import threading
import signal
import os
from pathlib import Path
import requests
import webbrowser

class EAIBApp:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.api_process = None
        self.streamlit_process = None
        self.running = True
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n🛑 Shutting down EAIB application...")
        self.running = False
        self.cleanup()
        sys.exit(0)
    
    def check_dependencies(self):
        """Check if required dependencies are installed"""
        print("🔍 Checking dependencies...")
        
        try:
            import streamlit
            import fastapi
            import uvicorn
            print("✅ All dependencies are installed")
            return True
        except ImportError as e:
            print(f"❌ Missing dependency: {e}")
            print("💡 Please install dependencies with: pip install -r requirements.txt")
            return False
    
    def check_environment(self):
        """Check environment setup"""
        print("🔍 Checking environment...")
        
        # Check if .env file exists
        env_file = self.project_root / ".env"
        if not env_file.exists():
            print("⚠️  .env file not found. Please copy env.example to .env and configure your API keys.")
            print("💡 Run: cp env.example .env")
            return False
        
        # Check if config file exists
        config_file = self.project_root / "config" / "data_config.yaml"
        if not config_file.exists():
            print("❌ config/data_config.yaml not found")
            return False
        
        print("✅ Environment setup looks good")
        return True
    
    def start_api_server(self):
        """Start the FastAPI server"""
        print("🚀 Starting API server...")
        
        api_script = self.project_root / "scripts" / "start_api.py"
        if not api_script.exists():
            print(f"❌ API script not found: {api_script}")
            return False
        
        try:
            self.api_process = subprocess.Popen([
                sys.executable, str(api_script)
            ], stdout=None, stderr=None)
            
            # Wait a bit for the server to start
            time.sleep(3)
            
            # Check if server is running
            try:
                response = requests.get("http://localhost:8000/", timeout=5)
                if response.status_code == 200:
                    print("✅ API server is running on http://localhost:8000")
                    return True
                else:
                    print(f"❌ API server returned status {response.status_code}")
                    return False
            except requests.exceptions.ConnectionError:
                print("❌ API server failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Error starting API server: {e}")
            return False
    
    def start_streamlit(self):
        """Start the Streamlit interface"""
        print("🚀 Starting Streamlit interface...")
        
        streamlit_script = self.project_root / "src" / "streamlit_app.py"
        print(f"🔍 Looking for Streamlit app at: {streamlit_script}")
        print(f"📁 Script exists: {streamlit_script.exists()}")
        if not streamlit_script.exists():
            print(f"❌ Streamlit app not found: {streamlit_script}")
            return False
        
        try:
            print(f"🚀 Starting Streamlit with command: {sys.executable} -m streamlit run {streamlit_script}")
            print(f"📁 Working directory: {self.project_root}")
            
            # Change to project root directory
            original_cwd = os.getcwd()
            os.chdir(self.project_root)
            
            self.streamlit_process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", str(streamlit_script),
                "--server.port", "8501",
                "--server.address", "0.0.0.0"
            ], stdout=None, stderr=None, cwd=self.project_root)
            
            # Restore original working directory
            os.chdir(original_cwd)
            
            print("⏳ Waiting for Streamlit to start...")
            # Wait a bit for Streamlit to start
            time.sleep(5)
            
            # Check if Streamlit is running
            try:
                response = requests.get("http://localhost:8501", timeout=5)
                if response.status_code == 200:
                    print("✅ Streamlit interface is running on http://localhost:8501")
                    return True
                else:
                    print(f"❌ Streamlit returned status {response.status_code}")
                    return False
            except requests.exceptions.ConnectionError:
                print("❌ Streamlit interface failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Error starting Streamlit: {e}")
            return False
    
    def open_browser(self):
        """Open the Streamlit interface in browser"""
        print("🌐 Opening browser...")
        try:
            webbrowser.open("http://localhost:8501")
            print("✅ Browser opened successfully")
        except Exception as e:
            print(f"⚠️  Could not open browser automatically: {e}")
            print("💡 Please manually open: http://localhost:8501")
    
    def monitor_processes(self):
        """Monitor running processes"""
        while self.running:
            # Check API server
            if self.api_process and self.api_process.poll() is not None:
                print("❌ API server stopped unexpectedly")
                self.running = False
                break
            
            # Check Streamlit
            if self.streamlit_process and self.streamlit_process.poll() is not None:
                print("❌ Streamlit interface stopped unexpectedly")
                self.running = False
                break
            
            time.sleep(5)
    
    def cleanup(self):
        """Clean up running processes"""
        print("🧹 Cleaning up processes...")
        
        if self.api_process:
            self.api_process.terminate()
            try:
                self.api_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.api_process.kill()
            print("✅ API server stopped")
        
        if self.streamlit_process:
            self.streamlit_process.terminate()
            try:
                self.streamlit_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.streamlit_process.kill()
            print("✅ Streamlit interface stopped")
    
    def run(self):
        """Main application runner"""
        print("🚀 EAIB Unified Application Launcher")
        print("=" * 50)
        
        # Check dependencies
        if not self.check_dependencies():
            return False
        
        # Check environment
        if not self.check_environment():
            return False
        
        # Start API server
        if not self.start_api_server():
            return False
        
        # Start Streamlit interface
        if not self.start_streamlit():
            return False
        
        # Open browser
        self.open_browser()
        
        print("\n🎉 EAIB application is running!")
        print("📋 Services:")
        print("   • API Server: http://localhost:8000")
        print("   • Streamlit Interface: http://localhost:8501")
        print("   • API Documentation: http://localhost:8000/docs")
        print("\n💡 Press Ctrl+C to stop the application")
        
        # Monitor processes
        try:
            self.monitor_processes()
        except KeyboardInterrupt:
            print("\n🛑 Received interrupt signal")
        
        self.cleanup()
        print("👋 EAIB application stopped")

def main():
    """Main entry point"""
    app = EAIBApp()
    app.run()

if __name__ == "__main__":
    main() 