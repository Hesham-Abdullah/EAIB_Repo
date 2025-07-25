#!/usr/bin/env python3
"""
Environment setup script for the EAIB scraping pipeline
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed successfully")
        
        # Install playwright browsers for crawl4ai
        print("\n🌐 Installing Playwright browsers for robust content extraction...")
        try:
            subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], 
                          check=True, capture_output=True)
            print("✅ Playwright browsers installed")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Playwright installation failed: {e}")
            print("💡 You can manually install with: python -m playwright install chromium")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        sys.exit(1)

def setup_environment_file():
    """Set up environment file with API keys"""
    env_file = Path("env.example")
    target_env = Path(".env")
    
    if not env_file.exists():
        print("❌ env.example file not found")
        return False
    
    if target_env.exists():
        print("⚠️  .env file already exists")
        return True
    
    # Copy template
    with open(env_file, 'r') as f:
        content = f.read()
    
    print("\n🔑 Setting up environment variables...")
    print("Please provide your API keys:")
    
    # Get API keys from user
    groq_key = input("Groq API Key (or press Enter to use provided): ").strip()
    if not groq_key:
        groq_key = "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"
    
    serper_key = input("Serper API Key (or press Enter to use provided): ").strip()
    if not serper_key:
        serper_key = "7117457cd2193a7e417a3f4082906afc98de09dd"
    
    mongo_connection = input("MongoDB Connection String (or press Enter for default): ").strip()
    if not mongo_connection:
        mongo_connection = "mongodb://localhost:27017/"
    
    # Replace placeholders
    content = content.replace("your_groq_api_key_here", groq_key)
    content = content.replace("your_serper_api_key_here", serper_key)
    content = content.replace("mongodb://localhost:27017/", mongo_connection)
    
    # Write .env file
    with open(target_env, 'w') as f:
        f.write(content)
    
    print("✅ Environment file created: .env")
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "logs",
        "data",
        "data/raw",
        "data/processed"
    ]
    
    print("\n📁 Creating directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")

def check_mongodb():
    """Check if MongoDB is accessible"""
    print("\n🗄️  Checking MongoDB connection...")
    try:
        # Try to import and connect
        from pymongo import MongoClient
        from dotenv import load_dotenv
        
        load_dotenv()
        connection_string = os.getenv("MONGODB_CONNECTION_STRING", "mongodb://localhost:27017/")
        
        client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        client.close()
        
        print("✅ MongoDB connection successful")
        return True
    except Exception as e:
        print(f"⚠️  MongoDB connection failed: {e}")
        print("💡 Make sure MongoDB is running or update the connection string in .env")
        return False

def test_apis():
    """Test API connections"""
    print("\n🔗 Testing API connections...")
    
    # Test environment loading
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        groq_key = os.getenv("GROQ_API_KEY")
        serper_key = os.getenv("SERPER_API_KEY")
        
        if groq_key and len(groq_key) > 10:
            print("✅ Groq API key loaded")
        else:
            print("⚠️  Groq API key not found or invalid")
        
        if serper_key and len(serper_key) > 10:
            print("✅ Serper API key loaded")
        else:
            print("⚠️  Serper API key not found or invalid")
            
    except Exception as e:
        print(f"❌ Error loading environment: {e}")

def main():
    """Main setup function"""
    print("🚀 EAIB Scraping Pipeline Setup")
    print("=" * 40)
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    install_dependencies()
    
    # Setup environment
    setup_environment_file()
    
    # Create directories
    create_directories()
    
    # Test MongoDB
    mongodb_ok = check_mongodb()
    
    # Test APIs
    test_apis()
    
    print("\n" + "=" * 40)
    print("🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Ensure MongoDB is running")
    if not mongodb_ok:
        print("2. Update MongoDB connection in .env file")
    print("3. Run the pipeline: python scripts/run_pipeline.py \"your search topic\"")
    print("\n💡 Example usage:")
    print('   python scripts/run_pipeline.py "electric vehicle charging stations"')
    print('   python scripts/run_pipeline.py "renewable energy" --num-results 15')
    print('   python scripts/run_pipeline.py "AI technology" --no-content  # Faster, no full content')
    print('   python scripts/run_pipeline.py "solar energy" --num-results 5  # With full content')

if __name__ == "__main__":
    main() 