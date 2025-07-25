import os
import yaml
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")

def get_env_variable(var_name: str, default: Optional[str] = None) -> str:
    """Get environment variable with optional default"""
    value = os.getenv(var_name, default)
    if value is None:
        raise ValueError(f"Environment variable {var_name} is required but not set")
    return value

def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).parent.parent.parent

def load_data_config() -> Dict[str, Any]:
    """Load data configuration with environment variable substitution"""
    config_path = get_project_root() / "config" / "data_config.yaml"
    config = load_config(str(config_path))
    
    # Substitute environment variables for API keys
    if 'apis' in config:
        for api_name, api_config in config['apis'].items():
            if 'api_key_env' in api_config:
                env_var = api_config['api_key_env']
                api_config['api_key'] = get_env_variable(env_var)
    
    # Substitute database connection string
    if 'database' in config and 'mongodb' in config['database']:
        mongo_config = config['database']['mongodb']
        if 'connection_string_env' in mongo_config:
            env_var = mongo_config['connection_string_env']
            mongo_config['connection_string'] = get_env_variable(env_var)
    
    return config

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove special characters but keep basic punctuation
    import re
    text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)]', '', text)
    
    return text.strip()

def generate_search_id(topic: str) -> str:
    """Generate unique search session ID"""
    import hashlib
    import datetime
    
    timestamp = datetime.datetime.now().isoformat()
    content = f"{topic}_{timestamp}"
    return hashlib.md5(content.encode()).hexdigest()[:12] 