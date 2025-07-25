#!/usr/bin/env python3
"""
Streamlit interface for EAIB Web Scraping Pipeline
End-to-end pipeline with configurable parameters and configuration editor
"""

import streamlit as st
import requests
import json
import time
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime
import copy

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import load_data_config
from src.utils.logging_config import logger

# Load configuration
config = load_data_config()

# API base URL
API_BASE_URL = "http://localhost:8000"

def check_api_connection():
    """Check if the API server is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_config_value(config_dict: Dict[str, Any], config_path: str, default=None):
    """Get value from config using dot notation"""
    try:
        config_value = config_dict
        for key in config_path.split('.'):
            config_value = config_value.get(key, {})
        return config_value if config_value != {} else default
    except:
        return default

def set_config_value(config_dict: Dict[str, Any], config_path: str, value: Any):
    """Set value in config using dot notation"""
    keys = config_path.split('.')
    current = config_dict
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value

def render_config_editor():
    """Render the configuration editor with separate sections"""
    st.header("⚙️ Configuration Editor")
    st.markdown("Edit configuration values that will be used as defaults throughout the pipeline")
    
    # Initialize session state for config if not exists
    if "edited_config" not in st.session_state:
        st.session_state.edited_config = copy.deepcopy(config)
    
    # Create tabs for different config sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Search Config", 
        "🤖 API Config", 
        "🧠 Processing Config", 
        "❓ Q&A Generation", 
        "💾 Database Config"
    ])
    
    # Tab 1: Search Configuration
    with tab1:
        st.subheader("🔍 Search Configuration")
        
        with st.container():
            st.markdown("#### Default Search Parameters")
            col1, col2 = st.columns(2)
            
            with col1:
                num_results = st.number_input(
                    "Default Number of Results",
                    min_value=1,
                    max_value=50,
                    value=get_config_value(st.session_state.edited_config, "search.default_params.num_results", 10),
                    help="Default number of search results to fetch"
                )
                set_config_value(st.session_state.edited_config, "search.default_params.num_results", num_results)
                
                extract_content = st.checkbox(
                    "Extract Content by Default",
                    value=get_config_value(st.session_state.edited_config, "search.default_params.extract_content", True),
                    help="Whether to extract full content from URLs by default"
                )
                set_config_value(st.session_state.edited_config, "search.default_params.extract_content", extract_content)
                
                use_default_sites = st.checkbox(
                    "Use Default Sites",
                    value=get_config_value(st.session_state.edited_config, "search.default_params.use_default_sites", True),
                    help="Use pre-configured quality sites by default"
                )
                set_config_value(st.session_state.edited_config, "search.default_params.use_default_sites", use_default_sites)
            
            with col2:
                search_strategy = st.selectbox(
                    "Default Search Strategy",
                    options=["mixed", "targeted", "broad"],
                    index=["mixed", "targeted", "broad"].index(
                        get_config_value(st.session_state.edited_config, "search.default_params.search_strategy", "mixed")
                    ),
                    help="Default search strategy"
                )
                set_config_value(st.session_state.edited_config, "search.default_params.search_strategy", search_strategy)
                
                country = st.text_input(
                    "Default Country",
                    value=get_config_value(st.session_state.edited_config, "search.default_params.country", "us"),
                    help="Default country for search results"
                )
                set_config_value(st.session_state.edited_config, "search.default_params.country", country)
                
                language = st.text_input(
                    "Default Language",
                    value=get_config_value(st.session_state.edited_config, "search.default_params.language", "en"),
                    help="Default language for search results"
                )
                set_config_value(st.session_state.edited_config, "search.default_params.language", language)
        
        with st.container():
            st.markdown("#### Candidate Sites")
            
            # Get current candidate sites
            current_sites = get_config_value(st.session_state.edited_config, "search.candidate_sites.sites", [])
            
            # Display existing sites
            if current_sites:
                sites_df = pd.DataFrame(current_sites)
                st.dataframe(sites_df, use_container_width=True)
            
            # Add new site
            with st.expander("➕ Add New Candidate Site"):
                new_site_name = st.text_input("Site Name", placeholder="e.g., Wikipedia")
                new_site_domain = st.text_input("Domain", placeholder="e.g., wikipedia.org")
                new_site_priority = st.number_input("Priority", min_value=1, max_value=5, value=3)
                new_site_desc = st.text_input("Description", placeholder="e.g., Encyclopedia articles")
                
                if st.button("Add Site"):
                    if new_site_name and new_site_domain:
                        new_site = {
                            "name": new_site_name,
                            "domain": new_site_domain,
                            "search_prefix": f"site:{new_site_domain}",
                            "priority": new_site_priority,
                            "description": new_site_desc
                        }
                        current_sites.append(new_site)
                        set_config_value(st.session_state.edited_config, "search.candidate_sites.sites", current_sites)
                        st.success(f"Added {new_site_name}")
                        st.rerun()
    
    # Tab 2: API Configuration
    with tab2:
        st.subheader("🤖 API Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Groq API Settings")
            groq_model = st.text_input(
                "Groq Model",
                value=get_config_value(st.session_state.edited_config, "apis.groq.model", "llama3-8b-8192"),
                help="Groq LLM model to use"
            )
            set_config_value(st.session_state.edited_config, "apis.groq.model", groq_model)
            
            groq_temp = st.slider(
                "Groq Temperature",
                min_value=0.0,
                max_value=2.0,
                value=float(get_config_value(st.session_state.edited_config, "apis.groq.temperature", 0.1)),
                step=0.1,
                help="Temperature for Groq API calls"
            )
            set_config_value(st.session_state.edited_config, "apis.groq.temperature", groq_temp)
            
            groq_tokens = st.number_input(
                "Groq Max Tokens",
                min_value=100,
                max_value=4000,
                value=get_config_value(st.session_state.edited_config, "apis.groq.max_tokens", 1000),
                help="Maximum tokens for Groq responses"
            )
            set_config_value(st.session_state.edited_config, "apis.groq.max_tokens", groq_tokens)
        
        with col2:
            st.markdown("#### SerpAPI Settings")
            serp_base_url = st.text_input(
                "SerpAPI Base URL",
                value=get_config_value(st.session_state.edited_config, "apis.serper.base_url", "https://google.serper.dev"),
                help="Base URL for SerpAPI"
            )
            set_config_value(st.session_state.edited_config, "apis.serper.base_url", serp_base_url)
            
            st.markdown("#### Content Extraction")
            crawler_timeout = st.number_input(
                "Crawler Timeout (ms)",
                min_value=5000,
                max_value=60000,
                value=get_config_value(st.session_state.edited_config, "processing.content_extraction.crawler.timeout", 30000),
                step=1000,
                help="Timeout for content extraction"
            )
            set_config_value(st.session_state.edited_config, "processing.content_extraction.crawler.timeout", crawler_timeout)
            
            rate_limit_delay = st.number_input(
                "Rate Limit Delay (seconds)",
                min_value=1,
                max_value=10,
                value=get_config_value(st.session_state.edited_config, "processing.content_extraction.crawler.rate_limit_delay", 2),
                help="Delay between content extraction requests"
            )
            set_config_value(st.session_state.edited_config, "processing.content_extraction.crawler.rate_limit_delay", rate_limit_delay)
            
            st.markdown("#### Request Timeouts")
            scraping_timeout = st.number_input(
                "Scraping Timeout (minutes)",
                min_value=1,
                max_value=10,
                value=get_config_value(st.session_state.edited_config, "apis.timeouts.scraping_minutes", 5),
                help="Maximum time to wait for scraping operations"
            )
            set_config_value(st.session_state.edited_config, "apis.timeouts.scraping_minutes", scraping_timeout)
    
    # Tab 3: Processing Configuration
    with tab3:
        st.subheader("🧠 Text Processing Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Segment Settings")
            min_segment_length = st.number_input(
                "Min Segment Length",
                min_value=10,
                max_value=500,
                value=get_config_value(st.session_state.edited_config, "processing.defaults.min_segment_length", 50),
                help="Minimum characters per text segment"
            )
            set_config_value(st.session_state.edited_config, "processing.defaults.min_segment_length", min_segment_length)
            
            max_segment_length = st.number_input(
                "Max Segment Length",
                min_value=500,
                max_value=5000,
                value=get_config_value(st.session_state.edited_config, "processing.defaults.max_segment_length", 1000),
                help="Maximum characters per text segment"
            )
            set_config_value(st.session_state.edited_config, "processing.defaults.max_segment_length", max_segment_length)
            
            quality_threshold = st.number_input(
                "Quality Threshold",
                min_value=1,
                max_value=5,
                value=get_config_value(st.session_state.edited_config, "processing.defaults.quality_threshold", 3),
                help="Minimum quality score (out of 5)"
            )
            set_config_value(st.session_state.edited_config, "processing.defaults.quality_threshold", quality_threshold)
        
        with col2:
            st.markdown("#### Deduplication Settings")
            similarity_threshold = st.slider(
                "Similarity Threshold",
                min_value=0.1,
                max_value=1.0,
                value=float(get_config_value(st.session_state.edited_config, "processing.defaults.similarity_threshold", 0.85)),
                step=0.05,
                help="Threshold for detecting duplicate content"
            )
            set_config_value(st.session_state.edited_config, "processing.defaults.similarity_threshold", similarity_threshold)
            
            st.markdown("#### Content Extraction")
            min_content_length = st.number_input(
                "Min Content Length",
                min_value=50,
                max_value=1000,
                value=get_config_value(st.session_state.edited_config, "processing.content_extraction.min_content_length", 100),
                help="Minimum content length to extract"
            )
            set_config_value(st.session_state.edited_config, "processing.content_extraction.min_content_length", min_content_length)
            
            max_content_length = st.number_input(
                "Max Content Length",
                min_value=1000,
                max_value=50000,
                value=get_config_value(st.session_state.edited_config, "processing.content_extraction.max_content_length", 10000),
                help="Maximum content length to extract"
            )
            set_config_value(st.session_state.edited_config, "processing.content_extraction.max_content_length", max_content_length)
    
    # Tab 4: Q&A Generation Configuration
    with tab4:
        st.subheader("❓ Q&A Generation Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Generation Parameters")
            max_qa_per_segment = st.number_input(
                "Max Q&A per Segment",
                min_value=1,
                max_value=10,
                value=get_config_value(st.session_state.edited_config, "qa_generation.defaults.max_qa_per_segment", 3),
                help="Maximum Q&A pairs per text segment"
            )
            set_config_value(st.session_state.edited_config, "qa_generation.defaults.max_qa_per_segment", max_qa_per_segment)
            
            max_questions_default = get_config_value(st.session_state.edited_config, "qa_generation.defaults.max_questions", None)
            max_questions = st.number_input(
                "Max Total Questions (0 = unlimited)",
                min_value=0,
                value=max_questions_default or 0,
                help="Maximum total questions to generate"
            )
            set_config_value(st.session_state.edited_config, "qa_generation.defaults.max_questions", max_questions if max_questions > 0 else None)
            
            qa_model = st.text_input(
                "Q&A Model",
                value=get_config_value(st.session_state.edited_config, "qa_generation.defaults.model", "llama-3.1-8b-instant"),
                help="Model to use for Q&A generation"
            )
            set_config_value(st.session_state.edited_config, "qa_generation.defaults.model", qa_model)
        
        with col2:
            st.markdown("#### Generation Settings")
            qa_temperature = st.slider(
                "Q&A Temperature",
                min_value=0.0,
                max_value=2.0,
                value=float(get_config_value(st.session_state.edited_config, "qa_generation.defaults.temperature", 0.3)),
                step=0.1,
                help="Temperature for Q&A generation"
            )
            set_config_value(st.session_state.edited_config, "qa_generation.defaults.temperature", qa_temperature)
            
            qa_max_tokens = st.number_input(
                "Q&A Max Tokens",
                min_value=100,
                max_value=4000,
                value=get_config_value(st.session_state.edited_config, "qa_generation.defaults.max_tokens", 1000),
                help="Maximum tokens for Q&A generation"
            )
            set_config_value(st.session_state.edited_config, "qa_generation.defaults.max_tokens", qa_max_tokens)
            
            context_window = st.number_input(
                "Context Window",
                min_value=100,
                max_value=2000,
                value=get_config_value(st.session_state.edited_config, "qa_generation.prompt_settings.context_window", 500),
                help="Characters of context around each segment"
            )
            set_config_value(st.session_state.edited_config, "qa_generation.prompt_settings.context_window", context_window)
    
    # Tab 5: Database Configuration
    with tab5:
        st.subheader("💾 Database Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### MongoDB Settings")
            db_name = st.text_input(
                "Database Name",
                value=get_config_value(st.session_state.edited_config, "database.mongodb.database_name", "scraping_pipeline"),
                help="MongoDB database name"
            )
            set_config_value(st.session_state.edited_config, "database.mongodb.database_name", db_name)
            
            st.markdown("#### Collection Names")
            unified_docs_collection = st.text_input(
                "Unified Documents Collection",
                value=get_config_value(st.session_state.edited_config, "database.mongodb.collections.unified_documents", "unified_documents"),
                help="Collection for unified documents"
            )
            set_config_value(st.session_state.edited_config, "database.mongodb.collections.unified_documents", unified_docs_collection)
            
            cleaned_data_collection = st.text_input(
                "Cleaned Data Collection",
                value=get_config_value(st.session_state.edited_config, "database.mongodb.collections.cleaned_data", "cleaned_data"),
                help="Collection for processed text data"
            )
            set_config_value(st.session_state.edited_config, "database.mongodb.collections.cleaned_data", cleaned_data_collection)
        
        with col2:
            qa_datasets_collection = st.text_input(
                "Q&A Datasets Collection",
                value=get_config_value(st.session_state.edited_config, "database.mongodb.collections.qa_datasets", "qa_datasets"),
                help="Collection for Q&A datasets"
            )
            set_config_value(st.session_state.edited_config, "database.mongodb.collections.qa_datasets", qa_datasets_collection)
            
            search_sessions_collection = st.text_input(
                "Search Sessions Collection",
                value=get_config_value(st.session_state.edited_config, "database.mongodb.collections.search_sessions", "search_sessions"),
                help="Collection for search sessions"
            )
            set_config_value(st.session_state.edited_config, "database.mongodb.collections.search_sessions", search_sessions_collection)
    
    # Configuration actions
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Configuration", type="primary", use_container_width=True):
            try:
                config_file_path = project_root / "config" / "data_config.yaml"
                with open(config_file_path, 'w') as f:
                    yaml.dump(st.session_state.edited_config, f, default_flow_style=False, indent=2)
                st.success("✅ Configuration saved successfully!")
            except Exception as e:
                st.error(f"❌ Error saving configuration: {e}")
    
    with col2:
        if st.button("🔄 Reset to Original", use_container_width=True):
            st.session_state.edited_config = copy.deepcopy(config)
            st.success("Configuration reset to original values")
            st.rerun()
    
    with col3:
        if st.button("📄 Export Config", use_container_width=True):
            config_yaml = yaml.dump(st.session_state.edited_config, default_flow_style=False, indent=2)
            st.download_button(
                label="Download YAML",
                data=config_yaml,
                file_name="data_config.yaml",
                mime="text/yaml"
            )

def render_pipeline_controls():
    """Render pipeline execution controls"""
    st.header("🚀 Pipeline Execution")
    
    # Pipeline execution options
    execution_mode = st.radio(
        "Select Execution Mode",
        options=["Complete Pipeline", "Individual Steps"],
        horizontal=True
    )
    
    if execution_mode == "Complete Pipeline":
        st.markdown("### 🔄 Complete End-to-End Pipeline")
        
        col1, col2 = st.columns(2)
        with col1:
            query = st.text_input("Search Query", placeholder="e.g., artificial intelligence")
            document_id = st.text_input("Document ID (optional)", placeholder="auto-generated if empty")
        
        with col2:
            include_pdf = st.checkbox("Include PDF Upload", value=False)
            if include_pdf:
                uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
        
        if st.button("🚀 Run Complete Pipeline", type="primary", use_container_width=True):
            if query:
                with st.spinner("Running complete pipeline..."):
                    run_complete_pipeline(query, document_id, uploaded_file if include_pdf else None)
            else:
                st.error("Please enter a search query")
    
    else:
        st.markdown("### 🎯 Individual Pipeline Steps")
        
        step_tabs = st.tabs(["🔍 Scraping", "📄 PDF Upload", "🧠 Processing", "❓ Q&A Generation"])
        
        # Individual step controls
        with step_tabs[0]:
            render_scraping_controls()
        
        with step_tabs[1]:
            render_pdf_upload_controls()
        
        with step_tabs[2]:
            render_processing_controls()
        
        with step_tabs[3]:
            render_qa_generation_controls()

def render_scraping_controls():
    """Render web scraping controls"""
    st.subheader("🔍 Web Scraping")
    
    # Use current config values as defaults
    current_config = st.session_state.get("edited_config", config)
    
    col1, col2 = st.columns(2)
    
    with col1:
        query = st.text_input("Search Query", placeholder="e.g., artificial intelligence")
        num_results = st.number_input(
            "Number of Results", 
            min_value=1, 
            max_value=50, 
            value=get_config_value(current_config, "search.default_params.num_results", 10)
        )
        extract_content = st.checkbox(
            "Extract Full Content", 
            value=get_config_value(current_config, "search.default_params.extract_content", True)
        )
    
    with col2:
        search_strategy = st.selectbox(
            "Search Strategy",
            options=["mixed", "targeted", "broad"],
            index=["mixed", "targeted", "broad"].index(
                get_config_value(current_config, "search.default_params.search_strategy", "mixed")
            )
        )
        use_default_sites = st.checkbox(
            "Use Default Sites", 
            value=get_config_value(current_config, "search.default_params.use_default_sites", True)
        )
        document_id = st.text_input("Document ID (optional)", placeholder="auto-generated if empty")
    
            # Helpful tips for users
        with st.expander("💡 Scraping Tips", expanded=False):
            st.markdown("""
            **To avoid timeouts:**
            - Start with fewer results (3-5) for testing
            - Disable content extraction for faster results
            - Use targeted search strategy for specific sites
            - Increase timeout in Configuration Editor if needed
            
            **Processing times:**
            - Search only: ~10-30 seconds
            - With content extraction: 2-10 minutes
            - Large requests (20+ results): 5-15 minutes
            """)
        
        if st.button("🚀 Start Scraping", type="primary", use_container_width=True):
            if query:
                run_scraping_step(query, num_results, extract_content, search_strategy, use_default_sites, document_id)
            else:
                st.error("Please enter a search query")

def render_pdf_upload_controls():
    """Render PDF upload controls"""
    st.subheader("📄 PDF Upload")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
    document_id = st.text_input(
        "Document ID", 
        value=st.session_state.get("current_document_id", ""),
        help="Use existing document ID to combine with scraped content"
    )
    
    if uploaded_file and st.button("📤 Upload PDF", type="primary", use_container_width=True):
        run_pdf_upload_step(uploaded_file, document_id)

def render_processing_controls():
    """Render text processing controls"""
    st.subheader("🧠 Text Processing")
    
    document_id = st.text_input(
        "Document ID to Process", 
        value=st.session_state.get("current_document_id", ""),
        help="Enter document ID to process"
    )
    
    if document_id and st.button("🧠 Process Document", type="primary", use_container_width=True):
        run_processing_step(document_id)
    elif not document_id:
        st.warning("⚠️ Please enter a document ID to process")

def render_qa_generation_controls():
    """Render Q&A generation controls"""
    st.subheader("❓ Q&A Generation")
    
    current_config = st.session_state.get("edited_config", config)
    
    col1, col2 = st.columns(2)
    
    with col1:
        document_id = st.text_input(
            "Document ID", 
            value=st.session_state.get("current_document_id", ""),
            help="Enter document ID for Q&A generation"
        )
        max_qa_per_segment = st.number_input(
            "Max Q&A per Segment",
            min_value=1,
            max_value=10,
            value=get_config_value(current_config, "qa_generation.defaults.max_qa_per_segment", 3)
        )
    
    with col2:
        max_questions_default = get_config_value(current_config, "qa_generation.defaults.max_questions", None)
        max_questions = st.number_input(
            "Max Total Questions (0 = unlimited)",
            min_value=0,
            value=max_questions_default or 0
        )
    
    if document_id and st.button("❓ Generate Q&A", type="primary", use_container_width=True):
        run_qa_generation_step(document_id, max_qa_per_segment, max_questions if max_questions > 0 else None)
    elif not document_id:
        st.warning("⚠️ Please enter a document ID for Q&A generation")

# Pipeline execution functions
def run_complete_pipeline(query: str, document_id: Optional[str], uploaded_file=None):
    """Run the complete end-to-end pipeline"""
    current_config = st.session_state.get("edited_config", config)
    
    # Show pipeline duration warning
    st.warning("⏱️ Complete pipeline may take 5-15 minutes depending on content size and complexity.")
    
    try:
        # Step 1: Web Scraping
        st.info("🔍 Step 1: Web Scraping... (this may take several minutes)")
        scrape_result = scrape_content({
            "query": query,
            "num_results": get_config_value(current_config, "search.default_params.num_results", 10),
            "extract_content": get_config_value(current_config, "search.default_params.extract_content", True),
            "use_default_sites": get_config_value(current_config, "search.default_params.use_default_sites", True),
            "search_strategy": get_config_value(current_config, "search.default_params.search_strategy", "mixed"),
            "document_id": document_id
        })
        
        if scrape_result["status"] != "success":
            st.error(f"❌ Scraping failed: {scrape_result['message']}")
            return
        
        doc_id = scrape_result["data"]["document_id"]
        st.session_state.current_document_id = doc_id
        st.success(f"✅ Scraping completed! Document ID: {doc_id}")
        
        # Step 2: PDF Upload (if provided)
        if uploaded_file:
            st.info("📄 Step 2: PDF Upload...")
            pdf_result = upload_pdf(uploaded_file, doc_id)
            if pdf_result["status"] == "success":
                st.success("✅ PDF uploaded successfully!")
            else:
                st.warning(f"⚠️ PDF upload failed: {pdf_result['message']}")
        
        # Step 3: Text Processing
        st.info("🧠 Step 3: Text Processing...")
        process_result = process_document(doc_id)
        if process_result["status"] != "success":
            st.error(f"❌ Processing failed: {process_result['message']}")
            return
        st.success("✅ Text processing completed!")
        
        # Step 4: Q&A Generation
        st.info("❓ Step 4: Q&A Generation...")
        qa_result = generate_qa(
            doc_id,
            get_config_value(current_config, "qa_generation.defaults.max_qa_per_segment", 3),
            get_config_value(current_config, "qa_generation.defaults.max_questions", None)
        )
        if qa_result["status"] != "success":
            st.error(f"❌ Q&A generation failed: {qa_result['message']}")
            return
        st.success("✅ Q&A generation completed!")
        
        # Success
        st.success("🎉 Complete pipeline executed successfully!")
        
        # Display final results
        with st.expander("📊 Final Results", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Search Results", scrape_result["data"]["total_results"])
            with col2:
                st.metric("Text Segments", process_result["data"]["final_segments"])
            with col3:
                st.metric("Q&A Pairs", qa_result["data"]["total_qa_pairs"])
            
            # Display generated Q&A pairs
            if qa_result["data"].get("qa_pairs"):
                st.subheader("❓ Generated Questions & Answers")
                qa_pairs = qa_result["data"]["qa_pairs"]
                
                # Summary of generation
                st.info(f"🎯 Generated {len(qa_pairs)} Q&A pairs from {qa_result['data']['segments_processed']} text segments")
                
                # Show first 10 Q&A pairs with option to see more
                display_count = min(10, len(qa_pairs))
                
                # Create a more organized display
                for i, qa in enumerate(qa_pairs[:display_count]):
                    st.markdown(f"### Q{i+1}")
                    st.markdown(f"**Question:** {qa.get('question', 'No question')}")
                    st.markdown(f"**Answer:** {qa.get('answer', 'No answer')}")
                    if qa.get('segment_index') is not None:
                        st.caption(f"📄 From segment {qa.get('segment_index') + 1}")
                    st.divider()
                
                if len(qa_pairs) > display_count:
                    st.info(f"📋 Showing {display_count} of {len(qa_pairs)} Q&A pairs. Check the generated files for the complete dataset.")
                
                # Download option for all Q&A pairs
                if st.button("📥 Download Q&A Dataset", type="secondary"):
                    qa_text = ""
                    for i, qa in enumerate(qa_pairs):
                        qa_text += f"Q{i+1}: {qa.get('question', 'No question')}\n"
                        qa_text += f"A{i+1}: {qa.get('answer', 'No answer')}\n\n"
                    
                    st.download_button(
                        label="💾 Download as Text",
                        data=qa_text,
                        file_name=f"qa_dataset_{doc_id}.txt",
                        mime="text/plain"
                    )
    
    except Exception as e:
        st.error(f"❌ Pipeline execution failed: {str(e)}")

def run_scraping_step(query: str, num_results: int, extract_content: bool, search_strategy: str, use_default_sites: bool, document_id: Optional[str]):
    """Run web scraping step"""
    # Show timeout warning for large requests
    if num_results > 10 or extract_content:
        st.info("⏱️ Large scraping requests may take several minutes. Please be patient...")
    
    with st.spinner("Scraping web content... This may take a few minutes for content extraction."):
        result = scrape_content({
            "query": query,
            "num_results": num_results,
            "extract_content": extract_content,
            "search_strategy": search_strategy,
            "use_default_sites": use_default_sites,
            "document_id": document_id
        })
        
        if result["status"] == "success":
            data = result["data"]
            st.session_state.current_document_id = data["document_id"]
            st.success(f"✅ Scraping completed! Document ID: {data['document_id']}")
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Results", data["total_results"])
            with col2:
                st.metric("Processing Time", f"{data['processing_time']:.2f}s")
            with col3:
                extraction_stats = data["content_extraction_stats"]
                st.metric("Success Rate", f"{extraction_stats.get('success_rate', 0):.1%}")
        else:
            st.error(f"❌ Scraping failed: {result['message']}")

def run_pdf_upload_step(file, document_id: Optional[str]):
    """Run PDF upload step"""
    with st.spinner("Uploading and processing PDF..."):
        result = upload_pdf(file, document_id)
        
        if result["status"] == "success":
            data = result["data"]
            st.session_state.current_document_id = data["document_id"]
            st.success(f"✅ PDF uploaded! Document ID: {data['document_id']}")
            
            # Display PDF info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Pages", data["page_count"])
            with col2:
                st.metric("Content Length", f"{data['content_length']:,} chars")
            with col3:
                st.metric("Processing Time", f"{data['processing_time']:.2f}s")
        else:
            st.error(f"❌ PDF upload failed: {result['message']}")

def run_processing_step(document_id: str):
    """Run text processing step"""
    with st.spinner("Processing document text..."):
        result = process_document(document_id)
        
        if result["status"] == "success":
            data = result["data"]
            st.success("✅ Document processed successfully!")
            
            # Display processing stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Original Length", f"{data['original_length']:,} chars")
            with col2:
                st.metric("Final Segments", data["final_segments"])
            with col3:
                st.metric("Deduplication", f"{data['deduplication_ratio']:.1%}")
            with col4:
                st.metric("Quality Retention", f"{data['quality_retention']:.1%}")
        else:
            st.error(f"❌ Processing failed: {result['message']}")

def run_qa_generation_step(document_id: str, max_qa_per_segment: int, max_questions: Optional[int]):
    """Run Q&A generation step"""
    with st.spinner("Generating Q&A dataset..."):
        result = generate_qa(document_id, max_qa_per_segment, max_questions)
        
        if result["status"] == "success":
            data = result["data"]
            st.success("✅ Q&A dataset generated successfully!")
            
            # Display generation stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Q&A Pairs", data["total_qa_pairs"])
            with col2:
                st.metric("Segments Processed", data["segments_processed"])
            with col3:
                st.metric("Processing Time", f"{data['processing_time']:.2f}s")
            
            # Show generated files
            st.subheader("📁 Generated Files")
            for file_path in data["files_created"]:
                st.write(f"• {file_path}")
            
            # Display generated Q&A pairs
            if data.get("qa_pairs"):
                st.subheader("❓ Generated Questions & Answers")
                qa_pairs = data["qa_pairs"]
                
                # Summary of generation
                st.info(f"🎯 Generated {len(qa_pairs)} Q&A pairs from {data['segments_processed']} text segments")
                
                # Show first 10 Q&A pairs with option to see more
                display_count = min(10, len(qa_pairs))
                
                # Create a more organized display
                for i, qa in enumerate(qa_pairs[:display_count]):
                    st.markdown(f"### Q{i+1}")
                    st.markdown(f"**Question:** {qa.get('question', 'No question')}")
                    st.markdown(f"**Answer:** {qa.get('answer', 'No answer')}")
                    if qa.get('segment_index') is not None:
                        st.caption(f"📄 From segment {qa.get('segment_index') + 1}")
                    st.divider()
                
                if len(qa_pairs) > display_count:
                    st.info(f"📋 Showing {display_count} of {len(qa_pairs)} Q&A pairs. Check the generated files for the complete dataset.")
                
                # Download option for all Q&A pairs
                if st.button("📥 Download Q&A Dataset", type="secondary"):
                    qa_text = ""
                    for i, qa in enumerate(qa_pairs):
                        qa_text += f"Q{i+1}: {qa.get('question', 'No question')}\n"
                        qa_text += f"A{i+1}: {qa.get('answer', 'No answer')}\n\n"
                    
                    st.download_button(
                        label="💾 Download as Text",
                        data=qa_text,
                        file_name=f"qa_dataset_{document_id}.txt",
                        mime="text/plain"
                    )
        else:
            st.error(f"❌ Q&A generation failed: {result['message']}")

# API call functions (existing functions)
def scrape_content(params: Dict[str, Any]) -> Dict[str, Any]:
    """Call the scrape API endpoint"""
    try:
        # Get timeout from config or use default
        current_config = st.session_state.get("edited_config", config)
        timeout_minutes = get_config_value(current_config, "apis.timeouts.scraping_minutes", 5)
        timeout = timeout_minutes * 60  # Convert to seconds
        
        response = requests.post(f"{API_BASE_URL}/scrape", json=params, timeout=timeout)
        if response.status_code == 200:
            return {"status": "success", "data": response.json()}
        else:
            return {"status": "error", "message": response.text}
    except requests.exceptions.Timeout:
        return {"status": "error", "message": "Request timed out. Web scraping is taking longer than expected. Try reducing the number of results or check your internet connection."}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Cannot connect to API server. Please ensure the API server is running."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def upload_pdf(file, document_id: Optional[str] = None) -> Dict[str, Any]:
    """Upload PDF file"""
    try:
        files = {"file": file}
        params = {"document_id": document_id} if document_id else {}
        response = requests.post(f"{API_BASE_URL}/upload-pdf", files=files, params=params, timeout=120)
        if response.status_code == 200:
            return {"status": "success", "data": response.json()}
        else:
            return {"status": "error", "message": response.text}
    except requests.exceptions.Timeout:
        return {"status": "error", "message": "PDF upload timed out. The file might be too large or the connection is slow."}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Cannot connect to API server. Please ensure the API server is running."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def process_document(document_id: str) -> Dict[str, Any]:
    """Process document text"""
    try:
        response = requests.post(f"{API_BASE_URL}/process-document/{document_id}", timeout=180)
        if response.status_code == 200:
            return {"status": "success", "data": response.json()}
        else:
            return {"status": "error", "message": response.text}
    except requests.exceptions.Timeout:
        return {"status": "error", "message": "Text processing timed out. The document might be very large or complex."}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Cannot connect to API server. Please ensure the API server is running."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def generate_qa(document_id: str, max_qa_per_segment: int, max_questions: Optional[int]) -> Dict[str, Any]:
    """Generate Q&A dataset"""
    try:
        params = {"max_qa_per_segment": max_qa_per_segment}
        if max_questions:
            params["max_questions"] = max_questions
        
        response = requests.post(f"{API_BASE_URL}/generate-qa/{document_id}", params=params, timeout=600)
        if response.status_code == 200:
            return {"status": "success", "data": response.json()}
        else:
            return {"status": "error", "message": response.text}
    except requests.exceptions.Timeout:
        return {"status": "error", "message": "Q&A generation timed out. This process can take several minutes for large documents. Try reducing max_questions or max_qa_per_segment."}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Cannot connect to API server. Please ensure the API server is running."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def main():
    try:
        st.set_page_config(
            page_title="EAIB Pipeline Interface",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🚀 EAIB Web Scraping Pipeline")
        st.markdown("End-to-end pipeline for web scraping, content processing, and Q&A generation")
        
        # Check API connection
        if not check_api_connection():
            st.error("❌ API server is not running. Please start the API server first:")
            st.code("python app.py")
            st.stop()
        
        st.success("✅ API server is running")
        
        # Main content area
        tab1, tab2 = st.tabs(["🚀 Pipeline Execution", "⚙️ Configuration Editor"])
        
        with tab1:
            render_pipeline_controls()
        
        with tab2:
            render_config_editor()
        
        # Sidebar status
        with st.sidebar:
            st.header("📊 Pipeline Status")
            if "current_document_id" in st.session_state:
                st.info(f"📋 Current Document: {st.session_state.current_document_id}")
            else:
                st.info("📋 No active document")
            
            # Quick actions
            st.header("⚡ Quick Actions")
            if st.button("🔄 Refresh Config", use_container_width=True):
                global config
                config = load_data_config()
                if "edited_config" in st.session_state:
                    st.session_state.edited_config = copy.deepcopy(config)
                st.success("Configuration refreshed!")
                st.rerun()
                
    except Exception as e:
        st.error(f"❌ Error loading Streamlit interface: {e}")
        st.code(f"Error details: {str(e)}")
        st.stop()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Fallback: show a simple error page
        st.set_page_config(page_title="EAIB Error", page_icon="❌")
        st.error("❌ EAIB Pipeline Interface Error")
        st.write(f"Error details: {str(e)}")
        st.write("Please check the console for more details.") 