#!/usr/bin/env python3
"""
FastAPI application for the web scraping pipeline
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import sys
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from datetime import datetime
import uvicorn
import PyPDF2
import io

from src.data_collection.web_scraper import WebScraper
from src.data_processing.storage import MongoStorage
from src.data_processing.text_processor import TextProcessor
from src.dataset_generation.llm_client import GroqLLMClient
from src.dataset_generation.qa_generator import QAGenerator
from src.utils.logging_config import logger
from src.utils.helpers import load_data_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config = load_data_config()

# Initialize FastAPI app
app = FastAPI(
    title="EAIB Web Scraping API",
    description="API for web scraping and content extraction pipeline",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ScrapeRequest(BaseModel):
    """Request model for scrape operations"""
    query: str = Field(..., description="Search query/topic", min_length=1, max_length=500)
    num_results: Optional[int] = Field(default=None, description="Number of results to fetch", ge=1, le=50)
    extract_content: Optional[bool] = Field(default=None, description="Whether to extract full content from URLs")
    preferred_sites: Optional[List[str]] = Field(default=None, description="List of preferred websites to search")
    use_default_sites: Optional[bool] = Field(default=None, description="Use default candidate sites if no preferred sites specified")
    search_strategy: Optional[str] = Field(default=None, description="Search strategy: mixed, targeted, broad")
    document_id: Optional[str] = Field(default=None, description="Document ID to save data to, generates random if not provided")
    
    @validator('search_strategy')
    def validate_search_strategy(cls, v):
        if v is not None and v not in ['mixed', 'targeted', 'broad']:
            raise ValueError('search_strategy must be one of: mixed, targeted, broad')
        return v

class ScrapeResponse(BaseModel):
    """Response model for scrape operations"""
    status: str
    document_id: str
    search_id: Optional[str] = None
    original_topic: str
    enhanced_topic: str
    total_results: int
    content_extraction_stats: Dict[str, Any]
    sample_results: List[Dict[str, Any]]
    candidate_sites_info: Dict[str, Any]
    processing_time: float
    timestamp: datetime

class PDFUploadResponse(BaseModel):
    """Response model for PDF upload operations"""
    status: str
    document_id: str
    filename: str
    page_count: int
    content_length: int
    content_preview: str
    processing_time: float
    timestamp: datetime

class ProcessResponse(BaseModel):
    """Response model for text processing operations"""
    status: str
    document_id: str
    original_length: int
    final_segments: int
    deduplication_ratio: float
    quality_retention: float
    processing_time: float
    message: Optional[str] = None

class QAGenerationResponse(BaseModel):
    """Response model for Q&A generation operations"""
    status: str
    document_id: str
    total_qa_pairs: int
    segments_processed: int
    processing_time: float
    files_created: List[str]
    qa_pairs: Optional[List[Dict[str, Any]]] = None
    message: Optional[str] = None

# Global pipeline instance
pipeline = None

def get_pipeline():
    """Get or create pipeline instance"""
    global pipeline
    if pipeline is None:
        pipeline = ScrapingPipeline()
    return pipeline

class ScrapingPipeline:
    """Main scraping pipeline orchestrator"""
    
    def __init__(self):
        self.config = load_data_config()
        self.scraper = WebScraper()
        self.storage = MongoStorage()
        self.llm_client = GroqLLMClient()
        self.text_processor = TextProcessor()
        self.qa_generator = QAGenerator()
        
        logger.info("Scraping pipeline initialized")
    
    def _get_param_with_default(self, param_value: Any, config_path: str, fallback: Any = None) -> Any:
        """Get parameter value with config default fallback"""
        if param_value is not None:
            return param_value
        
        # Navigate config path (e.g., "search.default_params.num_results")
        config_value = self.config
        path_parts = config_path.split('.')
        
        for i, key in enumerate(path_parts):
            if isinstance(config_value, dict) and key in config_value:
                config_value = config_value[key]
            else:
                logger.debug(f"Config path '{config_path}' not found, using fallback: {fallback}")
                return fallback
        
        return config_value
    
    def run_search_pipeline(
        self, 
        topic: str, 
        num_results: Optional[int] = None, 
        extract_content: Optional[bool] = None,
        preferred_sites: Optional[List[str]] = None,
        use_default_sites: Optional[bool] = None,
        search_strategy: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute complete search and processing pipeline"""
        start_time = datetime.now()
        logger.info(f"Starting pipeline for topic: '{topic}'")
        
        try:
            # Get parameters with config defaults
            num_results = self._get_param_with_default(num_results, "search.default_params.num_results", 10)
            extract_content = self._get_param_with_default(extract_content, "search.default_params.extract_content", True)
            use_default_sites = self._get_param_with_default(use_default_sites, "search.default_params.use_default_sites", True)
            search_strategy = self._get_param_with_default(search_strategy, "search.default_params.search_strategy", "mixed")
            preferred_sites = self._get_param_with_default(preferred_sites, "search.default_params.preferred_sites", None)
            
            # Generate or use provided document ID
            if not document_id:
                import uuid
                document_id = str(uuid.uuid4())[:12]
            
            # Update search configuration based on request parameters
            self._update_search_config(preferred_sites, use_default_sites, search_strategy)
            
            # Step 1: Use original topic (LLM enhancement disabled)
            logger.info("Step 1: Using original search topic")
            enhanced_topic = topic
            
            # Step 2: Perform web search using enhanced topic
            logger.info("Step 2: Performing web search")
            search_results = self.scraper.search_topic(enhanced_topic, num_results, extract_content)
            
            if not search_results:
                logger.warning("No search results found")
                return {
                    "status": "no_results", 
                    "topic": topic,
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
            
            # Step 3: Generate search summary
            logger.info("Step 3: Generating search summary")
            search_summary = self.scraper.get_search_summary(search_results)
            
            # Step 4: Process results (LLM processing disabled)
            logger.info("Step 4: Processing results")
            results_dict = [result.to_dict() for result in search_results]
            
            # LLM processing disabled as requested
            llm_summary = {"summary": "LLM processing disabled", "quality_score": 0}
            key_information = []
            follow_up_questions = []
            
            # Step 5: Store results in MongoDB
            logger.info("Step 5: Storing results in database")
            search_metadata = {
                **search_summary,
                'original_topic': topic,
                'enhanced_topic': enhanced_topic,
                'llm_summary': llm_summary,
                'key_information': key_information,
                'follow_up_questions': follow_up_questions,
                'preferred_sites': preferred_sites,
                'search_strategy': search_strategy
            }
            
            # Store or update document with search data
            search_id = self.storage.store_document_data(
                document_id, 
                data_type="search", 
                data={"results": results_dict, "metadata": search_metadata}
            )
            
            # Step 6: Prepare final output
            processing_time = (datetime.now() - start_time).total_seconds()
            
            pipeline_result = {
                "status": "success",
                "document_id": document_id,
                "search_id": search_id,
                "original_topic": topic,
                "enhanced_topic": enhanced_topic,
                "total_results": len(search_results),
                "content_extraction_stats": search_summary.get('content_extraction', {}),
                "sample_results": results_dict[:3],  # First 3 results for preview
                "candidate_sites_info": self.scraper.get_candidate_sites_info(),
                "processing_time": processing_time,
                "timestamp": datetime.now()
            }
            
            logger.info(f"Pipeline completed successfully for topic: '{topic}' in {processing_time:.2f}s")
            return pipeline_result
            
        except Exception as e:
            logger.error(f"Pipeline failed for topic '{topic}': {str(e)}")
            return {
                "status": "error",
                "topic": topic,
                "error": str(e),
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
    
    def _update_search_config(self, preferred_sites: Optional[List[str]], use_default_sites: bool, search_strategy: str):
        """Update search configuration based on request parameters"""
        candidate_sites_config = self.scraper.search_config.get('candidate_sites', {})
        
        if preferred_sites and not use_default_sites:
            # Use only preferred sites
            custom_sites = []
            for site in preferred_sites:
                custom_sites.append({
                    "name": f"Custom-{site}",
                    "domain": site,
                    "search_prefix": f"site:{site}",
                    "priority": 1,
                    "description": f"Custom site: {site}"
                })
            
            # Replace the entire configuration with custom sites only
            candidate_sites_config['sites'] = custom_sites
            candidate_sites_config['strategies']['default'] = search_strategy
            candidate_sites_config['strategies']['targeted_sites'] = preferred_sites
            
            # Ensure enabled
            candidate_sites_config['enabled'] = True
            
        elif preferred_sites and use_default_sites:
            # Add preferred sites to default sites
            default_sites = candidate_sites_config.get('sites', []).copy()
            existing_domains = {s['domain'] for s in default_sites}
            
            for site in preferred_sites:
                if site not in existing_domains:
                    default_sites.append({
                        "name": f"Custom-{site}",
                        "domain": site,
                        "search_prefix": f"site:{site}",
                        "priority": 1,
                        "description": f"Custom site: {site}"
                    })
            
            candidate_sites_config['sites'] = default_sites
            candidate_sites_config['strategies']['default'] = search_strategy
            
            # Update targeted sites to include preferred ones
            current_targeted = candidate_sites_config['strategies'].get('targeted_sites', [])
            updated_targeted = list(set(current_targeted + preferred_sites))
            candidate_sites_config['strategies']['targeted_sites'] = updated_targeted
            
        else:
            # Use default configuration
            candidate_sites_config['strategies']['default'] = search_strategy
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        try:
            db_stats = self.storage.get_stats()
            recent_searches = self.storage.get_recent_sessions(5)
            candidate_sites_info = self.scraper.get_candidate_sites_info()
            
            return {
                "database_stats": db_stats,
                "recent_searches": recent_searches,
                "candidate_sites_info": candidate_sites_info
            }
        except Exception as e:
            logger.error(f"Failed to get pipeline stats: {str(e)}")
            return {}
    
    def process_pdf_document(self, file_content: bytes, filename: str, document_id: Optional[str] = None) -> Dict[str, Any]:
        """Process uploaded PDF document"""
        start_time = datetime.now()
        logger.info(f"Processing PDF document: {filename}")
        
        try:
            # Generate or use provided document ID
            if not document_id:
                import uuid
                document_id = str(uuid.uuid4())[:12]
            
            # Parse PDF content
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            page_count = len(pdf_reader.pages)
            
            # Extract text from all pages
            full_text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    full_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1}: {str(e)}")
            
            # Clean and process text
            full_text = full_text.strip()
            content_length = len(full_text)
            
            # Create PDF metadata
            pdf_metadata = {
                'filename': filename,
                'page_count': page_count,
                'content_length': content_length,
                'upload_timestamp': datetime.now(),
                'file_size': len(file_content)
            }
            
            # Store or update document with PDF data
            stored_id = self.storage.store_document_data(
                document_id, 
                data_type="pdf", 
                data={
                    "metadata": pdf_metadata,
                    "content": full_text,
                    "content_preview": full_text[:500] + "..." if len(full_text) > 500 else full_text
                }
            )
            
            # Prepare response
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "status": "success",
                "document_id": document_id,
                "filename": filename,
                "page_count": page_count,
                "content_length": content_length,
                "content_preview": full_text[:300] + "..." if len(full_text) > 300 else full_text,
                "processing_time": processing_time,
                "timestamp": datetime.now()
            }
            
            logger.info(f"PDF processed successfully: {filename} ({page_count} pages, {content_length} chars)")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process PDF {filename}: {str(e)}")
            return {
                "status": "error",
                "filename": filename,
                "error": str(e),
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.storage.close_connection()
            logger.info("Pipeline cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {"message": "EAIB Web Scraping API", "version": "1.0.0"}

@app.post("/scrape", response_model=ScrapeResponse)
async def scrape_topic(request: ScrapeRequest):
    """Scrape web content for a topic"""
    try:
        pipeline = get_pipeline()
        result = pipeline.run_search_pipeline(
            topic=request.query,
            num_results=request.num_results,
            extract_content=request.extract_content,
            preferred_sites=request.preferred_sites,
            use_default_sites=request.use_default_sites,
            search_strategy=request.search_strategy,
            document_id=request.document_id
        )
        
        if result["status"] == "success":
            return ScrapeResponse(**result)
        elif result["status"] == "no_results":
            raise HTTPException(status_code=404, detail="No search results found")
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
            
    except Exception as e:
        logger.error(f"Scrape endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-pdf", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...), document_id: Optional[str] = None):
    """Upload and parse a PDF document"""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Read file content
        file_content = await file.read()
        
        # Validate file size (max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        if len(file_content) > max_size:
            raise HTTPException(status_code=400, detail="File size too large (max 10MB)")
        
        # Process PDF
        pipeline = get_pipeline()
        result = pipeline.process_pdf_document(file_content, file.filename, document_id)
        
        if result["status"] == "success":
            return PDFUploadResponse(**result)
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF upload endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-document/{document_id}", response_model=ProcessResponse)
async def process_document(document_id: str):
    """Process a document for Q&A dataset preparation"""
    try:
        pipeline = get_pipeline()
        result = pipeline.text_processor.process_document(document_id)
        
        if result["status"] == "success":
            return ProcessResponse(
                status=result["status"],
                document_id=document_id,
                original_length=result["original_length"],
                final_segments=result["final_segments"],
                deduplication_ratio=result["deduplication_ratio"],
                quality_retention=result["quality_retention"],
                processing_time=result["processing_time"]
            )
        else:
            raise HTTPException(status_code=500, detail=result.get("message", "Processing failed"))
            
    except Exception as e:
        logger.error(f"Document processing endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/process-stats")
async def get_processing_stats():
    """Get text processing statistics"""
    try:
        pipeline = get_pipeline()
        stats = pipeline.text_processor.get_processing_stats()
        return stats
    except Exception as e:
        logger.error(f"Processing stats endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-qa/{document_id}", response_model=QAGenerationResponse)
async def generate_qa_dataset(document_id: str, max_qa_per_segment: Optional[int] = None, max_questions: Optional[int] = None):
    """Generate Q&A dataset from processed document"""
    try:
        start_time = time.time()
        logger.info(f"Generating Q&A dataset for document: {document_id}")
        
        # Get parameters with config defaults
        pipeline = get_pipeline()
        max_qa_per_segment = pipeline._get_param_with_default(max_qa_per_segment, "qa_generation.defaults.max_qa_per_segment", 3)
        max_questions = pipeline._get_param_with_default(max_questions, "qa_generation.defaults.max_questions", None)
        
        # Generate Q&A dataset
        result = pipeline.qa_generator.generate_qa_dataset(document_id, max_qa_per_segment, max_questions)
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result.get("message", "Q&A generation failed"))
        
        # Create response
        response = QAGenerationResponse(
            status="success",
            document_id=document_id,
            total_qa_pairs=result["total_qa_pairs"],
            segments_processed=result["segments_processed"],
            processing_time=time.time() - start_time,
            files_created=result["files_created"],
            qa_pairs=result.get("qa_pairs", []),
            message=f"Generated {result['total_qa_pairs']} Q&A pairs from {result['segments_processed']} segments"
        )
        
        logger.info(f"Q&A generation completed for document {document_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Q&A generation endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/qa-stats")
async def get_qa_generation_stats():
    """Get Q&A generation statistics"""
    try:
        pipeline = get_pipeline()
        stats = pipeline.qa_generator.get_generation_stats()
        return stats
    except Exception as e:
        logger.error(f"Q&A stats endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if pipeline:
        pipeline.cleanup()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 