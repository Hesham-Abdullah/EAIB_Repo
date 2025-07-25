# EAIB (End-to-End AI Builder)

**Complete pipeline for web scraping, content processing, and Q&A dataset generation for LLM fine-tuning.**

## Pipeline Overview

### End-to-End Workflow:
1. **Web Scraping** → SerpAPI search + content extraction with Playwright/Chromium
2. **PDF Processing** → Upload and extract text from PDF documents  
3. **Text Processing** → Clean, deduplicate, normalize, and segment content
4. **Q&A Generation** → Generate training datasets using Groq LLM API
5. **Data Storage** → MongoDB with unified document management

### Key Features:
- **Unified Interface**: Streamlit web app + FastAPI backend
- **Configurable Pipeline**: Live config editing with YAML management
- **Production Features**: Health checks, logging, error handling

## Quick Start

### Local Development
```bash
pip install -r requirements.txt
python scripts/setup_nlp.py  # Install NLP models
cp env.template .env         # Configure API keys
python app.py               # Start unified application
```

**Access Points:**
- Interface: http://localhost:8501
- API Docs: http://localhost:8000/docs

## API Endpoints

### Core Pipeline Endpoints

#### `POST /scrape` - Web Content Scraping
```json
{
  "query": "artificial intelligence",
  "document_id": "optional-id",
  "num_results": 10,
  "extract_content": true,
  "use_default_sites": true,
  "search_strategy": "mixed",
  "preferred_sites": ["wikipedia.org"]
}
```
**Parameters:**
- `query` (required): Search topic
- `document_id` (optional): Custom document ID (auto-generated if not provided)
- `num_results` (1-50): Number of search results
- `extract_content` (bool): Whether to extract full page content
- `use_default_sites` (bool): Use configured candidate sites
- `search_strategy` ("mixed"|"targeted"|"broad"): Search approach
- `preferred_sites` (array): Additional domains to search

#### `POST /upload-pdf?document_id=<id>` - PDF Upload
**Body:** Multipart form with PDF file
**Query:** `document_id` (optional) - Document ID to append data to

#### `POST /process-document/{document_id}` - Text Processing
**Response:** Processing statistics (segments, deduplication ratio, quality retention)

#### `POST /generate-qa/{document_id}` - Q&A Generation
**Parameters:**
- `max_qa_per_segment` (1-10): Questions per text segment
- `max_questions` (optional): Total question limit

### Utility Endpoints
- `GET /process-stats` - Processing statistics
- `GET /qa-stats` - Q&A generation statistics

## Configuration

### Main Config (`config/data_config.yaml`)

#### Search Configuration
```yaml
search:
  default_params:
    num_results: 10
    extract_content: true
    use_default_sites: true
    search_strategy: "mixed"
  candidate_sites:
    - { name: "Wikipedia", domain: "wikipedia.org" }
    - { name: "TechCrunch", domain: "techcrunch.com" }
```

#### API Configuration
```yaml
apis:
  groq:
    model: "llama3-8b-8192"
    temperature: 0.1
    max_tokens: 1000
  serper:
    base_url: "https://google.serper.dev"
  timeouts:
    scraping_minutes: 5
    processing_minutes: 3
    qa_generation_minutes: 10
```

#### Processing Configuration
```yaml
processing:
  defaults:
    min_segment_length: 100
    max_segment_length: 2000
    similarity_threshold: 0.8
    quality_threshold: 0.5
  content_extraction:
    min_content_length: 200
    max_content_length: 50000
```

#### Q&A Generation
```yaml
qa_generation:
  defaults:
    max_qa_per_segment: 3
    max_questions: null
    model: "llama-3.1-8b-instant"
    temperature: 0.3
    max_tokens: 2000
```

### Environment Variables (`.env`)
```bash
# Required API Keys
GROQ_API_KEY=your_groq_api_key
SERPER_API_KEY=your_serper_api_key

# Database
MONGODB_CONNECTION_STRING=mongodb://localhost:27017/
MONGODB_URI=mongodb://localhost:27017/scraping_pipeline

# Application
LOG_LEVEL=INFO
LOG_FILE=logs/scraper.log
```

## Architecture

### Components
- **FastAPI Backend**: API server with pipeline orchestration
- **Streamlit Frontend**: Interactive web interface with config editor
- **MongoDB**: Document storage with unified data model
- **Playwright**: Web scraping with Chromium browser
- **Groq API**: LLM integration for Q&A generation

### Data Flow
```
Search Query → SerpAPI → Content Extraction → MongoDB
     ↓
PDF Upload → Text Extraction → MongoDB (same document)
     ↓
Text Processing → Clean/Segment/Deduplicate → MongoDB
     ↓
Q&A Generation → Groq API → Training Dataset → Files + MongoDB
```

### Storage Structure
```
MongoDB Collections:
├── unified_documents    # Raw scraped + PDF data
├── cleaned_data        # Processed text segments  
├── qa_datasets         # Generated Q&A pairs
└── search_sessions     # Search metadata (legacy)
```

## Pipeline Logic

### 1. Web Scraping
- **Search**: SerpAPI with query enhancement and site targeting
- **Extraction**: Playwright-based content scraping with anti-bot handling
- **Cleaning**: HTML parsing, CSS removal, text normalization
- **Strategies**: Mixed (default sites + broad), Targeted (specific sites), Broad (no filtering)

### 2. Text Processing
- **Consolidation**: Merge scraped + PDF content by document ID
- **Cleaning**: Remove HTML, normalize Unicode, fix encoding
- **Segmentation**: Split into chunks (100-2000 chars)
- **Deduplication**: Remove exact and near-duplicate segments
- **Quality Filtering**: Content length, language detection, readability

### 3. Q&A Generation
- **Model**: Groq Llama-3.1-8b-instant
- **Process**: Generate questions per segment with rate limiting
- **Output**: JSON, TXT, JSONL formats + MongoDB storage
- **Controls**: Max questions per segment and total limits

## Usage Examples

### Complete Pipeline
```python
# Streamlit Interface
1. Navigate to http://localhost:8501
2. Use "Pipeline Execution" tab
3. Enter search query and run complete pipeline
4. Download generated Q&A dataset
```

### API Usage
```bash
# Scrape content
curl -X POST "http://localhost:8000/scrape" \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "num_results": 5}'

# Process the document
curl -X POST "http://localhost:8000/process-document/{document_id}"

# Generate Q&A dataset
curl -X POST "http://localhost:8000/generate-qa/{document_id}?max_questions=20"
```

## Dependencies

**Core**: FastAPI, Streamlit, MongoDB, Playwright, Groq API, SerpAPI  
**NLP**: spaCy, NLTK, sentence-transformers, scikit-learn  
**Processing**: pandas, numpy, beautifulsoup4, crawl4ai, PyPDF2  

## Monitoring

- **Health Checks**: Built-in endpoint monitoring
- **Logging**: Structured logging with loguru
- **Statistics**: Processing metrics and generation stats
- **Error Handling**: Graceful failures with detailed messages

## TODO

### LLM Fine-tuning & Training
- [ ] **Model Training Pipeline**: Integration with Hugging Face transformers for fine-tuning
- [ ] **Training Configuration**: YAML-based training parameter management
- [ ] **Dataset Formatting**: Convert Q&A datasets to training formats (JSONL, Alpaca, ChatML)
- [ ] **Multi-GPU Training**: Distributed training support with accelerate
- [ ] **LoRA/QLoRA Integration**: Parameter-efficient fine-tuning methods
- [ ] **Training Monitoring**: Weights & Biases integration for experiment tracking

### Model Deployment & Serving
- [ ] **VLLM Integration**: High-performance inference server deployment
- [ ] **Model Registry**: MLflow model versioning and artifact management
- [ ] **API Gateway**: Production-ready model serving with authentication
- [ ] **Load Balancing**: Multi-instance model deployment


### Model Evaluation
- [ ] **Automated Evaluation**: BLEU, ROUGE, perplexity metrics
- [ ] **Human Evaluation**: Interface for manual quality assessment
- [ ] **Performance Monitoring**: Real-time inference quality tracking

### Workflow Orchestration
- [ ] **Airflow Integration**: DAG-based pipeline orchestration
- [ ] **Pipeline Scheduling**: Automated data collection and training workflows


### Containerization & Deployment
- [ ] **Docker Containers**: Multi-service containerization with MongoDB
- [ ] **Docker Compose**: One-command deployment setup

- [ ] **CI/CD Pipeline**: GitHub Actions for automated testing and deployment



---

**Quick Commands:**
- **Start**: `python app.py`
- **Interface**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs 