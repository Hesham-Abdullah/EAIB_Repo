# EAIB Web Scraping API

A FastAPI-based REST API for the web scraping and content extraction pipeline.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables
Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp env.example .env
```

### 3. Start the API Server
```bash
python scripts/start_api.py
```

The API will be available at `http://localhost:8000`

### 4. Test the API
```bash
python scripts/test_api.py
```

## 📚 API Endpoints

### Root Endpoint
- **GET** `/` - API information

### Search Operations
- **POST** `/search` - Perform web search and content extraction
- **GET** `/search/{search_id}` - Get specific search result
- **DELETE** `/search/{search_id}` - Delete search result

### Information Endpoints
- **GET** `/stats` - Get pipeline statistics
- **GET** `/candidate-sites` - Get candidate sites information

## 🔍 Search API Usage

### Basic Search
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "electric cars",
    "num_results": 5,
    "extract_content": true
  }'
```

### Targeted Search with Wikipedia
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "artificial intelligence",
    "num_results": 3,
    "extract_content": true,
    "preferred_sites": ["wikipedia.org"],
    "use_default_sites": false,
    "search_strategy": "targeted"
  }'
```

### Custom Sites Search
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "num_results": 4,
    "extract_content": true,
    "preferred_sites": ["github.com", "stackoverflow.com"],
    "use_default_sites": true,
    "search_strategy": "mixed"
  }'
```

## 📋 Request Parameters

### SearchRequest Model
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Search query/topic (1-500 chars) |
| `num_results` | integer | 10 | Number of results (1-50) |
| `extract_content` | boolean | true | Extract full content from URLs |
| `preferred_sites` | array | null | List of preferred websites |
| `use_default_sites` | boolean | true | Use default candidate sites |
| `search_strategy` | string | "mixed" | Strategy: mixed, targeted, broad |

## 🎯 Search Strategies

### Mixed Strategy (Default)
- Combines targeted sites with broad web search
- Uses default candidate sites + any preferred sites
- Good balance of quality and coverage

### Targeted Strategy
- Uses only specified candidate sites
- Higher quality but limited coverage
- Best for specific, authoritative sources

### Broad Strategy
- Uses broad web search with top priority sites
- Maximum coverage with some quality filtering
- Good for exploratory searches

## 🏗️ Response Format

### SearchResponse
```json
{
  "status": "success",
  "search_id": "abc123def456",
  "original_topic": "electric cars",
  "enhanced_topic": "(site:wikipedia.org electric cars) OR electric cars",
  "total_results": 5,
  "content_extraction_stats": {
    "successful": 4,
    "failed": 1,
    "total_content_length": 15000,
    "avg_content_length": 3750
  },
  "sample_results": [
    {
      "title": "Electric Car - Wikipedia",
      "url": "https://en.wikipedia.org/wiki/Electric_car",
      "extraction_status": "success",
      "content_length": 4500
    }
  ],
  "candidate_sites_info": {
    "enabled": true,
    "strategy": "mixed",
    "total_sites": 10
  },
  "processing_time": 12.5,
  "timestamp": "2024-01-15T10:30:00"
}
```

## 🎯 Candidate Sites

The API includes 10 pre-configured high-quality sites:

1. **Wikipedia** (Priority 1) - Comprehensive encyclopedia
2. **TechCrunch** (Priority 2) - Technology news
3. **Ars Technica** (Priority 2) - Technology and science
4. **MIT Technology Review** (Priority 2) - Technology insights
5. **Nature** (Priority 3) - Scientific research
6. **Science** (Priority 3) - Scientific publications
7. **IEEE Spectrum** (Priority 2) - Technology and engineering
8. **Stack Overflow** (Priority 4) - Programming Q&A
9. **GitHub** (Priority 4) - Open source projects
10. **Medium** (Priority 3) - Technical articles

## 🔧 Configuration

### Environment Variables
- `GROQ_API_KEY` - Groq API key for LLM processing
- `SERPER_API_KEY` - SerpAPI key for web search
- `MONGODB_CONNECTION_STRING` - MongoDB connection string
- `LOG_LEVEL` - Logging level (INFO, DEBUG, etc.)

### Configuration File
Edit `config/data_config.yaml` to modify:
- Candidate sites and priorities
- Search strategies
- Content extraction settings
- Database configuration

## 🚀 Performance Tips

1. **Use `extract_content: false`** for faster searches when content isn't needed
2. **Limit `num_results`** to 5-10 for faster processing
3. **Use targeted strategy** with specific sites for higher quality results
4. **Monitor processing_time** in responses to optimize queries

## 🐛 Troubleshooting

### Common Issues

1. **Connection Error**: Make sure the server is running on port 8000
2. **API Key Error**: Check your `.env` file has valid API keys
3. **MongoDB Error**: Ensure MongoDB is running and accessible
4. **No Results**: Try different search terms or broader strategy

### Debug Mode
Start with debug logging:
```bash
LOG_LEVEL=DEBUG python scripts/start_api.py
```

## 📖 API Documentation

Once the server is running, visit:
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔄 Development

### Running Tests
```bash
python scripts/test_api.py
```

### Hot Reload
The server runs with hot reload enabled, so changes to the code will automatically restart the server.

### Adding New Endpoints
1. Add new route functions in `src/api/main.py`
2. Define Pydantic models for request/response
3. Add error handling and logging
4. Update this README with new endpoint documentation 