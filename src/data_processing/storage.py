from typing import Dict, List, Any, Optional
from datetime import datetime
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from ..utils.logging_config import logger
from ..utils.helpers import load_data_config

class MongoStorage:
    """MongoDB storage interface for scraped data"""
    
    def __init__(self):
        self.config = load_data_config()
        self.db_config = self.config['database']['mongodb']
        
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self.collections: Dict[str, Collection] = {}
        
        self._connect()
    
    def _connect(self):
        """Establish connection to MongoDB"""
        try:
            connection_string = self.db_config['connection_string']
            self.client = MongoClient(connection_string)
            
            # Test connection
            self.client.admin.command('ping')
            
            self.db = self.client[self.db_config['database_name']]
            
            # Initialize collections
            for collection_name in self.db_config['collections'].values():
                self.collections[collection_name] = self.db[collection_name]
                
            logger.info(f"Connected to MongoDB database: {self.db_config['database_name']}")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
    
    def store_search_session(self, search_results: List[Dict[str, Any]], search_metadata: Dict[str, Any]) -> str:
        """Store complete search session as single optimized document with reduced redundancy"""
        try:
            collection = self.collections[self.db_config['collections']['search_sessions']]
            
            search_id = search_metadata.get('search_id', 'unknown')
            
            # Calculate statistics once to avoid redundancy
            total_results = len(search_results)
            successful_extractions = sum(1 for r in search_results if r.get('extraction_status') == 'success')
            failed_extractions = sum(1 for r in search_results if r.get('extraction_status') == 'failed')
            total_content_length = sum(r.get('content_length', 0) for r in search_results)
            avg_content_length = total_content_length / total_results if total_results > 0 else 0
            content_success_rate = (successful_extractions / total_results * 100) if total_results > 0 else 0
            
            # Extract domains and sources once
            domains = list(set(self._extract_domain(r.get('url', '')) for r in search_results if r.get('url')))
            sources = list(set(r.get('source', 'unknown') for r in search_results))
            
            # Calculate average snippet length
            total_snippet_length = sum(len(r.get('snippet', '')) for r in search_results)
            avg_snippet_length = total_snippet_length / total_results if total_results > 0 else 0
            
            # Extract keywords from successful content only
            successful_content = ' '.join(r.get('cleaned_content', '') for r in search_results if r.get('extraction_status') == 'success' and r.get('cleaned_content'))
            content_keywords = self._extract_content_keywords_from_text(successful_content) if successful_content else []
            
            # Create optimized single document structure
            session_document = {
                # Core session info
                'search_id': search_id,
                'original_topic': search_metadata.get('original_topic'),
                'enhanced_topic': search_metadata.get('enhanced_topic'),
                'created_at': datetime.now().isoformat(),
                
                # Optimized results array (remove redundant fields)
                'results': self._optimize_results_array(search_results),
                'results_count': total_results,
                
                # Consolidated statistics
                'stats': {
                    'total_results': total_results,
                    'sources': sources,
                    'domains': domains,
                    'avg_snippet_length': round(avg_snippet_length, 2),
                    'successful_extractions': successful_extractions,
                    'failed_extractions': failed_extractions,
                    'total_content_length': total_content_length,
                    'avg_content_length': round(avg_content_length, 2),
                    'content_success_rate': round(content_success_rate, 2)
                },
                
                # Searchable content (only essential for search)
                'searchable_content': {
                    'all_titles': ' '.join(r.get('title', '') for r in search_results),
                    'all_snippets': ' '.join(r.get('snippet', '') for r in search_results),
                    'all_cleaned_content': successful_content,
                    'content_keywords': content_keywords
                },
                
                # Processing metadata (only if exists)
                'processing_info': {
                    'llm_summary': search_metadata.get('llm_summary', {}),
                    'key_information': search_metadata.get('key_information', []),
                    'follow_up_questions': search_metadata.get('follow_up_questions', [])
                } if any(search_metadata.get(k) for k in ['llm_summary', 'key_information', 'follow_up_questions']) else None,
                
                # Search configuration (only essential)
                'search_config': {
                    'num_results_requested': search_metadata.get('total_results', 0),
                    'api_source': 'serper'
                }
            }
            
            # Insert single document
            insert_result = collection.insert_one(session_document)
            
            logger.info(f"Stored search session {search_id} with {len(search_results)} results as single document")
            
            return search_id
                
        except Exception as e:
            logger.error(f"Failed to store search session: {str(e)}")
            raise
    
    def store_document_data(self, document_id: str, data_type: str, data: Dict[str, Any]) -> str:
        """Store or update document data in MongoDB with unified document structure"""
        try:
            # Ensure connection
            self._connect()
            
            collection = self.db['unified_documents']
            timestamp = datetime.now()
            
            # Check if document already exists
            existing_doc = collection.find_one({"document_id": document_id})
            
            if existing_doc:
                # Update existing document
                update_data = {
                    f"{data_type}_data": data,
                    f"{data_type}_timestamp": timestamp,
                    "last_updated": timestamp
                }
                
                collection.update_one(
                    {"document_id": document_id},
                    {"$set": update_data}
                )
                
                logger.info(f"Updated document {document_id} with {data_type} data")
                return document_id
                
            else:
                # Create new document
                document = {
                    "document_id": document_id,
                    "created_at": timestamp,
                    "last_updated": timestamp,
                    f"{data_type}_data": data,
                    f"{data_type}_timestamp": timestamp
                }
                
                result = collection.insert_one(document)
                logger.info(f"Created new document {document_id} with {data_type} data")
                return document_id
                
        except Exception as e:
            logger.error(f"Failed to store document data: {str(e)}")
            raise
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve unified document by ID"""
        try:
            self._connect()
            collection = self.db['unified_documents']
            document = collection.find_one({"document_id": document_id})
            return document
        except Exception as e:
            logger.error(f"Failed to retrieve document {document_id}: {str(e)}")
            return None
    
    def store_pdf_document(self, document_data: Dict[str, Any]) -> str:
        """Store PDF document data in MongoDB (legacy method)"""
        try:
            # Ensure connection
            self._connect()
            
            # Insert document
            collection = self.db['pdf_documents']
            result = collection.insert_one(document_data)
            
            document_id = document_data['metadata']['document_id']
            logger.info(f"Stored PDF document {document_id} ({document_data['metadata']['filename']})")
            
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Failed to store PDF document: {str(e)}")
            raise
    
    def get_pdf_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve PDF document by ID (legacy method)"""
        try:
            self._connect()
            collection = self.db['pdf_documents']
            document = collection.find_one({"metadata.document_id": document_id})
            return document
        except Exception as e:
            logger.error(f"Failed to retrieve PDF document {document_id}: {str(e)}")
            return None
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            if not url:
                return 'unknown'
            parts = url.split('/')
            if len(parts) >= 3:
                return parts[2]
            return url
        except:
            return 'unknown'
    
    def _extract_content_keywords_from_text(self, text: str) -> List[str]:
        """Extract key terms from text content"""
        try:
            if not text:
                return []
            
            # Simple keyword extraction (could be enhanced with NLP libraries)
            words = text.lower().split()
            
            # Filter out common words and short words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
            
            # Count word frequencies
            word_freq = {}
            for word in words:
                # Clean word
                word = ''.join(c for c in word if c.isalnum()).lower()
                if len(word) >= 3 and word not in stop_words:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top keywords
            top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
            
            return [word for word, freq in top_keywords if freq >= 2]  # Only words that appear at least twice
            
        except Exception as e:
            logger.warning(f"Failed to extract keywords: {str(e)}")
            return []
    
    def _optimize_results_array(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize results array by removing redundant data and compressing content"""
        optimized_results = []
        
        for result in search_results:
            # Create optimized result with only essential fields
            optimized_result = {
                'title': result.get('title', ''),
                'url': result.get('url', ''),
                'snippet': result.get('snippet', ''),
                'position': result.get('position', 0),
                'source': result.get('source', 'unknown'),
                'extraction_status': result.get('extraction_status', 'pending'),
                'content_length': result.get('content_length', 0)
            }
            
            # Only include content if extraction was successful
            if result.get('extraction_status') == 'success':
                optimized_result['cleaned_content'] = result.get('cleaned_content', '')
                # Only include raw_content if it contains meaningful HTML structure (not just CSS)
                raw_content = result.get('raw_content', '')
                cleaned_content = result.get('cleaned_content', '')
                
                # Check if raw_content contains meaningful HTML (not just CSS)
                if raw_content and self._has_meaningful_html(raw_content):
                    # Only store if it's significantly different and contains actual content
                    if len(raw_content) > len(cleaned_content) * 2 and self._contains_actual_content(raw_content):
                        optimized_result['raw_content'] = raw_content
            
            # Only include error if extraction failed
            if result.get('extraction_status') == 'failed':
                optimized_result['extraction_error'] = result.get('extraction_error', '')
            
            optimized_results.append(optimized_result)
        
        return optimized_results
    
    def _has_meaningful_html(self, html_content: str) -> bool:
        """Check if HTML content contains meaningful structure (not just CSS)"""
        if not html_content:
            return False
        
        import re
        
        # Check for CSS-heavy content
        css_patterns = [
            r'\.\w+\s*{[^}]*}',  # CSS class definitions
            r'[a-zA-Z-]+:\s*[^;]+;',  # CSS properties
            r'var\(--[^)]+\)',  # CSS variables
            r'@media[^{]*{[^}]*}',  # CSS media queries
        ]
        
        css_count = 0
        for pattern in css_patterns:
            css_count += len(re.findall(pattern, html_content))
        
        # Check for meaningful HTML tags
        html_tags = ['<h1>', '<h2>', '<h3>', '<h4>', '<h5>', '<h6>', '<p>', '<div>', '<span>', '<article>', '<section>', '<main>']
        html_count = sum(html_content.count(tag) for tag in html_tags)
        
        # If CSS patterns dominate, it's not meaningful HTML
        if css_count > html_count * 2:
            return False
        
        return True
    
    def _contains_actual_content(self, html_content: str) -> bool:
        """Check if HTML content contains actual text content (not just markup)"""
        if not html_content:
            return False
        
        import re
        
        # Remove HTML tags to get text content
        text_content = re.sub(r'<[^>]+>', '', html_content)
        
        # Remove CSS and JavaScript
        text_content = re.sub(r'\.\w+\s*{[^}]*}', '', text_content)
        text_content = re.sub(r'[a-zA-Z-]+:\s*[^;]+;', '', text_content)
        text_content = re.sub(r'var\(--[^)]+\)', '', text_content)
        
        # Clean whitespace
        text_content = re.sub(r'\s+', ' ', text_content).strip()
        
        # Check if there's meaningful text content
        return len(text_content) > 100  # At least 100 characters of actual text
    
    def get_search_session(self, search_id: str) -> Dict[str, Any]:
        """Retrieve complete search session by ID"""
        try:
            collection = self.collections[self.db_config['collections']['search_sessions']]
            
            session = collection.find_one({'search_id': search_id})
            
            if session:
                session.pop('_id', None)  # Remove MongoDB _id
                logger.info(f"Retrieved search session: {search_id}")
                return session
            else:
                logger.warning(f"No session found for ID: {search_id}")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to retrieve search session {search_id}: {str(e)}")
            return {}
    
    def get_session_results(self, search_id: str) -> List[Dict[str, Any]]:
        """Extract just the results array from a search session"""
        try:
            session = self.get_search_session(search_id)
            results = session.get('results', [])
            
            logger.info(f"Retrieved {len(results)} results from session: {search_id}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve session results for ID {search_id}: {str(e)}")
            return []
    
    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent search sessions with summary info"""
        try:
            collection = self.collections[self.db_config['collections']['search_sessions']]
            
            # Project only summary fields for efficiency
            projection = {
                'search_id': 1,
                'original_topic': 1,
                'enhanced_topic': 1,
                'created_at': 1,
                'results_count': 1,
                'stats': 1
            }
            
            sessions = list(
                collection.find({}, projection)
                .sort('created_at', -1)
                .limit(limit)
            )
            
            # Remove MongoDB _id field
            for session in sessions:
                session.pop('_id', None)
            
            logger.info(f"Retrieved {len(sessions)} recent search sessions")
            return sessions
            
        except Exception as e:
            logger.error(f"Failed to retrieve recent sessions: {str(e)}")
            return []
    
    def search_content(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search across all stored content using text search"""
        try:
            collection = self.collections[self.db_config['collections']['search_sessions']]
            
            # Use MongoDB text search on searchable content
            search_filter = {
                '$or': [
                    {'searchable_content.all_titles': {'$regex': query, '$options': 'i'}},
                    {'searchable_content.all_snippets': {'$regex': query, '$options': 'i'}},
                    {'original_topic': {'$regex': query, '$options': 'i'}}
                ]
            }
            
            sessions = list(collection.find(search_filter).limit(limit))
            
            # Remove MongoDB _id field
            for session in sessions:
                session.pop('_id', None)
            
            logger.info(f"Found {len(sessions)} sessions matching query: {query}")
            return sessions
            
        except Exception as e:
            logger.error(f"Failed to search content: {str(e)}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimized database statistics"""
        try:
            collection = self.collections[self.db_config['collections']['search_sessions']]
            
            # Aggregate statistics
            pipeline = [
                {
                    '$group': {
                        '_id': None,
                        'total_sessions': {'$sum': 1},
                        'total_results': {'$sum': '$results_count'},
                        'unique_topics': {'$addToSet': '$original_topic'},
                        'avg_results_per_session': {'$avg': '$results_count'}
                    }
                }
            ]
            
            agg_result = list(collection.aggregate(pipeline))
            
            if agg_result:
                stats = agg_result[0]
                stats['unique_topics_count'] = len(stats.get('unique_topics', []))
                stats.pop('_id', None)
                stats.pop('unique_topics', None)  # Don't return the full list
            else:
                stats = {
                    'total_sessions': 0,
                    'total_results': 0,
                    'unique_topics_count': 0,
                    'avg_results_per_session': 0
                }
            
            stats['database_name'] = self.db_config['database_name']
            stats['timestamp'] = datetime.now().isoformat()
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {str(e)}")
            return {}
    
    def close_connection(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed") 