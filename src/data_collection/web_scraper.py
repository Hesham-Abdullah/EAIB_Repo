import json
import requests
import time
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urljoin, urlparse

# Import crawl4ai for robust content extraction
from crawl4ai import AsyncWebCrawler, CacheMode

from ..utils.logging_config import logger
from ..utils.helpers import load_data_config, clean_text, generate_search_id

@dataclass
class SearchResult:
    """Data class for search results with full content"""
    title: str
    url: str
    snippet: str
    position: int
    source: str
    search_id: str
    timestamp: datetime
    
    # Full content fields
    raw_content: str = ""
    cleaned_content: str = ""
    content_length: int = 0
    extraction_status: str = "pending"  # pending, success, failed, skipped
    extraction_error: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with optimized structure"""
        result = {
            'title': self.title,
            'url': self.url,
            'snippet': self.snippet,
            'position': self.position,
            'source': self.source,
            'extraction_status': self.extraction_status,
            'content_length': self.content_length
        }
        
        # Only include content if extraction was successful
        if self.extraction_status == 'success':
            result['cleaned_content'] = self.cleaned_content
            # Only include raw_content if it contains meaningful HTML structure
            if self.raw_content and self._has_meaningful_html_content(self.raw_content):
                if len(self.raw_content) > len(self.cleaned_content) * 2 and self._contains_actual_content(self.raw_content):
                    result['raw_content'] = self.raw_content
        
        # Only include error if extraction failed
        if self.extraction_status == 'failed':
            result['extraction_error'] = self.extraction_error
        
        return result
    
    def _has_meaningful_html_content(self, html_content: str) -> bool:
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

class WebScraper:
    """Web scraper using SerpAPI for search results"""
    
    def __init__(self):
        self.config = load_data_config()
        self.serper_config = self.config['apis']['serper']
        self.search_config = self.config['search']
        
        logger.info("WebScraper initialized")
    
    def search_topic(self, topic: str, num_results: Optional[int] = None, extract_content: bool = True) -> List[SearchResult]:
        """Search for a topic using SerpAPI and return structured results with full content"""
        logger.info(f"Starting search for topic: {topic}")
        
        search_id = generate_search_id(topic)
        num_results = num_results or self.search_config['default_params']['num_results']
        
        # Prepare search query
        enhanced_query = self._enhance_query(topic) if self.search_config['query_enhancement']['enabled'] else topic
        
        try:
            # Make API request to SerpAPI
            results = self._call_serper_api(enhanced_query, num_results)
            
            # Parse and structure results with proper limit
            structured_results = self._parse_search_results(results, search_id, num_results)
            
            # Extract full content from each URL
            if extract_content and structured_results:
                logger.info(f"Extracting full content from {len(structured_results)} URLs...")
                structured_results = self._extract_content_from_results(structured_results)
            
            logger.info(f"Successfully retrieved {len(structured_results)} results for topic: {topic}")
            return structured_results
            
        except Exception as e:
            logger.error(f"Error searching for topic '{topic}': {str(e)}")
            raise
    
    def _enhance_query(self, topic: str) -> str:
        """Enhance search query with additional context if enabled"""
        if not self.search_config['query_enhancement']['add_context_keywords']:
            return topic
        
        # For now, keep it simple to test
        enhanced_query = topic
        logger.debug(f"Enhanced query: {enhanced_query}")
        return enhanced_query
    
    def _call_serper_api(self, query: str, num_results: int) -> Dict[str, Any]:
        """Make API call to SerpAPI with candidate site targeting"""
        url = f"{self.serper_config['base_url']}{self.serper_config['endpoints']['search']}"
        
        # Check if candidate sites are enabled
        candidate_sites_config = self.search_config.get('candidate_sites', {})
        if candidate_sites_config.get('enabled', False):
            # Use targeted search with candidate sites
            enhanced_query = self._build_targeted_query(query, num_results)
        else:
            # Use regular search
            enhanced_query = query
        
        payload = {
            "q": enhanced_query,
            "num": num_results,
            "gl": self.search_config['default_params']['country'],
            "hl": self.search_config['default_params']['language']
        }
        
        headers = {
            'X-API-KEY': self.serper_config['api_key'],
            'Content-Type': 'application/json'
        }
        
        logger.info(f"Making SerpAPI request with query: '{enhanced_query}'")
        logger.info(f"Payload: {payload}")
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            # Debug logging to see what we're getting
            logger.info(f"SerpAPI raw response keys: {list(result.keys())}")
            logger.info(f"SerpAPI organic results count: {len(result.get('organic', []))}")
            if 'organic' in result and result['organic']:
                logger.info(f"First organic result: {result['organic'][0]}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"SerpAPI request failed: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse SerpAPI response: {str(e)}")
            raise
    
    def _build_targeted_query(self, query: str, num_results: int) -> str:
        """Build a targeted query using candidate sites"""
        candidate_sites_config = self.search_config.get('candidate_sites', {})
        sites = candidate_sites_config.get('sites', [])
        strategies = candidate_sites_config.get('strategies', {})
        
        if not sites:
            return query
        
        # Get strategy
        strategy = strategies.get('default', 'mixed')
        
        if strategy == 'targeted':
            # Use only targeted sites
            targeted_sites = strategies.get('targeted_sites', [])
            site_prefixes = []
            for site in sites:
                if site['domain'] in targeted_sites:
                    site_prefixes.append(site['search_prefix'])
            
            if site_prefixes:
                # Combine multiple sites with OR operator
                site_query = " OR ".join([f"({prefix} {query})" for prefix in site_prefixes])
                return f"({site_query})"
        
        elif strategy == 'mixed':
            # Mix targeted sites with broad search
            targeted_sites = strategies.get('targeted_sites', [])
            site_prefixes = []
            for site in sites:
                if site['domain'] in targeted_sites:
                    site_prefixes.append(site['search_prefix'])
            
            if site_prefixes:
                # Use targeted sites first, then broad search
                site_query = " OR ".join([f"({prefix} {query})" for prefix in site_prefixes])
                return f"({site_query}) OR {query}"
        
        elif strategy == 'broad':
            # Use broad search with some site targeting
            # Select top priority sites
            priority_sites = [site for site in sites if site.get('priority', 5) <= 2]
            if priority_sites:
                site_prefixes = [site['search_prefix'] for site in priority_sites[:3]]
                site_query = " OR ".join([f"({prefix} {query})" for prefix in site_prefixes])
                return f"({site_query}) OR {query}"
        
        # Default to original query
        return query
    
    def get_candidate_sites_info(self) -> Dict[str, Any]:
        """Get information about configured candidate sites"""
        candidate_sites_config = self.search_config.get('candidate_sites', {})
        if not candidate_sites_config.get('enabled', False):
            return {"enabled": False, "sites": []}
        
        sites = candidate_sites_config.get('sites', [])
        strategies = candidate_sites_config.get('strategies', {})
        
        return {
            "enabled": True,
            "total_sites": len(sites),
            "strategy": strategies.get('default', 'mixed'),
            "targeted_sites": strategies.get('targeted_sites', []),
            "sites": [
                {
                    "name": site['name'],
                    "domain": site['domain'],
                    "priority": site.get('priority', 5),
                    "description": site.get('description', '')
                }
                for site in sites
            ]
        }
    
    def _parse_search_results(self, api_response: Dict[str, Any], search_id: str, num_results: int) -> List[SearchResult]:
        """Parse SerpAPI response into structured SearchResult objects"""
        results = []
        timestamp = datetime.now()
        
        logger.info(f"Parsing response with keys: {list(api_response.keys())}")
        
        # Parse organic search results
        organic_results = api_response.get('organic', [])
        logger.info(f"Found {len(organic_results)} organic results to parse, limiting to {num_results}")
        
        for i, result in enumerate(organic_results[:num_results]):
            try:
                search_result = SearchResult(
                    title=clean_text(result.get('title', '')),
                    url=result.get('link', ''),
                    snippet=clean_text(result.get('snippet', '')),
                    position=result.get('position', i + 1),
                    source='serper_organic',
                    search_id=search_id,
                    timestamp=timestamp
                )
                
                # Only add results with valid content
                if search_result.title and search_result.url:
                    results.append(search_result)
                    
            except Exception as e:
                logger.warning(f"Failed to parse search result {i}: {str(e)}")
                continue
        
        # Parse news results if available
        news_results = api_response.get('news', [])
        for i, result in enumerate(news_results):
            try:
                search_result = SearchResult(
                    title=clean_text(result.get('title', '')),
                    url=result.get('link', ''),
                    snippet=clean_text(result.get('snippet', '')),
                    position=len(results) + i + 1,
                    source='serper_news',
                    search_id=search_id,
                    timestamp=timestamp
                )
                
                if search_result.title and search_result.url:
                    results.append(search_result)
                    
            except Exception as e:
                logger.warning(f"Failed to parse news result {i}: {str(e)}")
                continue
        
        return results
    
    def get_search_summary(self, results: List[SearchResult]) -> Dict[str, Any]:
        """Generate a summary of search results"""
        if not results:
            return {}
        
        # Calculate content extraction stats
        successful_extractions = sum(1 for r in results if r.extraction_status == "success")
        failed_extractions = sum(1 for r in results if r.extraction_status == "failed")
        total_content_length = sum(r.content_length for r in results)
        
        return {
            'total_results': len(results),
            'sources': list(set(r.source for r in results)),
            'search_id': results[0].search_id,
            'timestamp': results[0].timestamp.isoformat(),
            'url_domains': list(set(r.url.split('/')[2] if len(r.url.split('/')) > 2 else r.url for r in results)),
            'content_extraction': {
                'successful': successful_extractions,
                'failed': failed_extractions,
                'total_content_length': total_content_length,
                'avg_content_length': total_content_length / len(results) if results else 0
            }
        }
    
    def _extract_content_from_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Extract full content from all search result URLs using crawl4ai"""
        
        if not results:
            return results
            
        try:
            # Use synchronous approach to avoid event loop conflicts
            return self._extract_content_sync(results)
            
        except Exception as e:
            logger.error(f"Failed to run content extraction: {str(e)}")
            # Fallback: mark all as failed
            for result in results:
                result.extraction_status = "failed"
                result.extraction_error = f"Extraction failed: {str(e)}"
            return results
    

    
    def _extract_content_sync(self, results: List[SearchResult]) -> List[SearchResult]:
        """Simple content extraction using requests"""
        
        import requests
        from bs4 import BeautifulSoup
        import time
        
        for i, result in enumerate(results):
            try:
                logger.info(f"Extracting content from URL {i+1}/{len(results)}: {result.url}")
                
                # Simple HTTP request with user agent
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                response = requests.get(result.url, headers=headers, timeout=30)
                response.raise_for_status()
                
                # Parse with BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract text content
                for script in soup(["script", "style"]):
                    script.decompose()
                
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                cleaned_content = ' '.join(chunk for chunk in chunks if chunk)
                
                # Update result
                result.raw_content = response.text
                result.cleaned_content = cleaned_content
                result.content_length = len(cleaned_content)
                result.extraction_status = "success"
                
                logger.info(f"✅ Extracted {len(cleaned_content)} characters from {result.url}")
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.warning(f"❌ Failed to extract content from {result.url}: {str(e)}")
                result.extraction_status = "failed"
                result.extraction_error = str(e)
        
        return results
    
    def _extract_enhanced_content(self, crawl_result, url: str) -> str:
        """Enhanced content extraction based on the provided example approach"""
        
        try:
            # Get HTML content from crawl4ai result
            html_content = ""
            if hasattr(crawl_result, 'html') and crawl_result.html:
                html_content = crawl_result.html
            elif hasattr(crawl_result, 'cleaned_html') and crawl_result.cleaned_html:
                html_content = crawl_result.cleaned_html
            elif hasattr(crawl_result, 'markdown') and crawl_result.markdown:
                # If we have markdown, use it directly
                return self._clean_markdown_content(crawl_result.markdown)
            else:
                raise Exception("No usable content found in crawl result")
            
            # Parse HTML with BeautifulSoup
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Extract relevant content sections
            relevant_sections = []
            
            # Look for sections with content-related keywords
            content_keywords = [
                "overview", "content", "article", "main", "section", "description",
                "information", "details", "summary", "about", "introduction"
            ]
            
            # First try to find main content areas
            main_content_selectors = [
                'main', 'article', 'section[role="main"]', '.main-content',
                '.content', '.post-content', '.entry-content', '.article-content',
                '.page-content', '.body-content'
            ]
            
            for selector in main_content_selectors:
                elements = soup.select(selector)
                for elem in elements:
                    # Skip navigation and menu elements
                    if self._is_navigation_element(elem):
                        continue
                    
                    text = self._extract_clean_text_from_element(elem)
                    if len(text) > 100:  # Only consider substantial content blocks
                        relevant_sections.append(text)
            
            # If no main content found, look for content with keywords
            if not relevant_sections:
                for tag in ['div', 'section', 'article', 'main']:
                    elements = soup.find_all(tag)
                    for elem in elements:
                        # Skip navigation and menu elements
                        if self._is_navigation_element(elem):
                            continue
                        
                        text = elem.get_text(separator=' ', strip=True)
                        if len(text) > 100:  # Only consider substantial content blocks
                            # Check if section contains content-related keywords
                            if any(keyword in text.lower() for keyword in content_keywords):
                                cleaned_text = self._extract_clean_text_from_element(elem)
                                if cleaned_text:
                                    relevant_sections.append(cleaned_text)
            
            # If still no content found, take the body content
            if not relevant_sections:
                body = soup.find('body')
                if body:
                    text = self._extract_clean_text_from_element(body)
                    if text:
                        relevant_sections.append(text)
            
            if not relevant_sections:
                logger.warning(f"Could not extract meaningful content from {url}")
                return ""
            
            # Combine and limit content
            combined_content = " ".join(relevant_sections)[:8000]  # Limit to 8000 chars
            
            # Final cleaning
            cleaned_content = self._final_clean_content(combined_content)
            
            logger.info(f"Successfully extracted {len(cleaned_content)} characters from {url}")
            return cleaned_content
            
        except Exception as e:
            logger.error(f"Enhanced content extraction failed for {url}: {str(e)}")
            return ""
    
    def _is_navigation_element(self, element) -> bool:
        """Check if element is navigation/menu related"""
        if not element:
            return False
        
        # Check class names for navigation indicators
        class_attr = str(element.get('class', [])).lower()
        nav_indicators = ['nav', 'menu', 'header', 'footer', 'sidebar', 'breadcrumb']
        if any(indicator in class_attr for indicator in nav_indicators):
            return True
        
        # Check for common navigation tags
        nav_tags = ['nav', 'menu', 'ul', 'li']
        if element.name in nav_tags:
            return True
        
        return False
    
    def _extract_clean_text_from_element(self, element) -> str:
        """Extract clean text from a single element"""
        if not element:
            return ""
        
        # Remove script and style elements
        for script in element(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = element.get_text(separator=' ', strip=True)
        
        # Clean up the text
        if text:
            import re
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            # Remove lines that are too short or contain navigation-like content
            lines = text.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if len(line) > 20:  # Only meaningful lines
                    # Skip navigation-like content
                    nav_words = ['menu', 'search', 'login', 'register', 'contact', 'home', 'about']
                    if not any(word in line.lower() for word in nav_words):
                        cleaned_lines.append(line)
            
            text = ' '.join(cleaned_lines)
            
            return text if len(text) > 50 else ""  # Only return substantial content
        
        return ""
    
    def _clean_markdown_content(self, markdown_content: str) -> str:
        """Clean markdown content"""
        if not markdown_content:
            return ""
        
        import re
        
        # Remove markdown formatting but keep text
        text = re.sub(r'#+\s*', '', markdown_content)  # Remove headers
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)  # Remove italic
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Remove links
        text = re.sub(r'`([^`]+)`', r'\1', text)  # Remove code
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def _final_clean_content(self, content: str) -> str:
        """Final cleaning of extracted content"""
        if not content:
            return ""
        
        import re
        
        # Remove CSS and HTML artifacts
        content = re.sub(r'\.\w+\s*{[^}]*}', '', content)
        content = re.sub(r'[a-zA-Z-]+:\s*[^;]+;', '', content)
        content = re.sub(r'var\(--[^)]+\)', '', content)
        
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        # Remove lines that are just artifacts
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and len(line) > 10 and not line.startswith('.') and not line.startswith('#'):
                # Check if line contains meaningful text
                if not re.match(r'^[.#\s{}:;]+$', line):
                    cleaned_lines.append(line)
        
        return ' '.join(cleaned_lines)
    
    def _extract_clean_text_from_soup(self, soup) -> str:
        """Extract clean text content from BeautifulSoup object, aggressively filtering out CSS and non-content elements"""
        
        # First, remove all style and script elements completely
        for element in soup.find_all(['style', 'script', 'noscript']):
            element.decompose()
        
        # Remove all elements with CSS classes that contain common non-content patterns
        css_patterns = [
            'nav', 'menu', 'sidebar', 'footer', 'header', 'advertisement', 'ad-',
            'banner', 'popup', 'modal', 'overlay', 'cookie', 'privacy', 'terms',
            'social', 'share', 'comment', 'related', 'recommended', 'widget',
            'module', 'component', 'layout', 'grid', 'container', 'wrapper'
        ]
        
        # Remove elements with CSS classes containing these patterns
        for pattern in css_patterns:
            for element in soup.find_all(class_=lambda x: x and pattern in x.lower()):
                element.decompose()
        
        # Remove elements with inline styles that hide content
        for element in soup.find_all(style=True):
            style_attr = element.get('style', '').lower()
            if any(hide_style in style_attr for hide_style in [
                'display: none', 'visibility: hidden', 'position: absolute',
                'opacity: 0', 'height: 0', 'width: 0', 'overflow: hidden'
            ]):
                element.decompose()
        
        # Remove common non-content elements
        unwanted_tags = [
            'nav', 'footer', 'header', 'aside', 'menu', 'menuitem', 'form',
            'input', 'button', 'select', 'textarea', 'iframe', 'embed',
            'canvas', 'svg', 'math', 'object', 'applet', 'embed'
        ]
        
        for tag in unwanted_tags:
            for element in soup.find_all(tag):
                element.decompose()
        
        # Remove elements with data attributes that indicate non-content
        for element in soup.find_all(attrs={'data-testid': True}):
            testid = element.get('data-testid', '').lower()
            if any(non_content in testid for non_content in ['nav', 'menu', 'sidebar', 'footer', 'header', 'ad']):
                element.decompose()
        
        # Extract text using a more focused approach
        text_elements = []
        
        # Priority 1: Main content containers
        main_selectors = [
            'main', 'article', 'section[role="main"]', '.main-content',
            '.content', '.post-content', '.entry-content', '.article-content'
        ]
        
        for selector in main_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = self._extract_text_from_element(element)
                if text:
                    text_elements.append(text)
        
        # Priority 2: Headings and paragraphs (only if no main content found)
        if not text_elements:
            content_selectors = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']
            for selector in content_selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = self._extract_text_from_element(element)
                    if text:
                        text_elements.append(text)
        
        # Priority 3: Fallback to body if still no content
        if not text_elements:
            body = soup.find('body')
            if body:
                text = self._extract_text_from_element(body)
                if text:
                    text_elements.append(text)
        
        # Combine and clean text
        if text_elements:
            combined_text = ' '.join(text_elements)
            return self._clean_extracted_text(combined_text)
        
        return ""
    
    def _extract_text_from_element(self, element) -> str:
        """Extract clean text from a single element, filtering out CSS and non-content"""
        if not element:
            return ""
        
        # Remove any remaining style elements within this element
        for style_elem in element.find_all('style'):
            style_elem.decompose()
        
        # Get text content
        text = element.get_text(separator=' ', strip=True)
        
        # Filter out CSS-like content
        if text:
            # Remove CSS rules and selectors
            import re
            # Remove CSS class definitions
            text = re.sub(r'\.\w+[^{]*{[^}]*}', '', text)
            # Remove CSS property definitions
            text = re.sub(r'[a-zA-Z-]+:\s*[^;]+;', '', text)
            # Remove CSS selectors
            text = re.sub(r'[.#][a-zA-Z0-9_-]+', '', text)
            # Remove CSS variables
            text = re.sub(r'var\(--[^)]+\)', '', text)
            # Remove CSS media queries
            text = re.sub(r'@media[^{]*{[^}]*}', '', text)
            
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            # Only return if meaningful content remains
            if len(text) > 20 and not text.startswith('.') and not text.startswith('#'):
                return text
        
        return ""
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        import re
        
        # Remove CSS-related content that might have slipped through
        text = re.sub(r'\.\w+[^{]*{[^}]*}', '', text)
        text = re.sub(r'[a-zA-Z-]+:\s*[^;]+;', '', text)
        text = re.sub(r'[.#][a-zA-Z0-9_-]+', '', text)
        text = re.sub(r'var\(--[^)]+\)', '', text)
        
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove lines that are just CSS or HTML artifacts
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and len(line) > 5 and not line.startswith('.') and not line.startswith('#'):
                # Check if line contains meaningful text (not just CSS)
                if not re.match(r'^[.#\s{}:;]+$', line):
                    cleaned_lines.append(line)
        
        return ' '.join(cleaned_lines) 