from typing import Dict, List, Any, Optional
from groq import Groq
import json

from ..utils.logging_config import logger
from ..utils.helpers import load_data_config

class GroqLLMClient:
    """Groq API client for content processing and enhancement"""
    
    def __init__(self):
        self.config = load_data_config()
        self.groq_config = self.config['apis']['groq']
        
        self.client = Groq(api_key=self.groq_config['api_key'])
        
        logger.info("GroqLLMClient initialized")
    
    def enhance_search_topic(self, topic: str) -> str:
        """Enhance search topic with better search terms using Groq"""
        try:
            prompt = f"""
Given the search topic: "{topic}"

Generate a better, more comprehensive search query that would help find the most relevant and detailed information about this topic. 

Requirements:
- Make it more specific and targeted
- Add relevant keywords that would improve search results
- Keep it concise (max 10 words)
- Focus on finding factual, detailed information

Return only the enhanced search query without any explanation.
"""
            
            response = self.client.chat.completions.create(
                model=self.groq_config['model'],
                messages=[{"role": "user", "content": prompt}],
                temperature=self.groq_config['temperature'],
                max_tokens=100
            )
            
            enhanced_query = response.choices[0].message.content.strip()
            logger.debug(f"Enhanced query: '{topic}' -> '{enhanced_query}'")
            return enhanced_query
            
        except Exception as e:
            logger.error(f"Failed to enhance search topic: {str(e)}")
            return topic  # Fallback to original topic
    
    def summarize_search_results(self, search_results: List[Dict[str, Any]], topic: str) -> Dict[str, Any]:
        """Generate a comprehensive summary of search results"""
        try:
            # Prepare content for summarization
            content_pieces = []
            for result in search_results[:8]:  # Limit to top 8 results
                content_pieces.append(f"Title: {result['title']}\nSnippet: {result['snippet']}\nURL: {result['url']}")
            
            combined_content = "\n\n---\n\n".join(content_pieces)
            
            prompt = f"""
Based on the following search results about "{topic}", provide a comprehensive summary:

{combined_content}

Please provide:
1. A detailed summary (3-4 sentences) of the key information found
2. List the main topics/themes covered
3. Identify any notable sources or authoritative websites
4. Rate the overall information quality (1-10 scale)

Format your response as valid JSON with keys: summary, main_topics, notable_sources, quality_score
"""
            
            response = self.client.chat.completions.create(
                model=self.groq_config['model'],
                messages=[{"role": "user", "content": prompt}],
                temperature=self.groq_config['temperature'],
                max_tokens=self.groq_config['max_tokens']
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to parse as JSON, fallback to text summary if it fails
            try:
                summary_data = json.loads(content)
            except json.JSONDecodeError:
                summary_data = {
                    "summary": content,
                    "main_topics": [],
                    "notable_sources": [],
                    "quality_score": 5
                }
            
            logger.info(f"Generated summary for topic: {topic}")
            return summary_data
            
        except Exception as e:
            logger.error(f"Failed to summarize search results: {str(e)}")
            return {
                "summary": "Summary generation failed",
                "main_topics": [],
                "notable_sources": [],
                "quality_score": 0
            }
    
    def extract_key_information(self, search_results: List[Dict[str, Any]], topic: str) -> List[Dict[str, Any]]:
        """Extract key information points from search results"""
        try:
            key_info = []
            
            for result in search_results[:5]:  # Process top 5 results
                prompt = f"""
From this search result about "{topic}":

Title: {result['title']}
Snippet: {result['snippet']}

Extract the most important factual information. Provide:
1. Key facts (bullet points)
2. Important numbers/statistics if any
3. Relevance score to the topic (1-10)

Format as JSON with keys: key_facts, statistics, relevance_score
"""
                
                response = self.client.chat.completions.create(
                    model=self.groq_config['model'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,  # Lower temperature for factual extraction
                    max_tokens=300
                )
                
                content = response.choices[0].message.content.strip()
                
                try:
                    extracted_info = json.loads(content)
                    extracted_info['source_url'] = result['url']
                    extracted_info['source_title'] = result['title']
                    key_info.append(extracted_info)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse extracted info for: {result['title']}")
                    continue
            
            logger.info(f"Extracted key information from {len(key_info)} results")
            return key_info
            
        except Exception as e:
            logger.error(f"Failed to extract key information: {str(e)}")
            return []
    
    def generate_follow_up_questions(self, topic: str, search_results: List[Dict[str, Any]]) -> List[str]:
        """Generate follow-up questions based on search results"""
        try:
            # Prepare brief content summary
            titles = [result['title'] for result in search_results[:5]]
            titles_text = "\n".join(f"- {title}" for title in titles)
            
            prompt = f"""
Based on search results for "{topic}" with these titles:

{titles_text}

Generate 5 follow-up questions that would help gather more specific or detailed information about this topic. Make the questions:
- Specific and actionable
- Different aspects of the topic
- Useful for deeper research

Return as a JSON array of questions.
"""
            
            response = self.client.chat.completions.create(
                model=self.groq_config['model'],
                messages=[{"role": "user", "content": prompt}],
                temperature=self.groq_config['temperature'],
                max_tokens=400
            )
            
            content = response.choices[0].message.content.strip()
            
            try:
                questions = json.loads(content)
                if isinstance(questions, list):
                    logger.info(f"Generated {len(questions)} follow-up questions")
                    return questions
            except json.JSONDecodeError:
                pass
            
            # Fallback: extract questions from text
            lines = content.split('\n')
            questions = [line.strip('- ').strip() for line in lines if line.strip() and '?' in line]
            return questions[:5]
            
        except Exception as e:
            logger.error(f"Failed to generate follow-up questions: {str(e)}")
            return [] 