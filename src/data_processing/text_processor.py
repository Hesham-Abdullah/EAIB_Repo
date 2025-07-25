#!/usr/bin/env python3
"""
Text Processing Pipeline for Q&A Dataset Preparation
Handles data extraction, cleaning, deduplication, normalization, and tokenization
"""

import re
import hashlib
import unicodedata
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
import logging
from collections import defaultdict, Counter

# NLP and processing libraries
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import langdetect
import ftfy
import textstat

# Database
from .storage import MongoStorage

# Configure logging
logger = logging.getLogger(__name__)

class TextProcessor:
    """Main text processing pipeline for cleaning and preparing Q&A datasets"""
    
    def __init__(self):
        """Initialize the text processor with required models and tools"""
        self.storage = MongoStorage()
        
        # Initialize NLP models
        self._init_nlp_models()
        
        # Processing statistics
        self.stats = {
            'documents_processed': 0,
            'segments_created': 0,
            'duplicates_removed': 0,
            'quality_filtered': 0,
            'total_processing_time': 0.0
        }
        
        logger.info("TextProcessor initialized successfully")
    
    def _init_nlp_models(self):
        """Initialize NLP models and download required data"""
        try:
            # Load spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
            
            # Initialize sentence transformer for semantic similarity
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}")
                self.sentence_model = None
            
            # Download NLTK data
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
            except Exception as e:
                logger.warning(f"Failed to download NLTK data: {e}")
            
            # Initialize TF-IDF vectorizer for deduplication
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize NLP models: {e}")
            raise
    
    def process_document(self, document_id: str) -> Dict[str, Any]:
        """
        Main processing pipeline for a document
        
        Args:
            document_id: ID of document in unified_documents collection
            
        Returns:
            Dictionary with processing results and statistics
        """
        start_time = datetime.now()
        logger.info(f"Starting text processing for document: {document_id}")
        
        try:
            # Step 1: Data Extraction & Consolidation
            raw_content = self._extract_and_consolidate(document_id)
            if not raw_content:
                return {"status": "error", "message": "No content found"}
            
            # Step 2: Text Cleaning & Preprocessing
            cleaned_content = self._clean_and_preprocess(raw_content)
            
            # Step 3: Content Segmentation
            segments = self._segment_content(cleaned_content)
            
            # Step 4: Deduplication
            unique_segments = self._deduplicate_segments(segments)
            
            # Step 5: Quality Filtering
            quality_segments = self._quality_filter(unique_segments)
            
            # Step 6: Data Normalization & Tokenization
            processed_segments = self._normalize_and_tokenize(quality_segments)
            
            # Step 7: Store results
            result = self._store_processed_data(document_id, processed_segments, raw_content)
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats['documents_processed'] += 1
            self.stats['total_processing_time'] += processing_time
            
            result.update({
                "status": "success",
                "processing_time": processing_time,
                "original_length": len(raw_content.get('combined_text', '')),
                "final_segments": len(processed_segments),
                "deduplication_ratio": 1 - (len(unique_segments) / max(len(segments), 1)),
                "quality_retention": len(quality_segments) / max(len(unique_segments), 1)
            })
            
            logger.info(f"Document {document_id} processed successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process document {document_id}: {str(e)}")
            return {
                "status": "error", 
                "message": str(e),
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
    
    def _extract_and_consolidate(self, document_id: str) -> Dict[str, Any]:
        """
        Step 1: Extract and consolidate text from unified document
        
        Returns:
            Dictionary with consolidated text content and metadata
        """
        logger.info(f"Extracting content for document: {document_id}")
        
        # Get document from database
        document = self.storage.get_document(document_id)
        if not document:
            raise ValueError(f"Document {document_id} not found")
        
        consolidated = {
            'search_content': [],
            'pdf_content': '',
            'combined_text': '',
            'metadata': {
                'document_id': document_id,
                'has_search_data': False,
                'has_pdf_data': False,
                'extraction_timestamp': datetime.now()
            }
        }
        
        # Extract search data content
        search_data = document.get('search_data', {})
        if search_data and 'results' in search_data:
            consolidated['metadata']['has_search_data'] = True
            
            for result in search_data['results']:
                if result.get('cleaned_content'):
                    consolidated['search_content'].append({
                        'title': result.get('title', ''),
                        'url': result.get('url', ''),
                        'content': result['cleaned_content'],
                        'source': 'web_scrape'
                    })
        
        # Extract PDF content
        pdf_data = document.get('pdf_data', {})
        if pdf_data and 'content' in pdf_data:
            consolidated['metadata']['has_pdf_data'] = True
            consolidated['pdf_content'] = pdf_data['content']
        
        # Combine all text content
        all_text = []
        
        # Add search content
        for item in consolidated['search_content']:
            if item['title']:
                all_text.append(f"Title: {item['title']}")
            if item['content']:
                all_text.append(item['content'])
        
        # Add PDF content
        if consolidated['pdf_content']:
            all_text.append(consolidated['pdf_content'])
        
        consolidated['combined_text'] = '\n\n'.join(all_text)
        
        logger.info(f"Extracted {len(consolidated['combined_text'])} characters from document {document_id}")
        return consolidated
    
    def _clean_and_preprocess(self, content: Dict[str, Any]) -> str:
        """
        Step 2: Clean and preprocess text content
        
        Returns:
            Cleaned text string
        """
        logger.info("Starting text cleaning and preprocessing")
        
        text = content.get('combined_text', '')
        if not text:
            return ''
        
        # Fix encoding issues
        text = ftfy.fix_text(text)
        
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Remove HTML/XML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page markers from PDFs
        text = re.sub(r'--- Page \d+ ---', ' ', text)
        
        # Remove repeated punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Fix common OCR errors
        text = re.sub(r'\b(\w)\1{3,}\b', r'\1', text)  # Remove repeated characters
        
        # Remove very short lines (likely formatting artifacts)
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 10]
        text = '\n'.join(cleaned_lines)
        
        # Final cleanup
        text = text.strip()
        
        logger.info(f"Text cleaned: {len(text)} characters remaining")
        return text
    
    def _segment_content(self, text: str) -> List[Dict[str, Any]]:
        """
        Segment content into meaningful chunks
        
        Returns:
            List of text segments with metadata
        """
        logger.info("Segmenting content into chunks")
        
        if not text:
            return []
        
        segments = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        for para_idx, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if len(paragraph) < 50:  # Skip very short paragraphs
                continue
            
            # Further split long paragraphs by sentences
            if self.nlp:
                doc = self.nlp(paragraph)
                sentences = [sent.text.strip() for sent in doc.sents]
            else:
                # Fallback sentence splitting
                sentences = re.split(r'[.!?]+', paragraph)
                sentences = [s.strip() for s in sentences if s.strip()]
            
            # Group sentences into segments of reasonable length
            current_segment = []
            current_length = 0
            target_length = 200  # Target words per segment
            
            for sentence in sentences:
                word_count = len(sentence.split())
                
                if current_length + word_count > target_length and current_segment:
                    # Create segment
                    segment_text = ' '.join(current_segment)
                    if len(segment_text) > 50:  # Minimum segment length
                        segments.append({
                            'text': segment_text,
                            'word_count': current_length,
                            'paragraph_index': para_idx,
                            'hash': hashlib.md5(segment_text.encode()).hexdigest()
                        })
                    
                    current_segment = [sentence]
                    current_length = word_count
                else:
                    current_segment.append(sentence)
                    current_length += word_count
            
            # Add remaining segment
            if current_segment:
                segment_text = ' '.join(current_segment)
                if len(segment_text) > 50:
                    segments.append({
                        'text': segment_text,
                        'word_count': current_length,
                        'paragraph_index': para_idx,
                        'hash': hashlib.md5(segment_text.encode()).hexdigest()
                    })
        
        logger.info(f"Created {len(segments)} text segments")
        return segments
    
    def _deduplicate_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step 3: Remove duplicate and near-duplicate segments
        
        Returns:
            List of unique segments
        """
        logger.info("Starting deduplication process")
        
        if not segments:
            return []
        
        # Exact deduplication using hashes
        seen_hashes = set()
        unique_segments = []
        
        for segment in segments:
            if segment['hash'] not in seen_hashes:
                seen_hashes.add(segment['hash'])
                unique_segments.append(segment)
        
        exact_duplicates_removed = len(segments) - len(unique_segments)
        logger.info(f"Removed {exact_duplicates_removed} exact duplicates")
        
        # Near-duplicate detection using TF-IDF similarity
        if len(unique_segments) > 1 and len(unique_segments) < 1000:  # Only for reasonable sizes
            try:
                texts = [seg['text'] for seg in unique_segments]
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
                similarity_matrix = cosine_similarity(tfidf_matrix)
                
                # Find near-duplicates (similarity > 0.8)
                to_remove = set()
                for i in range(len(similarity_matrix)):
                    for j in range(i + 1, len(similarity_matrix)):
                        if similarity_matrix[i][j] > 0.8:
                            # Keep the longer segment
                            if unique_segments[i]['word_count'] >= unique_segments[j]['word_count']:
                                to_remove.add(j)
                            else:
                                to_remove.add(i)
                
                # Remove near-duplicates
                final_segments = [seg for idx, seg in enumerate(unique_segments) if idx not in to_remove]
                
                near_duplicates_removed = len(unique_segments) - len(final_segments)
                logger.info(f"Removed {near_duplicates_removed} near-duplicates")
                
                unique_segments = final_segments
                
            except Exception as e:
                logger.warning(f"Near-duplicate detection failed: {e}")
        
        self.stats['duplicates_removed'] += len(segments) - len(unique_segments)
        return unique_segments
    
    def _quality_filter(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter segments based on quality metrics
        
        Returns:
            List of high-quality segments
        """
        logger.info("Applying quality filtering")
        
        if not segments:
            return []
        
        quality_segments = []
        
        for segment in segments:
            text = segment['text']
            word_count = segment['word_count']
            
            # Quality checks
            quality_score = 0
            max_score = 5
            
            # Length check (not too short or too long)
            if 20 <= word_count <= 300:
                quality_score += 1
            
            # Language detection
            try:
                if langdetect.detect(text) == 'en':
                    quality_score += 1
            except:
                pass
            
            # Readability check
            try:
                readability = textstat.flesch_reading_ease(text)
                if 30 <= readability <= 90:  # Reasonable readability range
                    quality_score += 1
            except:
                pass
            
            # Coherence check (proper sentence structure)
            try:
                sentence_count = len(re.split(r'[.!?]+', text))
                if sentence_count >= 1 and len(text) / sentence_count > 10:  # Reasonable sentence length
                    quality_score += 1
            except Exception as e:
                logger.warning(f"Coherence check failed: {e}")
            
            # Content density (not just lists or navigation)
            try:
                # Check if text is not just punctuation/numbers/whitespace
                if not re.match(r'^[\s\*\d\.\(\)\-]*$', text) and len(text.split()) > len(re.findall(r'\d+', text)):
                    quality_score += 1
            except Exception as e:
                logger.warning(f"Content density check failed: {e}")
                # Default to including the segment if regex fails
                quality_score += 1
            
            # Keep segments with quality score >= 3
            if quality_score >= 3:
                segment['quality_score'] = quality_score / max_score
                quality_segments.append(segment)
        
        filtered_count = len(segments) - len(quality_segments)
        self.stats['quality_filtered'] += filtered_count
        
        logger.info(f"Quality filtering: kept {len(quality_segments)}/{len(segments)} segments")
        return quality_segments
    
    def _normalize_and_tokenize(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step 4: Normalize and tokenize text segments
        
        Returns:
            List of normalized and tokenized segments
        """
        logger.info("Starting normalization and tokenization")
        
        if not segments:
            return []
        
        processed_segments = []
        
        for segment in segments:
            text = segment['text']
            
            try:
                # Text normalization
                normalized_text = self._normalize_text(text)
                
                # Tokenization and NLP processing
                if self.nlp:
                    try:
                        doc = self.nlp(normalized_text)
                        
                        # Extract linguistic features
                        tokens = [token.text for token in doc if not token.is_space]
                        lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
                        entities = [(ent.text, ent.label_) for ent in doc.ents]
                        
                        # POS tags
                        pos_tags = [(token.text, token.pos_) for token in doc if not token.is_space]
                        
                        segment.update({
                            'normalized_text': normalized_text,
                            'tokens': tokens,
                            'lemmas': lemmas,
                            'entities': entities,
                            'pos_tags': pos_tags,
                            'token_count': len(tokens)
                        })
                    except Exception as e:
                        logger.warning(f"spaCy processing failed for segment: {e}")
                        # Fallback to simple tokenization
                        tokens = normalized_text.split()
                        segment.update({
                            'normalized_text': normalized_text,
                            'tokens': tokens,
                            'token_count': len(tokens)
                        })
                else:
                    # Fallback tokenization with error handling
                    try:
                        tokens = re.findall(r'\b\w+\b', normalized_text.lower())
                    except Exception as e:
                        logger.warning(f"Fallback tokenization failed: {e}")
                        # Simple split as last resort
                        tokens = normalized_text.lower().split()
                    
                    segment.update({
                        'normalized_text': normalized_text,
                        'tokens': tokens,
                        'token_count': len(tokens)
                    })
            
            except Exception as e:
                logger.error(f"Failed to process segment: {e}")
                # Add minimal segment data to continue processing
                segment.update({
                    'normalized_text': text,
                    'tokens': text.split(),
                    'token_count': len(text.split())
                })
            
            processed_segments.append(segment)
        
        self.stats['segments_created'] += len(processed_segments)
        logger.info(f"Normalized and tokenized {len(processed_segments)} segments")
        return processed_segments
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text content
        
        Returns:
            Normalized text
        """
        try:
            # Convert to lowercase for consistency (keeping original for display)
            # Fix common abbreviations
            text = re.sub(r'\bU\.S\.A?\b', 'United States', text)
            text = re.sub(r'\bAI\b', 'artificial intelligence', text)
            text = re.sub(r'\bML\b', 'machine learning', text)
            
            # Normalize quotes (use Unicode escapes)
            text = re.sub(r'[\u201c\u201d\u201e]', '"', text)  # Various double quotes
            text = re.sub(r'[\u2018\u2019\u201a]', "'", text)  # Various single quotes
            
            # Normalize dashes (use Unicode escapes)
            text = re.sub(r'[\u2013\u2014\u2212]', '-', text)  # En dash, em dash, minus sign
            
            # Normalize numbers
            text = re.sub(r'\b\d{1,3}(?:,\d{3})+\b', lambda m: m.group().replace(',', ''), text)
            
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
        except Exception as e:
            logger.warning(f"Text normalization failed: {e}")
            # Return original text if normalization fails
            text = text.strip()
        
        return text
    
    def _store_processed_data(self, document_id: str, segments: List[Dict[str, Any]], raw_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store processed data in cleaned_data collection
        
        Returns:
            Storage result dictionary
        """
        logger.info(f"Storing processed data for document: {document_id}")
        
        # Prepare document for storage
        cleaned_document = {
            'document_id': document_id,
            'processing_metadata': {
                'processed_at': datetime.now(),
                'processor_version': '1.0',
                'total_segments': len(segments),
                'has_search_data': raw_content['metadata']['has_search_data'],
                'has_pdf_data': raw_content['metadata']['has_pdf_data'],
                'language': 'en'  # Assuming English for now
            },
            'original_stats': {
                'raw_character_count': len(raw_content.get('combined_text', '')),
                'search_sources': len(raw_content.get('search_content', [])),
                'has_pdf': bool(raw_content.get('pdf_content'))
            },
            'processed_segments': segments,
            'quality_metrics': {
                'avg_quality_score': sum(seg.get('quality_score', 0) for seg in segments) / max(len(segments), 1),
                'avg_word_count': sum(seg['word_count'] for seg in segments) / max(len(segments), 1),
                'total_tokens': sum(seg.get('token_count', 0) for seg in segments)
            }
        }
        
        # Store in cleaned_data collection
        try:
            collection = self.storage.db['cleaned_data']
            
            # Upsert the document
            collection.replace_one(
                {'document_id': document_id},
                cleaned_document,
                upsert=True
            )
            
            logger.info(f"Successfully stored processed data for document: {document_id}")
            return {'stored': True, 'collection': 'cleaned_data'}
            
        except Exception as e:
            logger.error(f"Failed to store processed data: {e}")
            return {'stored': False, 'error': str(e)}
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset processing statistics"""
        self.stats = {
            'documents_processed': 0,
            'segments_created': 0,
            'duplicates_removed': 0,
            'quality_filtered': 0,
            'total_processing_time': 0.0
        } 