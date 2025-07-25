#!/usr/bin/env python3
"""
Q&A Dataset Generation using Groq API with Qwen model
Converts cleaned text data into question-answer pairs for fine-tuning
"""

import json
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from pathlib import Path
import asyncio
import time

from groq import Groq
from ..data_processing.storage import MongoStorage

# Configure logging
logger = logging.getLogger(__name__)

class QAGenerator:
    """Generate Q&A pairs from cleaned text segments using Groq Qwen model"""
    
    def __init__(self):
        """Initialize the Q&A generator"""
        self.storage = MongoStorage()
        self.groq_client = None
        self.data_folder = Path("data")
        self.data_folder.mkdir(exist_ok=True)
        
        # Initialize Groq client
        self._init_groq_client()
        
        # Q&A generation statistics
        self.stats = {
            'documents_processed': 0,
            'segments_processed': 0,
            'qa_pairs_generated': 0,
            'failed_generations': 0,
            'total_processing_time': 0.0
        }
        
        logger.info("QAGenerator initialized successfully")
    
    def _init_groq_client(self):
        """Initialize Groq API client"""
        try:
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                logger.error("GROQ_API_KEY not found in environment variables")
                raise ValueError("GROQ_API_KEY is required")
            
            self.groq_client = Groq(api_key=api_key)
            logger.info("Groq client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            raise
    
    def generate_qa_dataset(self, document_id: str, max_qa_per_segment: int = 3, max_questions: int = None) -> Dict[str, Any]:
        """
        Generate Q&A dataset from cleaned document
        
        Args:
            document_id: ID of document in cleaned_data collection
            max_qa_per_segment: Maximum Q&A pairs to generate per text segment
            max_questions: Maximum total questions to generate (None for no limit)
            
        Returns:
            Dictionary with generation results and statistics
        """
        start_time = datetime.now()
        logger.info(f"Starting Q&A generation for document: {document_id}")
        
        try:
            # Get cleaned data
            cleaned_data = self._get_cleaned_data(document_id)
            if not cleaned_data:
                return {"status": "error", "message": "Cleaned data not found"}
            
            # Generate Q&A pairs from segments
            qa_pairs = self._process_segments(cleaned_data['processed_segments'], max_qa_per_segment, max_questions)
            
            # Create dataset structure
            qa_dataset = self._create_dataset_structure(document_id, cleaned_data, qa_pairs)
            
            # Save to database
            self._save_to_database(document_id, qa_dataset)
            
            # Save to text files
            self._save_to_files(document_id, qa_dataset)
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats['documents_processed'] += 1
            self.stats['total_processing_time'] += processing_time
            
            result = {
                "status": "success",
                "document_id": document_id,
                "total_qa_pairs": len(qa_pairs),
                "segments_processed": len(cleaned_data['processed_segments']),
                "processing_time": processing_time,
                "files_created": [
                    f"data/{document_id}_qa_dataset.json",
                    f"data/{document_id}_qa_pairs.txt"
                ],
                "qa_pairs": qa_pairs  # Include actual Q&A pairs in response
            }
            
            logger.info(f"Q&A generation completed for {document_id}: {len(qa_pairs)} pairs in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate Q&A dataset for {document_id}: {str(e)}")
            return {
                "status": "error", 
                "message": str(e),
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
    
    def _get_cleaned_data(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get cleaned data from database"""
        try:
            collection = self.storage.db['cleaned_data']
            document = collection.find_one({"document_id": document_id})
            return document
        except Exception as e:
            logger.error(f"Failed to get cleaned data for {document_id}: {e}")
            return None
    
    def _process_segments(self, segments: List[Dict[str, Any]], max_qa_per_segment: int, max_questions: int = None) -> List[Dict[str, Any]]:
        """Process text segments to generate Q&A pairs"""
        all_qa_pairs = []
        
        for i, segment in enumerate(segments):
            try:
                # Check if we've reached the maximum questions limit
                if max_questions and len(all_qa_pairs) >= max_questions:
                    logger.info(f"Reached maximum questions limit ({max_questions}), stopping generation")
                    break
                
                logger.info(f"Processing segment {i+1}/{len(segments)}")
                
                # Calculate how many questions to generate for this segment
                remaining_questions = max_questions - len(all_qa_pairs) if max_questions else max_qa_per_segment
                questions_for_segment = min(max_qa_per_segment, remaining_questions) if max_questions else max_qa_per_segment
                
                if questions_for_segment <= 0:
                    break
                
                # Generate Q&A pairs for this segment
                qa_pairs = self._generate_qa_from_segment(segment, questions_for_segment)
                
                # Add segment metadata to each Q&A pair
                for qa in qa_pairs:
                    qa.update({
                        'segment_index': i,
                        'source_segment': segment.get('text', ''),
                        'word_count': segment.get('word_count', 0),
                        'quality_score': segment.get('quality_score', 0.0)
                    })
                
                all_qa_pairs.extend(qa_pairs)
                self.stats['segments_processed'] += 1
                
                # Rate limiting to respect API limits
                time.sleep(1)
                
            except Exception as e:
                logger.warning(f"Failed to process segment {i}: {e}")
                self.stats['failed_generations'] += 1
                continue
        
        return all_qa_pairs
    
    def _generate_qa_from_segment(self, segment: Dict[str, Any], max_qa: int) -> List[Dict[str, Any]]:
        """Generate Q&A pairs from a single text segment using Groq"""
        text = segment.get('normalized_text', segment.get('text', ''))
        
        if len(text.strip()) < 50:  # Skip very short segments
            return []
        
        # Create prompt for Q&A generation
        prompt = self._create_qa_prompt(text, max_qa)
        
        try:
            # Call Groq API with currently supported model
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Using currently supported model
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at creating high-quality question-answer pairs for machine learning training. Generate clear, specific questions with accurate, comprehensive answers based on the provided text."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=1500,
                top_p=0.9
            )
            
            # Parse the response
            qa_pairs = self._parse_qa_response(response.choices[0].message.content)
            self.stats['qa_pairs_generated'] += len(qa_pairs)
            
            return qa_pairs
            
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            self.stats['failed_generations'] += 1
            return []
    
    def _create_qa_prompt(self, text: str, max_qa: int) -> str:
        """Create prompt for Q&A generation"""
        prompt = f"""
Based on the following text, generate {max_qa} high-quality question-answer pairs suitable for fine-tuning a language model.

Requirements:
1. Questions should be clear, specific, and answerable from the text
2. Answers should be comprehensive but concise
3. Cover different aspects of the content (facts, concepts, relationships)
4. Vary question types (what, how, why, when, where)
5. Ensure answers are factually accurate based on the text

Text:
{text}

Please format your response as JSON with this structure:
[
  {{
    "question": "Your question here?",
    "answer": "Your comprehensive answer here.",
    "question_type": "factual|conceptual|analytical",
    "difficulty": "easy|medium|hard"
  }}
]

Generate exactly {max_qa} question-answer pairs:
"""
        return prompt
    
    def _parse_qa_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse the Q&A response from Groq"""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                qa_data = json.loads(json_str)
                
                # Validate and clean the data
                valid_qa_pairs = []
                for item in qa_data:
                    if isinstance(item, dict) and 'question' in item and 'answer' in item:
                        qa_pair = {
                            'question': item['question'].strip(),
                            'answer': item['answer'].strip(),
                            'question_type': item.get('question_type', 'factual'),
                            'difficulty': item.get('difficulty', 'medium'),
                            'generated_at': datetime.now().isoformat()
                        }
                        valid_qa_pairs.append(qa_pair)
                
                return valid_qa_pairs
            
            # Fallback: try to parse manually if JSON parsing fails
            return self._manual_parse_qa(response)
            
        except Exception as e:
            logger.warning(f"Failed to parse Q&A response: {e}")
            return self._manual_parse_qa(response)
    
    def _manual_parse_qa(self, response: str) -> List[Dict[str, Any]]:
        """Manual parsing fallback for Q&A responses"""
        qa_pairs = []
        
        # Look for question-answer patterns
        lines = response.split('\n')
        current_q = None
        current_a = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for question patterns
            if line.startswith(('Q:', 'Question:', '**Q:', '1.', '2.', '3.')):
                if current_q and current_a:
                    qa_pairs.append({
                        'question': current_q,
                        'answer': current_a,
                        'question_type': 'factual',
                        'difficulty': 'medium',
                        'generated_at': datetime.now().isoformat()
                    })
                current_q = re.sub(r'^(Q:|Question:|\*\*Q:|\d+\.)\s*', '', line)
                current_a = None
                
            # Look for answer patterns
            elif line.startswith(('A:', 'Answer:', '**A:')):
                current_a = re.sub(r'^(A:|Answer:|\*\*A:)\s*', '', line)
                
            # Continue building answer
            elif current_q and current_a is not None:
                current_a += ' ' + line
        
        # Add the last pair
        if current_q and current_a:
            qa_pairs.append({
                'question': current_q,
                'answer': current_a,
                'question_type': 'factual',
                'difficulty': 'medium',
                'generated_at': datetime.now().isoformat()
            })
        
        return qa_pairs
    
    def _create_dataset_structure(self, document_id: str, cleaned_data: Dict[str, Any], qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create structured dataset for storage"""
        return {
            'document_id': document_id,
            'generated_at': datetime.now().isoformat(),
            'source_info': {
                'original_segments': len(cleaned_data['processed_segments']),
                'processing_metadata': cleaned_data.get('processing_metadata', {}),
                'quality_metrics': cleaned_data.get('quality_metrics', {})
            },
            'qa_dataset': {
                'total_pairs': len(qa_pairs),
                'pairs': qa_pairs
            },
            'generation_stats': {
                'segments_processed': self.stats['segments_processed'],
                'qa_pairs_generated': len(qa_pairs),
                'avg_pairs_per_segment': len(qa_pairs) / max(len(cleaned_data['processed_segments']), 1)
            }
        }
    
    def _save_to_database(self, document_id: str, qa_dataset: Dict[str, Any]):
        """Save Q&A dataset to MongoDB"""
        try:
            collection = self.storage.db['qa_datasets']
            
            # Upsert the dataset
            collection.replace_one(
                {'document_id': document_id},
                qa_dataset,
                upsert=True
            )
            
            logger.info(f"Saved Q&A dataset to database for document: {document_id}")
            
        except Exception as e:
            logger.error(f"Failed to save Q&A dataset to database: {e}")
            raise
    
    def _save_to_files(self, document_id: str, qa_dataset: Dict[str, Any]):
        """Save Q&A dataset to text files"""
        try:
            # Create a JSON-serializable copy of the dataset
            serializable_dataset = self._make_json_serializable(qa_dataset)
            
            # Save as JSON file
            json_file = self.data_folder / f"{document_id}_qa_dataset.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_dataset, f, indent=2, ensure_ascii=False)
            
            # Save as plain text file for easy reading
            txt_file = self.data_folder / f"{document_id}_qa_pairs.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(f"Q&A Dataset for Document: {document_id}\n")
                f.write(f"Generated: {serializable_dataset['generated_at']}\n")
                f.write(f"Total Q&A Pairs: {serializable_dataset['qa_dataset']['total_pairs']}\n")
                f.write("=" * 80 + "\n\n")
                
                for i, qa in enumerate(serializable_dataset['qa_dataset']['pairs'], 1):
                    f.write(f"Pair {i}:\n")
                    f.write(f"Question: {qa['question']}\n")
                    f.write(f"Answer: {qa['answer']}\n")
                    f.write(f"Type: {qa.get('question_type', 'N/A')}\n")
                    f.write(f"Difficulty: {qa.get('difficulty', 'N/A')}\n")
                    f.write("-" * 40 + "\n\n")
            
            # Save in training format (JSONL)
            jsonl_file = self.data_folder / f"{document_id}_training.jsonl"
            with open(jsonl_file, 'w', encoding='utf-8') as f:
                for qa in serializable_dataset['qa_dataset']['pairs']:
                    training_example = {
                        "messages": [
                            {"role": "user", "content": qa['question']},
                            {"role": "assistant", "content": qa['answer']}
                        ]
                    }
                    f.write(json.dumps(training_example, ensure_ascii=False) + '\n')
            
            logger.info(f"Saved Q&A files: JSON, TXT, and JSONL for document: {document_id}")
            
        except Exception as e:
            logger.error(f"Failed to save Q&A files: {e}")
            raise
    
    def _make_json_serializable(self, obj):
        """Convert datetime objects to ISO format strings for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get Q&A generation statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset generation statistics"""
        self.stats = {
            'documents_processed': 0,
            'segments_processed': 0,
            'qa_pairs_generated': 0,
            'failed_generations': 0,
            'total_processing_time': 0.0
        } 