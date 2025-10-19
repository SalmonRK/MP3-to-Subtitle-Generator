#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text Segmentation Utility for SRT Generation
Handles intelligent text segmentation for subtitle generation
"""

import re
from typing import List, Dict, Tuple
import math


class TextSegmenter:
    """
    Intelligent text segmenter for subtitle generation
    """
    
    def __init__(self, max_chars_per_segment: int = 42, max_duration: float = 7.0):
        """
        Initialize text segmenter
        
        Args:
            max_chars_per_segment: Maximum characters per subtitle segment
            max_duration: Maximum duration in seconds per segment
        """
        self.max_chars = max_chars_per_segment
        self.max_duration = max_duration
        
        # Thai sentence ending patterns
        self.thai_sentence_endings = r'[.!?。!?]+'
        self.thai_clause_separators = r'[,;:，；:]+'
        
        # Common Thai particles that indicate natural breaks
        self.thai_particles = [
            'ครับ', 'ค่ะ', 'คะ', 'นะ', 'น่ะ', 'น่า', 'เถอะ', 'เถิด', 
            'เลย', 'มั้ย', 'ไหม', 'ไหม', 'เรอ', 'เร็ว', 'ซิ', 'ซี', 'อ่ะ',
            'อะ', 'อ่า', 'อา', 'ง่ะ', 'งับ', 'เอง', 'จริง', 'จ้า', 'จ๊ะ'
        ]
        
        # Build regex pattern for Thai particles
        self.particle_pattern = r'(' + '|'.join(self.thai_particles) + r')'
    
    def segment_transcription(self, transcription_result: Dict) -> List[Dict]:
        """
        Segment transcription result into subtitle-friendly segments
        
        Args:
            transcription_result: Result from ASR model with segments and word timestamps
            
        Returns:
            List of segmented subtitle entries
        """
        segments = transcription_result.get("segments", [])
        if not segments:
            return []
        
        # If we have word-level timestamps, use them for better segmentation
        if segments and segments[0].get("words"):
            return self._segment_with_word_timestamps(segments)
        else:
            return self._segment_with_segment_timestamps(segments)
    
    def _segment_with_word_timestamps(self, segments: List[Dict]) -> List[Dict]:
        """
        Create segments using word-level timestamps for precise timing
        
        Args:
            segments: List of segments with word timestamps
            
        Returns:
            List of improved subtitle segments
        """
        improved_segments = []
        
        for segment in segments:
            words = segment.get("words", [])
            if not words:
                # Fallback to segment-level timing
                improved_segments.append({
                    "start": segment.get("start", 0),
                    "end": segment.get("end", 0),
                    "text": segment.get("text", "").strip()
                })
                continue
            
            # Group words into logical subtitle segments
            word_groups = self._group_words(words)
            
            for group in word_groups:
                if len(group) > 0:
                    start_time = group[0].get("start", 0)
                    end_time = group[-1].get("end", 0)
                    text = "".join([word.get("word", "") for word in group])
                    
                    # Clean up text
                    text = text.strip()
                    
                    if text:
                        improved_segments.append({
                            "start": start_time,
                            "end": end_time,
                            "text": text
                        })
        
        return improved_segments
    
    def _segment_with_segment_timestamps(self, segments: List[Dict]) -> List[Dict]:
        """
        Create segments using segment-level timestamps
        
        Args:
            segments: List of segments with start/end times
            
        Returns:
            List of improved subtitle segments
        """
        improved_segments = []
        
        for segment in segments:
            text = segment.get("text", "").strip()
            if not text:
                continue
            
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            duration = end_time - start_time
            
            # If segment is too long or has too many characters, split it
            if duration > self.max_duration or len(text) > self.max_chars:
                sub_segments = self._split_text_segment(text, start_time, end_time)
                improved_segments.extend(sub_segments)
            else:
                improved_segments.append({
                    "start": start_time,
                    "end": end_time,
                    "text": text
                })
        
        return improved_segments
    
    def _group_words(self, words: List[Dict]) -> List[List[Dict]]:
        """
        Group words into logical subtitle segments
        
        Args:
            words: List of word dictionaries with timestamps
            
        Returns:
            List of word groups
        """
        groups = []
        current_group = []
        current_chars = 0
        current_start = None
        current_end = None
        
        for word in words:
            word_text = word.get("word", "")
            word_start = word.get("start", 0)
            word_end = word.get("end", 0)
            
            # Initialize timing for first word
            if current_start is None:
                current_start = word_start
            
            current_end = word_end
            
            # Check if adding this word would exceed limits
            would_exceed_chars = current_chars + len(word_text) > self.max_chars
            would_exceed_duration = (current_end - current_start) > self.max_duration
            
            # Check for natural break points
            has_natural_break = self._has_natural_break(word_text)
            
            # Start new group if needed
            if (current_group and (would_exceed_chars or would_exceed_duration or has_natural_break)):
                if current_group:
                    groups.append(current_group)
                current_group = [word]
                current_chars = len(word_text)
                current_start = word_start
                current_end = word_end
            else:
                current_group.append(word)
                current_chars += len(word_text)
        
        # Add the last group
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _has_natural_break(self, word_text: str) -> bool:
        """
        Check if word contains natural break indicators
        
        Args:
            word_text: The word text to check
            
        Returns:
            True if word indicates a natural break point
        """
        # Check for sentence endings
        if re.search(self.thai_sentence_endings, word_text):
            return True
        
        # Check for Thai particles
        if re.search(self.particle_pattern, word_text):
            return True
        
        return False
    
    def _split_text_segment(self, text: str, start_time: float, end_time: float) -> List[Dict]:
        """
        Split a long text segment into smaller, subtitle-friendly pieces
        
        Args:
            text: The text to split
            start_time: Start time of the segment
            end_time: End time of the segment
            
        Returns:
            List of split segments
        """
        duration = end_time - start_time
        
        # Try to split at natural break points first
        sentences = re.split(f'({self.thai_sentence_endings})', text)
        
        # Rejoin sentence endings with their sentences
        text_pieces = []
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                text_pieces.append(sentences[i] + sentences[i + 1])
            else:
                text_pieces.append(sentences[i])
        
        # Filter out empty pieces
        text_pieces = [piece.strip() for piece in text_pieces if piece.strip()]
        
        # If we have multiple pieces, distribute time proportionally
        if len(text_pieces) > 1:
            segments = []
            total_chars = sum(len(piece) for piece in text_pieces)
            current_time = start_time
            
            for i, piece in enumerate(text_pieces):
                # Calculate proportional duration
                if i == len(text_pieces) - 1:
                    # Last piece gets remaining time
                    piece_end = end_time
                else:
                    piece_duration = (len(piece) / total_chars) * duration
                    piece_end = current_time + piece_duration
                
                segments.append({
                    "start": current_time,
                    "end": piece_end,
                    "text": piece
                })
                
                current_time = piece_end
            
            return segments
        
        # If no natural breaks, try to split at clause separators
        clauses = re.split(f'({self.thai_clause_separators})', text)
        
        # Rejoin clause separators with their clauses
        text_pieces = []
        for i in range(0, len(clauses), 2):
            if i + 1 < len(clauses):
                text_pieces.append(clauses[i] + clauses[i + 1])
            else:
                text_pieces.append(clauses[i])
        
        # Filter out empty pieces
        text_pieces = [piece.strip() for piece in text_pieces if piece.strip()]
        
        # If we have multiple clauses, distribute time proportionally
        if len(text_pieces) > 1:
            segments = []
            total_chars = sum(len(piece) for piece in text_pieces)
            current_time = start_time
            
            for i, piece in enumerate(text_pieces):
                # Calculate proportional duration
                if i == len(text_pieces) - 1:
                    # Last piece gets remaining time
                    piece_end = end_time
                else:
                    piece_duration = (len(piece) / total_chars) * duration
                    piece_end = current_time + piece_duration
                
                segments.append({
                    "start": current_time,
                    "end": piece_end,
                    "text": piece
                })
                
                current_time = piece_end
            
            return segments
        
        # If still too long, split by character count
        if len(text) > self.max_chars:
            return self._split_by_chars(text, start_time, end_time)
        
        # If duration is too long but text is short, split time evenly
        if duration > self.max_duration:
            num_splits = math.ceil(duration / self.max_duration)
            split_duration = duration / num_splits
            
            segments = []
            for i in range(num_splits):
                segment_start = start_time + (i * split_duration)
                segment_end = min(segment_start + split_duration, end_time)
                
                segments.append({
                    "start": segment_start,
                    "end": segment_end,
                    "text": text
                })
            
            return segments
        
        # Default: return original segment
        return [{
            "start": start_time,
            "end": end_time,
            "text": text
        }]
    
    def _split_by_chars(self, text: str, start_time: float, end_time: float) -> List[Dict]:
        """
        Split text by character count when no natural breaks are found
        
        Args:
            text: The text to split
            start_time: Start time of the segment
            end_time: End time of the segment
            
        Returns:
            List of split segments
        """
        duration = end_time - start_time
        segments = []
        
        # Calculate number of splits needed
        num_splits = math.ceil(len(text) / self.max_chars)
        split_duration = duration / num_splits
        
        for i in range(num_splits):
            start_idx = i * self.max_chars
            end_idx = min((i + 1) * self.max_chars, len(text))
            
            segment_text = text[start_idx:end_idx]
            segment_start = start_time + (i * split_duration)
            segment_end = min(segment_start + split_duration, end_time)
            
            segments.append({
                "start": segment_start,
                "end": segment_end,
                "text": segment_text
            })
        
        return segments