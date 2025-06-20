"""
Document Chunking Utilities

Advanced document chunking strategies for optimal RAG performance.
"""

from typing import List, Dict, Any, Optional
import re
import logging

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Advanced document chunking with multiple strategies"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.chunk_overlap = self.config.get("chunk_overlap", 200)
        self.strategy = self.config.get("strategy", "sliding_window")
    
    def chunk_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Chunk a document into smaller pieces"""
        if self.strategy == "sliding_window":
            return self._sliding_window_chunking(content, metadata)
        elif self.strategy == "semantic":
            return self._semantic_chunking(content, metadata)
        elif self.strategy == "paragraph":
            return self._paragraph_chunking(content, metadata)
        else:
            return self._simple_chunking(content, metadata)
    
    def _sliding_window_chunking(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Sliding window chunking with overlap"""
        chunks = []
        
        if len(content) <= self.chunk_size:
            return [{
                "content": content,
                "metadata": metadata or {},
                "chunk_index": 0,
                "chunk_type": "single"
            }]
        
        start = 0
        chunk_index = 0
        
        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            chunk_content = content[start:end]
            
            # Try to break at sentence boundary
            if end < len(content):
                last_period = chunk_content.rfind('. ')
                if last_period > self.chunk_size // 2:
                    end = start + last_period + 1
                    chunk_content = content[start:end]
            
            chunks.append({
                "content": chunk_content.strip(),
                "metadata": metadata or {},
                "chunk_index": chunk_index,
                "chunk_type": "sliding_window",
                "start_position": start,
                "end_position": end
            })
            
            start = end - self.chunk_overlap
            chunk_index += 1
        
        return chunks
    
    def _semantic_chunking(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Semantic-aware chunking"""
        # Simple implementation - break by paragraphs and combine up to chunk size
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= self.chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append({
                        "content": current_chunk.strip(),
                        "metadata": metadata or {},
                        "chunk_index": chunk_index,
                        "chunk_type": "semantic"
                    })
                    chunk_index += 1
                
                current_chunk = paragraph + "\n\n"
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                "content": current_chunk.strip(),
                "metadata": metadata or {},
                "chunk_index": chunk_index,
                "chunk_type": "semantic"
            })
        
        return chunks
    
    def _paragraph_chunking(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Paragraph-based chunking"""
        paragraphs = content.split('\n\n')
        chunks = []
        
        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                chunks.append({
                    "content": paragraph.strip(),
                    "metadata": metadata or {},
                    "chunk_index": i,
                    "chunk_type": "paragraph"
                })
        
        return chunks
    
    def _simple_chunking(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Simple fixed-size chunking"""
        chunks = []
        chunk_index = 0
        
        for i in range(0, len(content), self.chunk_size):
            chunk_content = content[i:i + self.chunk_size]
            chunks.append({
                "content": chunk_content,
                "metadata": metadata or {},
                "chunk_index": chunk_index,
                "chunk_type": "simple"
            })
            chunk_index += 1
        
        return chunks
