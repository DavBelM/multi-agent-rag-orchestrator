"""
Document Loader

Advanced document loading and processing for the multi-agent RAG system.
"""

import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
import mimetypes
import logging

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Document loader for various file formats"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.supported_formats = ['.txt', '.md', '.pdf', '.docx', '.json']
    
    async def load_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Load documents from file paths"""
        documents = []
        
        for file_path in file_paths:
            try:
                doc = await self.load_document(file_path)
                if doc:
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
        
        return documents
    
    async def load_document(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load a single document"""
        path = Path(file_path)
        
        if not path.exists():
            logger.warning(f"File not found: {file_path}")
            return None
        
        content = ""
        metadata = {
            "filename": path.name,
            "file_path": str(path),
            "file_size": path.stat().st_size,
            "file_type": path.suffix.lower()
        }
        
        # Read content based on file type
        if path.suffix.lower() in ['.txt', '.md']:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            content = f"Unsupported file type: {path.suffix}"
        
        return {
            "content": content,
            "metadata": metadata,
            "source": str(path)
        }
