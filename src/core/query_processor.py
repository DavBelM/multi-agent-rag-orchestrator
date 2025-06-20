"""
Query Processor

Advanced query processing and understanding for the multi-agent RAG system.
"""

import asyncio
from typing import Dict, Any, List, Optional
import re
import logging

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Advanced query processing and analysis"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process and analyze a query"""
        return {
            "original_query": query,
            "processed_query": query.strip(),
            "query_type": await self._classify_query(query),
            "keywords": await self._extract_keywords(query),
            "intent": await self._extract_intent(query)
        }
    
    async def _classify_query(self, query: str) -> str:
        """Classify the type of query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what is', 'define', 'definition']):
            return 'definition'
        elif any(word in query_lower for word in ['how to', 'how do', 'steps']):
            return 'how_to'
        elif any(word in query_lower for word in ['why', 'reason', 'because']):
            return 'explanation'
        elif any(word in query_lower for word in ['compare', 'difference', 'versus']):
            return 'comparison'
        else:
            return 'general'
    
    async def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query"""
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', query.lower())
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords
    
    async def _extract_intent(self, query: str) -> str:
        """Extract user intent from query"""
        query_lower = query.lower()
        
        if '?' in query:
            return 'question'
        elif any(word in query_lower for word in ['find', 'search', 'look for']):
            return 'search'
        elif any(word in query_lower for word in ['analyze', 'analysis', 'study']):
            return 'analysis'
        else:
            return 'general'
