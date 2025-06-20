"""
Research Agent

Specialized agent for information gathering and retrieval from various sources.
Handles document search, web scraping, and data collection tasks.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import time

from .base_agent import BaseAgent, AgentMessage
from ..utils.chunking import DocumentChunker
from ..core.vector_store import VectorStore
from ..core.embedding_manager import EmbeddingManager


class ResearchAgent(BaseAgent):
    """
    Research Agent specialized in information gathering and retrieval.
    
    Capabilities:
    - Document search and retrieval
    - Semantic similarity search
    - Multi-source information gathering
    - Query expansion and refinement
    - Source credibility assessment
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_manager: EmbeddingManager,
        config: Optional[Dict[str, Any]] = None
    ):
        capabilities = [
            "document_search",
            "semantic_retrieval",
            "query_expansion",
            "source_assessment",
            "information_synthesis"
        ]
        
        super().__init__(
            agent_id="research_agent",
            name="Research Agent",
            description="Specialized agent for information gathering and retrieval",
            capabilities=capabilities,
            config=config
        )
        
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.chunker = DocumentChunker()
        
        # Research-specific configuration
        self.max_documents = config.get("max_documents", 20) if config else 20
        self.similarity_threshold = config.get("similarity_threshold", 0.7) if config else 0.7
        self.max_chunk_size = config.get("max_chunk_size", 1000) if config else 1000
    
    def process_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Process incoming messages"""
        message_type = message.message_type
        content = message.content
        
        if message_type == "research_request":
            return {
                "status": "acknowledged",
                "task_id": content.get("task_id"),
                "estimated_time": "30-60 seconds"
            }
        elif message_type == "status_query":
            return self.get_metrics()
        else:
            return {"status": "unknown_message_type", "message_type": message_type}
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a research task"""
        start_time = time.time()
        task_type = task.get("type", "general_research")
        
        try:
            self.update_state("working", task_type, 0.1)
            
            if task_type == "document_search":
                result = await self._document_search(task)
            elif task_type == "semantic_search":
                result = await self._semantic_search(task)
            elif task_type == "comprehensive_research":
                result = await self._comprehensive_research(task)
            else:
                result = await self._general_research(task)
            
            self.update_state("completed", task_type, 1.0)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.update_metrics(processing_time, True)
            
            return {
                "status": "success",
                "result": result,
                "processing_time": processing_time,
                "agent_id": self.agent_id
            }
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {str(e)}")
            self.update_state("error", task_type, 0.0)
            
            processing_time = time.time() - start_time
            self.update_metrics(processing_time, False)
            
            return {
                "status": "error",
                "error": str(e),
                "processing_time": processing_time,
                "agent_id": self.agent_id
            }
    
    async def _document_search(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform document search based on keywords or phrases"""
        query = task.get("query", "")
        filters = task.get("filters", {})
        
        self.update_state("working", "document_search", 0.3)
        
        # Perform vector search
        results = await self.vector_store.similarity_search(
            query=query,
            limit=self.max_documents,
            threshold=self.similarity_threshold,
            filters=filters
        )
        
        self.update_state("working", "document_search", 0.7)
        
        # Process and rank results
        processed_results = []
        for result in results:
            processed_results.append({
                "content": result.get("content", ""),
                "metadata": result.get("metadata", {}),
                "score": result.get("score", 0.0),
                "source": result.get("source", "unknown")
            })
        
        return {
            "search_type": "document_search",
            "query": query,
            "results": processed_results,
            "total_found": len(processed_results),
            "search_metadata": {
                "threshold": self.similarity_threshold,
                "max_documents": self.max_documents
            }
        }
    
    async def _semantic_search(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform semantic search with query expansion"""
        query = task.get("query", "")
        expand_query = task.get("expand_query", True)
        
        self.update_state("working", "semantic_search", 0.2)
        
        # Expand query if requested
        expanded_queries = [query]
        if expand_query:
            expanded_queries.extend(await self._expand_query(query))
        
        self.update_state("working", "semantic_search", 0.5)
        
        # Perform search for each expanded query
        all_results = []
        for exp_query in expanded_queries:
            results = await self.vector_store.similarity_search(
                query=exp_query,
                limit=self.max_documents // len(expanded_queries),
                threshold=self.similarity_threshold
            )
            all_results.extend(results)
        
        self.update_state("working", "semantic_search", 0.8)
        
        # Deduplicate and rank results
        unique_results = self._deduplicate_results(all_results)
        ranked_results = self._rank_results(unique_results, query)
        
        return {
            "search_type": "semantic_search",
            "original_query": query,
            "expanded_queries": expanded_queries,
            "results": ranked_results[:self.max_documents],
            "total_found": len(ranked_results),
            "deduplication_stats": {
                "original_count": len(all_results),
                "deduplicated_count": len(unique_results)
            }
        }
    
    async def _comprehensive_research(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive research combining multiple strategies"""
        query = task.get("query", "")
        research_depth = task.get("depth", "medium")  # shallow, medium, deep
        
        self.update_state("working", "comprehensive_research", 0.1)
        
        # Strategy 1: Direct semantic search
        semantic_results = await self._semantic_search({
            "query": query,
            "expand_query": True
        })
        
        self.update_state("working", "comprehensive_research", 0.4)
        
        # Strategy 2: Keyword-based search
        keyword_results = await self._document_search({
            "query": query
        })
        
        self.update_state("working", "comprehensive_research", 0.7)
        
        # Strategy 3: Related concept search (if deep research)
        related_results = []
        if research_depth == "deep":
            related_concepts = await self._extract_related_concepts(query)
            for concept in related_concepts:
                concept_results = await self.vector_store.similarity_search(
                    query=concept,
                    limit=5,
                    threshold=self.similarity_threshold * 0.8
                )
                related_results.extend(concept_results)
        
        self.update_state("working", "comprehensive_research", 0.9)
        
        # Combine and synthesize results
        all_results = (
            semantic_results["results"] +
            keyword_results["results"] +
            related_results
        )
        
        # Deduplicate and rank
        unique_results = self._deduplicate_results(all_results)
        final_results = self._rank_results(unique_results, query)
        
        # Generate research summary
        summary = await self._generate_research_summary(final_results, query)
        
        return {
            "search_type": "comprehensive_research",
            "query": query,
            "research_depth": research_depth,
            "results": final_results[:self.max_documents],
            "summary": summary,
            "research_strategies": {
                "semantic_search": len(semantic_results["results"]),
                "keyword_search": len(keyword_results["results"]),
                "related_concepts": len(related_results)
            },
            "total_found": len(final_results)
        }
    
    async def _general_research(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """General research fallback method"""
        return await self._semantic_search(task)
    
    async def _expand_query(self, query: str) -> List[str]:
        """Expand query with related terms and synonyms"""
        # Simple query expansion logic
        # In a real implementation, you might use NLP models or thesauri
        expanded = []
        
        # Add variations
        words = query.split()
        if len(words) > 1:
            # Add individual words as separate queries
            expanded.extend(words)
            
            # Add combinations
            for i in range(len(words) - 1):
                expanded.append(" ".join(words[i:i+2]))
        
        return expanded[:3]  # Limit expansion
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on content similarity"""
        unique_results = []
        seen_content = set()
        
        for result in results:
            content = result.get("content", "")
            # Simple deduplication based on content hash
            content_hash = hash(content[:200])  # Use first 200 chars for hash
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
        
        return unique_results
    
    def _rank_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Rank results based on relevance to query"""
        # Simple ranking based on score and content length
        for result in results:
            base_score = result.get("score", 0.0)
            content_length = len(result.get("content", ""))
            
            # Boost score based on content length (within reason)
            length_boost = min(content_length / 1000, 0.2)
            result["final_score"] = base_score + length_boost
        
        # Sort by final score
        return sorted(results, key=lambda x: x.get("final_score", 0.0), reverse=True)
    
    async def _extract_related_concepts(self, query: str) -> List[str]:
        """Extract related concepts for deeper research"""
        # Simple concept extraction
        # In practice, you'd use NER, topic modeling, or knowledge graphs
        concepts = []
        
        words = query.lower().split()
        
        # Add conceptual variations
        for word in words:
            if len(word) > 4:  # Only for meaningful words
                concepts.append(f"{word} definition")
                concepts.append(f"{word} examples")
                concepts.append(f"{word} applications")
        
        return concepts[:5]  # Limit concepts
    
    async def _generate_research_summary(self, results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Generate a summary of research findings"""
        if not results:
            return {
                "overview": "No relevant information found",
                "key_findings": [],
                "source_count": 0
            }
        
        # Extract key information
        sources = set()
        total_content_length = 0
        score_distribution = []
        
        for result in results:
            if "source" in result.get("metadata", {}):
                sources.add(result["metadata"]["source"])
            total_content_length += len(result.get("content", ""))
            score_distribution.append(result.get("score", 0.0))
        
        avg_score = sum(score_distribution) / len(score_distribution) if score_distribution else 0.0
        
        return {
            "overview": f"Found {len(results)} relevant documents for query: '{query}'",
            "key_findings": [
                f"Total content analyzed: {total_content_length:,} characters",
                f"Average relevance score: {avg_score:.3f}",
                f"Unique sources: {len(sources)}"
            ],
            "source_count": len(sources),
            "result_count": len(results),
            "quality_metrics": {
                "average_score": avg_score,
                "score_range": [min(score_distribution), max(score_distribution)] if score_distribution else [0, 0],
                "content_coverage": total_content_length
            }
        }
    
    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a query and return research results."""
        task = {
            "type": "general_research",
            "query": query,
            "context": context or {},
            "timestamp": datetime.now().isoformat()
        }
        return await self.execute_task(task)
