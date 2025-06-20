"""
Multi-Agent RAG Orchestrator

A sophisticated multi-agent system for advanced RAG operations.
Coordinates specialized agents for research, analysis, writing, and validation.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
import uuid

from .core.vector_store import VectorStore
from .core.embedding_manager import EmbeddingManager
from .core.communication_layer import MessageBus, AgentMessage, MessageType
from .core.state_manager import StateManager
from .utils.monitoring import MetricsCollector, PerformanceTracker
from .config import OrchestratorConfig
from .agents.research_agent import ResearchAgent
from .agents.analysis_agent import AnalysisAgent
from .agents.writing_agent import WritingAgent
from .agents.coordinator_agent import CoordinatorAgent
from .agents.validation_agent import ValidationAgent

logger = logging.getLogger(__name__)


@dataclass
class QueryRequest:
    """Request structure for multi-agent query processing."""
    query: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    preferences: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.session_id is None:
            self.session_id = str(uuid.uuid4())
        if self.context is None:
            self.context = {}
        if self.preferences is None:
            self.preferences = {}


@dataclass
class QueryResponse:
    """Response structure for multi-agent query processing."""
    query: str
    response: str
    session_id: str
    sources: List[str]
    confidence: float
    processing_time: float
    agents_involved: List[str]
    metadata: Dict[str, Any]


class MultiAgentRAGOrchestrator:
    """
    Multi-Agent RAG Orchestrator
    
    Coordinates specialized agents to handle complex queries through:
    - Research and information gathering
    - Analysis and synthesis
    - Response generation and writing
    - Validation and quality assurance
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """Initialize the orchestrator."""
        self.config = config or OrchestratorConfig()
        
        # Core components
        self.vector_store = VectorStore(config=self.config.vector_store.__dict__)
        self.embedding_manager = EmbeddingManager(config=self.config.embedding.__dict__)
        self.message_bus = MessageBus()
        self.state_manager = StateManager()
        self.metrics = MetricsCollector()
        self.performance_tracker = PerformanceTracker()
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Active sessions
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Multi-Agent RAG Orchestrator initialized")
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all specialized agents."""
        agents = {}
        
        try:
            agents['research'] = ResearchAgent(
                vector_store=self.vector_store,
                embedding_manager=self.embedding_manager
            )
            
            agents['analysis'] = AnalysisAgent(
                config=self.config.analysis.__dict__ if hasattr(self.config, 'analysis') else {}
            )
            
            agents['writing'] = WritingAgent()
            
            agents['validation'] = ValidationAgent()
            
            # Coordinator needs to know about other agents
            agents['coordinator'] = CoordinatorAgent(
                available_agents={k: v for k, v in agents.items() if k != 'coordinator'}
            )
            
            logger.info(f"Initialized {len(agents)} agents")
            
        except Exception as e:
            logger.error(f"Error initializing agents: {e}")
            # Return empty dict if initialization fails
            agents = {}
            
        return agents
    
    async def start(self):
        """Start the orchestrator and all components."""
        await self.message_bus.start()
        logger.info("Orchestrator started")
    
    async def stop(self):
        """Stop the orchestrator and cleanup."""
        await self.message_bus.stop()
        logger.info("Orchestrator stopped")
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """
        Process a query using the multi-agent system.
        
        Args:
            request: Query request with query text and optional parameters
            
        Returns:
            QueryResponse with generated response and metadata
        """
        start_time = datetime.now()
        self.performance_tracker.start_timer("query_processing")
        
        try:
            # Initialize session
            if request.session_id not in self.active_sessions:
                self.active_sessions[request.session_id] = {
                    'created_at': start_time,
                    'queries': [],
                    'context': request.context
                }
            
            session = self.active_sessions[request.session_id]
            session['queries'].append({
                'query': request.query,
                'timestamp': start_time
            })
            
            # Determine which agents to use
            agents_to_use = self._select_agents(request)
            
            # Process query through selected agents
            results = {}
            
            # Research phase
            if 'research' in agents_to_use and 'research' in self.agents:
                research_task = {
                    "task_type": "document_search",
                    "query": request.query,
                    "context": request.context
                }
                research_result = await self.agents['research'].execute_task(research_task)
                results['research'] = research_result
            
            # Analysis phase
            if 'analysis' in agents_to_use and 'analysis' in self.agents:
                analysis_task = {
                    "task_type": "content_analysis",
                    "data": results.get('research', {}),
                    "query": request.query,
                    "context": request.context
                }
                analysis_result = await self.agents['analysis'].execute_task(analysis_task)
                results['analysis'] = analysis_result
            
            # Writing phase
            if 'writing' in agents_to_use and 'writing' in self.agents:
                writing_task = {
                    "task_type": "synthesis_report",
                    "query": request.query,
                    "research_data": results.get('research', {}),
                    "analysis_data": results.get('analysis', {}),
                    "context": request.context
                }
                writing_result = await self.agents['writing'].execute_task(writing_task)
                results['writing'] = writing_result
            
            # Validation phase
            if 'validation' in agents_to_use and 'validation' in self.agents:
                validation_task = {
                    "task_type": "response_validation",
                    "response": results.get('writing', {}),
                    "query": request.query,
                    "context": request.context
                }
                validation_result = await self.agents['validation'].execute_task(validation_task)
                results['validation'] = validation_result
            
            # Combine results
            final_response = self._combine_results(results, request)
            
            processing_time = self.performance_tracker.end_timer("query_processing")
            
            # Record metrics
            self.metrics.record_metric("query_processing_time", processing_time)
            self.metrics.increment_counter("queries_processed")
            
            response = QueryResponse(
                query=request.query,
                response=final_response.get('response', 'No response generated'),
                session_id=request.session_id or str(uuid.uuid4()),
                sources=final_response.get('sources', []),
                confidence=final_response.get('confidence', 0.0),
                processing_time=processing_time,
                agents_involved=agents_to_use,
                metadata=final_response.get('metadata', {})
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            processing_time = self.performance_tracker.end_timer("query_processing")
            
            return QueryResponse(
                query=request.query,
                response=f"Error processing query: {str(e)}",
                session_id=request.session_id or str(uuid.uuid4()),
                sources=[],
                confidence=0.0,
                processing_time=processing_time,
                agents_involved=[],
                metadata={'error': str(e)}
            )
    
    def _select_agents(self, request: QueryRequest) -> List[str]:
        """Select which agents to use based on the query."""
        # For now, use all agents for comprehensive processing
        return ['research', 'analysis', 'writing', 'validation']
    
    def _combine_results(self, results: Dict[str, Any], request: QueryRequest) -> Dict[str, Any]:
        """Combine results from multiple agents into final response."""
        # Start with writing result if available
        final_result = results.get('writing', {})
        
        # Add sources from research
        research_sources = results.get('research', {}).get('sources', [])
        final_result['sources'] = research_sources
        
        # Use validation confidence if available
        validation_confidence = results.get('validation', {}).get('confidence', 0.7)
        final_result['confidence'] = validation_confidence
        
        # Combine metadata
        final_result['metadata'] = {
            'research': results.get('research', {}),
            'analysis': results.get('analysis', {}),
            'validation': results.get('validation', {})
        }
        
        return final_result
    
    async def add_documents(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """Add documents to the vector store."""
        try:
            # Convert documents to proper format
            document_dicts = []
            for i, doc in enumerate(documents):
                if isinstance(doc, str):
                    doc_dict = {
                        "content": doc,
                        "id": f"doc_{i}",
                        "metadata": metadata[i] if metadata and i < len(metadata) else {}
                    }
                else:
                    doc_dict = doc
                document_dicts.append(doc_dict)
            
            # Generate embeddings
            embeddings = await self.embedding_manager.generate_embeddings([d["content"] for d in document_dicts])
            
            # Convert embeddings list to numpy array
            import numpy as np
            vectors = np.array(embeddings)
            
            # Add to vector store
            await self.vector_store.add_documents(document_dicts, vectors)
            
            logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    async def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            'active': True,
            'agents': list(self.agents.keys()),
            'active_sessions': len(self.active_sessions),
            'metrics': self.metrics.get_metrics(),
            'vector_store_size': getattr(self.vector_store, 'size', 0)
        }
    
    async def clear_session(self, session_id: str):
        """Clear a specific session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Cleared session: {session_id}")
    
    async def clear_all_sessions(self):
        """Clear all active sessions."""
        self.active_sessions.clear()
        logger.info("Cleared all sessions")
