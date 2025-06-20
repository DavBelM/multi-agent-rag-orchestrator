"""
Multi-Agent RAG Orchestrator

The main orchestrator that coordinates multiple specialized AI agents
to process complex queries through collaborative intelligence.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid

from .agents import (
    BaseAgent, 
    ResearchAgent, 
    AnalysisAgent, 
    WritingAgent, 
    CoordinatorAgent, 
    ValidationAgent,
    AgentRegistry
)
from .core.vector_store import VectorStore
from .core.embedding_manager import EmbeddingManager
from .core.workflow_engine import WorkflowEngine, WorkflowState, WorkflowResult
from .core.communication_layer import MessageBus, AgentMessage, MessageType, AgentRegistry
from .core.state_manager import StateManager, ConversationState
from .utils.monitoring import MetricsCollector, PerformanceTracker
from .config import OrchestratorConfig

logger = logging.getLogger(__name__)

class QueryComplexity(Enum):
    """Query complexity levels that determine agent coordination strategy"""
    SIMPLE = "simple"           # Single agent can handle
    MODERATE = "moderate"       # 2-3 agents needed
    COMPLEX = "complex"         # Full agent collaboration
    CRITICAL = "critical"       # Maximum validation and review

class ProcessingMode(Enum):
    """Processing modes for different use cases"""
    FAST = "fast"              # Quick responses, minimal validation
    BALANCED = "balanced"      # Standard processing with validation
    THOROUGH = "thorough"      # Deep analysis with multiple perspectives
    RESEARCH = "research"      # Academic-level rigor and citations

@dataclass
class QueryRequest:
    """Represents a query request with metadata and requirements"""
    query: str
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    complexity: QueryComplexity = QueryComplexity.MODERATE
    mode: ProcessingMode = ProcessingMode.BALANCED
    required_agents: Optional[Set[str]] = None
    context: Optional[Dict[str, Any]] = None
    constraints: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class OrchestratorResponse:
    """Comprehensive response from the multi-agent system"""
    response: str
    confidence: float
    processing_time: float
    agent_contributions: Dict[str, Any]
    workflow_trace: List[Dict[str, Any]]
    validation_results: Dict[str, Any]
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    session_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

class MultiAgentOrchestrator:
    """
    Main orchestrator that coordinates multiple AI agents for complex query processing.
    
    This class manages the lifecycle of agent interactions, workflow execution,
    and result aggregation to provide sophisticated multi-perspective responses.
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """Initialize the multi-agent orchestrator"""
        self.config = config or OrchestratorConfig()
        
        # Core components
        self.vector_store = VectorStore(config=self.config.vector_store.__dict__)
        self.embedding_manager = EmbeddingManager(config=self.config.embedding.__dict__)
        self.agent_registry = AgentRegistry()
        self.workflow_engine = WorkflowEngine(config=self.config.workflow.__dict__)
        self.message_bus = MessageBus()
        self.state_manager = StateManager()
        self.metrics = MetricsCollector()
        self.performance_tracker = PerformanceTracker()
        
        # Agent instances
        self.agents: Dict[str, BaseAgent] = {}
        
        # Active sessions
        self.active_sessions: Dict[str, ConversationState] = {}
        
        # Initialize system
        self._initialize_agents()
        self._setup_message_routing()
        
        logger.info("Multi-Agent RAG Orchestrator initialized successfully")
    
    def _initialize_agents(self):
        """Initialize all specialized agents"""
        agent_configs = self.config.agents
        
        # Create specialized agents
        self.agents = {
            "research": ResearchAgent(
                vector_store=self.vector_store,
                embedding_manager=self.embedding_manager,
                config=agent_configs.research.__dict__
            ),
            "analysis": AnalysisAgent(
                vector_store=self.vector_store,
                embedding_manager=self.embedding_manager,
                config=agent_configs.analysis.__dict__
            ),
            "writing": WritingAgent(
                config=agent_configs.writing.__dict__
            ),
            "coordinator": CoordinatorAgent(
                available_agents=["research", "analysis", "writing", "validation"],
                config=agent_configs.coordinator.__dict__
            ),
            "validation": ValidationAgent(
                config=agent_configs.validation.__dict__
            )
        }
        
        # Register agents
        for agent_name, agent in self.agents.items():
            self.agent_registry.register(agent_name, agent)
        
        logger.info(f"Initialized {len(self.agents)} specialized agents")
    
    def _setup_message_routing(self):
        """Set up message routing between agents"""
        # Subscribe agents to relevant message types
        for agent in self.agents.values():
            self.message_bus.subscribe(agent.name, agent.handle_message)
        
        # Set up workflow coordination messages
        self.message_bus.subscribe("workflow", self.workflow_engine.handle_message)
    
    async def process_query(self, request: QueryRequest) -> OrchestratorResponse:
        """
        Process a query through the multi-agent system
        
        Args:
            request: The query request with context and requirements
            
        Returns:
            Comprehensive response with agent contributions and metadata
        """
        start_time = self.performance_tracker.start_timer()
        
        try:
            # Initialize conversation state
            conversation_state = ConversationState(
                session_id=request.session_id,
                query=request.query,
                complexity=request.complexity,
                mode=request.mode,
                timestamp=request.timestamp
            )
            
            self.active_sessions[request.session_id] = conversation_state
            
            # Determine optimal workflow
            workflow = await self._plan_workflow(request)
            
            # Execute workflow
            workflow_result = await self.workflow_engine.execute(
                workflow=workflow,
                context={
                    "request": request,
                    "conversation_state": conversation_state
                }
            )
            
            # Aggregate results
            response = await self._aggregate_results(
                request=request,
                workflow_result=workflow_result,
                conversation_state=conversation_state
            )
            
            # Record metrics
            processing_time = self.performance_tracker.end_timer(start_time)
            await self._record_metrics(request, response, processing_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            processing_time = self.performance_tracker.end_timer(start_time)
            
            # Return error response
            return OrchestratorResponse(
                response=f"I apologize, but I encountered an error processing your query: {str(e)}",
                confidence=0.0,
                processing_time=processing_time,
                agent_contributions={},
                workflow_trace=[],
                validation_results={"error": str(e)},
                sources=[],
                metadata={"error": True, "error_message": str(e)},
                session_id=request.session_id
            )
        
        finally:
            # Cleanup session if needed
            if request.session_id in self.active_sessions:
                # Keep recent sessions for context
                if len(self.active_sessions) > self.config.max_active_sessions:
                    oldest_session = min(self.active_sessions.keys(), 
                                       key=lambda x: self.active_sessions[x].timestamp)
                    del self.active_sessions[oldest_session]
    
    async def _plan_workflow(self, request: QueryRequest) -> Dict[str, Any]:
        """Plan the optimal workflow based on query requirements"""
        
        # Get coordinator agent's workflow recommendation
        coordinator = self.agents["coordinator"]
        workflow_plan = await coordinator.plan_workflow(request)
        
        # Adapt based on complexity and mode
        if request.complexity == QueryComplexity.SIMPLE:
            # Single agent workflow
            if request.mode == ProcessingMode.FAST:
                return self._create_fast_workflow(request)
            else:
                return self._create_simple_workflow(request)
        
        elif request.complexity == QueryComplexity.MODERATE:
            # Multi-agent collaboration
            return self._create_collaborative_workflow(request)
        
        elif request.complexity == QueryComplexity.COMPLEX:
            # Full orchestration with validation
            return self._create_comprehensive_workflow(request)
        
        else:  # CRITICAL
            # Maximum rigor with multiple validation rounds
            return self._create_critical_workflow(request)
    
    def _create_fast_workflow(self, request: QueryRequest) -> Dict[str, Any]:
        """Create a fast workflow for simple queries"""
        return {
            "type": "sequential",
            "steps": [
                {
                    "agent": "research",
                    "task": "quick_search",
                    "parallel": False,
                    "timeout": 10
                },
                {
                    "agent": "writing", 
                    "task": "generate_response",
                    "parallel": False,
                    "timeout": 15
                }
            ],
            "validation": False,
            "max_duration": 30
        }
    
    def _create_simple_workflow(self, request: QueryRequest) -> Dict[str, Any]:
        """Create a simple workflow with basic validation"""
        return {
            "type": "sequential", 
            "steps": [
                {
                    "agent": "research",
                    "task": "comprehensive_search",
                    "parallel": False,
                    "timeout": 30
                },
                {
                    "agent": "analysis",
                    "task": "basic_analysis", 
                    "parallel": False,
                    "timeout": 20
                },
                {
                    "agent": "writing",
                    "task": "generate_response",
                    "parallel": False,
                    "timeout": 25
                },
                {
                    "agent": "validation",
                    "task": "quick_validation",
                    "parallel": False,
                    "timeout": 15
                }
            ],
            "validation": True,
            "max_duration": 90
        }
    
    def _create_collaborative_workflow(self, request: QueryRequest) -> Dict[str, Any]:
        """Create a collaborative workflow with parallel processing"""
        return {
            "type": "hybrid",
            "steps": [
                {
                    "phase": "research",
                    "agents": ["research"],
                    "task": "comprehensive_search",
                    "parallel": False,
                    "timeout": 45
                },
                {
                    "phase": "analysis",
                    "agents": ["analysis"],
                    "task": "deep_analysis",
                    "parallel": True,
                    "timeout": 60,
                    "depends_on": ["research"]
                },
                {
                    "phase": "synthesis",
                    "agents": ["writing"],
                    "task": "synthesis_response",
                    "parallel": False,
                    "timeout": 40,
                    "depends_on": ["analysis"]
                },
                {
                    "phase": "validation",
                    "agents": ["validation"],
                    "task": "comprehensive_validation",
                    "parallel": True,
                    "timeout": 30,
                    "depends_on": ["synthesis"]
                }
            ],
            "validation": True,
            "consensus_required": True,
            "max_duration": 180
        }
    
    def _create_comprehensive_workflow(self, request: QueryRequest) -> Dict[str, Any]:
        """Create a comprehensive workflow with full agent collaboration"""
        return {
            "type": "comprehensive",
            "steps": [
                {
                    "phase": "planning",
                    "agents": ["coordinator"],
                    "task": "detailed_planning",
                    "parallel": False,
                    "timeout": 20
                },
                {
                    "phase": "research",
                    "agents": ["research"],
                    "task": "exhaustive_search",
                    "parallel": False, 
                    "timeout": 60,
                    "depends_on": ["planning"]
                },
                {
                    "phase": "multi_analysis",
                    "agents": ["analysis"],
                    "task": "multi_perspective_analysis",
                    "parallel": True,
                    "timeout": 90,
                    "depends_on": ["research"]
                },
                {
                    "phase": "collaborative_writing",
                    "agents": ["writing", "analysis"],
                    "task": "collaborative_synthesis",
                    "parallel": True,
                    "timeout": 60,
                    "depends_on": ["multi_analysis"]
                },
                {
                    "phase": "comprehensive_validation",
                    "agents": ["validation", "research"],
                    "task": "fact_check_and_validate",
                    "parallel": True,
                    "timeout": 45,
                    "depends_on": ["collaborative_writing"]
                },
                {
                    "phase": "final_review",
                    "agents": ["coordinator"],
                    "task": "quality_review",
                    "parallel": False,
                    "timeout": 30,
                    "depends_on": ["comprehensive_validation"]
                }
            ],
            "validation": True,
            "consensus_required": True,
            "multiple_iterations": True,
            "max_duration": 300
        }
    
    def _create_critical_workflow(self, request: QueryRequest) -> Dict[str, Any]:
        """Create a critical workflow with maximum rigor"""
        return {
            "type": "critical",
            "steps": [
                {
                    "phase": "strategic_planning",
                    "agents": ["coordinator"],
                    "task": "strategic_planning",
                    "parallel": False,
                    "timeout": 30
                },
                {
                    "phase": "multi_source_research",
                    "agents": ["research"],
                    "task": "academic_research",
                    "parallel": False,
                    "timeout": 120,
                    "depends_on": ["strategic_planning"]
                },
                {
                    "phase": "expert_analysis",
                    "agents": ["analysis"],
                    "task": "expert_level_analysis",
                    "parallel": True,
                    "timeout": 150,
                    "depends_on": ["multi_source_research"]
                },
                {
                    "phase": "peer_review_writing",
                    "agents": ["writing", "analysis"],
                    "task": "academic_writing",
                    "parallel": True,
                    "timeout": 120,
                    "depends_on": ["expert_analysis"]
                },
                {
                    "phase": "rigorous_validation",
                    "agents": ["validation", "research", "analysis"],
                    "task": "rigorous_fact_checking",
                    "parallel": True,
                    "timeout": 90,
                    "depends_on": ["peer_review_writing"]
                },
                {
                    "phase": "quality_assurance",
                    "agents": ["coordinator", "validation"],
                    "task": "final_qa_review",
                    "parallel": False,
                    "timeout": 60,
                    "depends_on": ["rigorous_validation"]
                }
            ],
            "validation": True,
            "consensus_required": True,
            "multiple_iterations": True,
            "peer_review": True,
            "citation_required": True,
            "max_duration": 600
        }
    
    async def _aggregate_results(
        self, 
        request: QueryRequest,
        workflow_result: WorkflowResult,
        conversation_state: ConversationState
    ) -> OrchestratorResponse:
        """Aggregate results from all agents into final response"""
        
        # Extract agent contributions
        agent_contributions = {}
        sources = []
        confidence_scores = []
        
        for step_result in workflow_result.step_results:
            agent_name = step_result.agent_name
            agent_contributions[agent_name] = {
                "task": step_result.task,
                "result": step_result.result,
                "confidence": step_result.confidence,
                "processing_time": step_result.processing_time,
                "metadata": step_result.metadata
            }
            
            # Collect sources
            if "sources" in step_result.metadata:
                sources.extend(step_result.metadata["sources"])
            
            # Collect confidence scores
            confidence_scores.append(step_result.confidence)
        
        # Calculate overall confidence
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        
        # Get final response (typically from writing agent)
        final_response = workflow_result.final_result.get("response", "")
        
        # Validation results
        validation_results = {}
        if "validation" in agent_contributions:
            validation_results = agent_contributions["validation"].get("result", {})
        
        return OrchestratorResponse(
            response=final_response,
            confidence=overall_confidence,
            processing_time=workflow_result.total_duration,
            agent_contributions=agent_contributions,
            workflow_trace=workflow_result.execution_trace,
            validation_results=validation_results,
            sources=sources,
            metadata={
                "workflow_type": workflow_result.workflow_type,
                "total_steps": len(workflow_result.step_results),
                "session_id": request.session_id,
                "complexity": request.complexity.value,
                "mode": request.mode.value
            },
            session_id=request.session_id
        )
    
    async def _record_metrics(
        self, 
        request: QueryRequest, 
        response: OrchestratorResponse,
        processing_time: float
    ):
        """Record metrics for monitoring and analytics"""
        
        # Basic metrics
        self.metrics.record_query(
            session_id=request.session_id,
            query_length=len(request.query),
            complexity=request.complexity.value,
            mode=request.mode.value,
            processing_time=processing_time,
            confidence=response.confidence,
            agent_count=len(response.agent_contributions)
        )
        
        # Agent performance metrics
        for agent_name, contribution in response.agent_contributions.items():
            self.metrics.record_agent_performance(
                agent_name=agent_name,
                task=contribution["task"],
                processing_time=contribution["processing_time"],
                confidence=contribution["confidence"]
            )
        
        # Workflow metrics
        self.metrics.record_workflow(
            workflow_type=response.metadata["workflow_type"],
            total_steps=response.metadata["total_steps"],
            success=len(response.response) > 0,
            duration=processing_time
        )
    
    async def get_session_history(self, session_id: str) -> Optional[ConversationState]:
        """Get conversation history for a session"""
        return self.active_sessions.get(session_id)
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        status = {}
        for agent_name, agent in self.agents.items():
            status[agent_name] = await agent.get_status()
        return status
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        return {
            "orchestrator": {
                "active_sessions": len(self.active_sessions),
                "total_queries": self.metrics.get_total_queries(),
                "average_processing_time": self.metrics.get_average_processing_time(),
                "uptime": self.performance_tracker.get_uptime()
            },
            "agents": await self.get_agent_status(),
            "workflow_engine": self.workflow_engine.get_metrics(),
            "message_bus": self.message_bus.get_metrics(),
            "performance": self.performance_tracker.get_summary()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }
        
        # Check agents
        for agent_name, agent in self.agents.items():
            try:
                agent_health = await agent.health_check()
                health_status["components"][f"agent_{agent_name}"] = agent_health
            except Exception as e:
                health_status["components"][f"agent_{agent_name}"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["status"] = "degraded"
        
        # Check other components
        try:
            health_status["components"]["workflow_engine"] = self.workflow_engine.health_check()
            health_status["components"]["message_bus"] = self.message_bus.health_check()
            health_status["components"]["state_manager"] = self.state_manager.health_check()
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
        
        return health_status
    
    async def shutdown(self):
        """Gracefully shutdown the orchestrator"""
        logger.info("Shutting down Multi-Agent RAG Orchestrator...")
        
        # Stop all agents
        for agent in self.agents.values():
            await agent.shutdown()
        
        # Stop workflow engine
        await self.workflow_engine.shutdown()
        
        # Clear sessions
        self.active_sessions.clear()
        
        logger.info("Multi-Agent RAG Orchestrator shutdown complete")
