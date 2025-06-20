"""
Multi-Agent RAG Orchestrator

A sophisticated multi-agent system for collaborative document analysis and query processing.
This package provides specialized AI agents that work together to deliver comprehensive,
multi-perspective responses to complex queries.

Key Components:
- Agent Framework: Base classes and utilities for agent development
- Specialized Agents: Research, Analysis, Writing, Coordination, and Validation agents
- Orchestration Engine: Workflow management and agent coordination
- Communication Layer: Inter-agent messaging and state management
- Monitoring System: Performance tracking and analytics
"""

__version__ = "1.0.0"
__author__ = "Mitali"
__email__ = "mitali@example.com"

# Core orchestration components
from .orchestrator import MultiAgentRAGOrchestrator
from .core.workflow_engine import WorkflowEngine

# Agent implementations
from .agents import (
    ResearchAgent,
    AnalysisAgent,
    WritingAgent,
    CoordinatorAgent,
    ValidationAgent
)

# Core components
from .core.vector_store import VectorStore
from .core.embedding_manager import EmbeddingManager
from .core.state_manager import StateManager
from .core.communication_layer import CommunicationLayer

__all__ = [
    # Core orchestration
    "MultiAgentRAGOrchestrator",
    "WorkflowEngine",
    
    # Agents
    "ResearchAgent",
    "AnalysisAgent", 
    "WritingAgent",
    "CoordinatorAgent",
    "ValidationAgent",
    
    # Core components
    "VectorStore",
    "EmbeddingManager",
    "StateManager",
    "CommunicationLayer"
]
