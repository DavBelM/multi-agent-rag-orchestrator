"""
Base Agent Class

This module provides the base class for all agents in the multi-agent RAG system.
All specialized agents inherit from this base class to ensure consistent interfaces.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import uuid

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """Message structure for agent communication"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    recipient: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    message_type: str = "standard"
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 1  # 1=low, 2=medium, 3=high


@dataclass
class AgentState:
    """Agent state tracking"""
    status: str = "idle"  # idle, working, completed, error
    current_task: Optional[str] = None
    progress: float = 0.0
    last_activity: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Base class for all agents in the multi-agent RAG system.
    
    Provides common functionality including:
    - Message handling
    - State management
    - Logging
    - Performance monitoring
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        capabilities: List[str],
        config: Optional[Dict[str, Any]] = None
    ):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.capabilities = capabilities
        self.config = config or {}
        
        # Initialize state
        self.state = AgentState()
        self.message_history: List[AgentMessage] = []
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Performance metrics
        self.performance_metrics = {
            'tasks_completed': 0,
            'total_processing_time': 0.0,
            'success_rate': 0.0,
            'average_response_time': 0.0
        }
        
        self.logger.info(f"Initialized agent: {self.name} ({self.agent_id})")
    
    def update_state(self, status: str, task: Optional[str] = None, progress: float = 0.0):
        """Update agent state"""
        self.state.status = status
        self.state.current_task = task
        self.state.progress = progress
        self.state.last_activity = datetime.now()
        
        self.logger.debug(f"State updated: {status} - {task} ({progress:.1%})")
    
    def send_message(self, recipient: str, content: Dict[str, Any], message_type: str = "standard") -> str:
        """Send a message to another agent or component"""
        message = AgentMessage(
            sender=self.agent_id,
            recipient=recipient,
            content=content,
            message_type=message_type
        )
        
        self.message_history.append(message)
        self.logger.debug(f"Message sent to {recipient}: {message_type}")
        
        return message.id
    
    def receive_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Receive and process a message"""
        self.message_history.append(message)
        self.logger.debug(f"Message received from {message.sender}: {message.message_type}")
        
        return self.process_message(message)
    
    @abstractmethod
    def process_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Process a received message - implemented by each agent"""
        pass
    
    @abstractmethod
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task - implemented by each agent"""
        pass
    
    def get_capabilities(self) -> List[str]:
        """Return agent capabilities"""
        return self.capabilities
    
    def get_state(self) -> AgentState:
        """Return current agent state"""
        return self.state
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return performance metrics"""
        return {
            **self.performance_metrics,
            'state': {
                'status': self.state.status,
                'current_task': self.state.current_task,
                'progress': self.state.progress,
                'last_activity': self.state.last_activity.isoformat()
            }
        }
    
    def update_metrics(self, task_time: float, success: bool):
        """Update performance metrics"""
        self.performance_metrics['tasks_completed'] += 1
        self.performance_metrics['total_processing_time'] += task_time
        
        # Update success rate
        total_tasks = self.performance_metrics['tasks_completed']
        if success:
            current_successes = self.performance_metrics['success_rate'] * (total_tasks - 1)
            self.performance_metrics['success_rate'] = (current_successes + 1) / total_tasks
        else:
            current_successes = self.performance_metrics['success_rate'] * (total_tasks - 1)
            self.performance_metrics['success_rate'] = current_successes / total_tasks
        
        # Update average response time
        self.performance_metrics['average_response_time'] = (
            self.performance_metrics['total_processing_time'] / total_tasks
        )
    
    def __str__(self) -> str:
        return f"{self.name} ({self.agent_id}) - {self.state.status}"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(agent_id='{self.agent_id}', name='{self.name}')"
