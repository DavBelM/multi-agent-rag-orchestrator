"""
Communication Layer

Advanced communication and messaging system for multi-agent coordination.
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import logging
import json

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages that can be exchanged between agents."""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    ERROR_REPORT = "error_report"
    COORDINATION = "coordination"
    BROADCAST = "broadcast"
    QUERY = "query"
    RESULT = "result"


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication."""
    id: str
    sender: str
    recipient: str
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None
    priority: int = 0  # Higher numbers = higher priority


class MessageBus:
    """Central message bus for agent communication."""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        
    async def start(self):
        """Start the message bus."""
        self.running = True
        
    async def stop(self):
        """Stop the message bus."""
        self.running = False
        
    def subscribe(self, agent_name: str, handler: Callable[[AgentMessage], None]):
        """Subscribe an agent to receive messages."""
        if agent_name not in self.subscribers:
            self.subscribers[agent_name] = []
        self.subscribers[agent_name].append(handler)
        
    async def send_message(self, message: AgentMessage):
        """Send a message through the bus."""
        await self.message_queue.put(message)


class AgentRegistry:
    """Registry for tracking available agents and their capabilities."""
    
    def __init__(self):
        self.agents: Dict[str, Dict[str, Any]] = {}
        
    def register_agent(self, name: str, agent_type: str, capabilities: List[str], 
                      metadata: Optional[Dict[str, Any]] = None):
        """Register an agent with the registry."""
        self.agents[name] = {
            "type": agent_type,
            "capabilities": capabilities,
            "metadata": metadata or {},
            "registered_at": datetime.now(),
            "status": "active"
        }
        
    def get_agent_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about an agent."""
        return self.agents.get(name)


class CommunicationLayer:
    """Advanced communication layer for agent coordination"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.message_queues: Dict[str, List[Dict[str, Any]]] = {}
        self.subscribers: Dict[str, List[Callable]] = {}
    
    async def send_message(self, sender: str, recipient: str, message: Dict[str, Any]) -> str:
        """Send a message between agents"""
        message_id = f"msg_{datetime.now().isoformat()}_{sender}_{recipient}"
        
        message_envelope = {
            "id": message_id,
            "sender": sender,
            "recipient": recipient,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to recipient's queue
        if recipient not in self.message_queues:
            self.message_queues[recipient] = []
        
        self.message_queues[recipient].append(message_envelope)
        
        # Notify subscribers
        await self._notify_subscribers(recipient, message_envelope)
        
        return message_id
    
    async def receive_messages(self, agent_id: str) -> List[Dict[str, Any]]:
        """Receive messages for an agent"""
        messages = self.message_queues.get(agent_id, [])
        self.message_queues[agent_id] = []  # Clear queue
        return messages
    
    async def subscribe(self, agent_id: str, callback: Callable):
        """Subscribe to messages for an agent"""
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = []
        self.subscribers[agent_id].append(callback)
    
    async def _notify_subscribers(self, agent_id: str, message: Dict[str, Any]):
        """Notify subscribers of new messages"""
        if agent_id in self.subscribers:
            for callback in self.subscribers[agent_id]:
                try:
                    await callback(message)
                except Exception as e:
                    logger.error(f"Error in subscriber callback: {e}")
