"""
State Manager

Advanced state management and persistence for the multi-agent RAG system.
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConversationState:
    """State of a conversation/session."""
    session_id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class StateManager:
    """Advanced state management for multi-agent systems"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.state_storage: Dict[str, Any] = {}
        self.storage_path = Path(self.config.get("storage_path", "data/state"))
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    async def save_state(self, key: str, state: Dict[str, Any]) -> bool:
        """Save state for a key"""
        try:
            self.state_storage[key] = {
                "data": state,
                "timestamp": datetime.now().isoformat(),
                "version": 1
            }
            
            # Persist to disk
            state_file = self.storage_path / f"{key}.json"
            with open(state_file, 'w') as f:
                json.dump(self.state_storage[key], f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save state for {key}: {e}")
            return False
    
    async def load_state(self, key: str) -> Optional[Dict[str, Any]]:
        """Load state for a key"""
        try:
            # Try memory first
            if key in self.state_storage:
                return self.state_storage[key]["data"]
            
            # Try disk
            state_file = self.storage_path / f"{key}.json"
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                
                self.state_storage[key] = state_data
                return state_data["data"]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load state for {key}: {e}")
            return None
    
    async def delete_state(self, key: str) -> bool:
        """Delete state for a key"""
        try:
            # Remove from memory
            if key in self.state_storage:
                del self.state_storage[key]
            
            # Remove from disk
            state_file = self.storage_path / f"{key}.json"
            if state_file.exists():
                state_file.unlink()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete state for {key}: {e}")
            return False
    
    async def list_states(self) -> List[str]:
        """List all stored state keys"""
        keys = set(self.state_storage.keys())
        
        # Add keys from disk
        for state_file in self.storage_path.glob("*.json"):
            keys.add(state_file.stem)
        
        return list(keys)
    
    async def get_state_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get metadata about a state"""
        if key in self.state_storage:
            return {
                "key": key,
                "timestamp": self.state_storage[key]["timestamp"],
                "version": self.state_storage[key]["version"],
                "size": len(str(self.state_storage[key]["data"]))
            }
        
        state_file = self.storage_path / f"{key}.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                
                return {
                    "key": key,
                    "timestamp": state_data["timestamp"],
                    "version": state_data["version"],
                    "size": state_file.stat().st_size
                }
            except Exception as e:
                logger.error(f"Failed to get state info for {key}: {e}")
        
        return None
