"""
Configuration Management

Configuration classes and utilities for the multi-agent RAG orchestrator.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentConfig:
    """Configuration for individual agents."""
    max_tokens: int = 4000
    temperature: float = 0.7
    timeout: float = 30.0
    retry_attempts: int = 3
    capabilities: list = field(default_factory=list)
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    dimension: int = 384
    metric: str = "cosine"
    storage_path: str = "data/vector_store"
    max_vectors: int = 100000


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    max_length: int = 512
    device: str = "cpu"


@dataclass
class WorkflowConfig:
    """Configuration for workflow engine."""
    max_concurrent_workflows: int = 10
    default_timeout: float = 300.0
    checkpoint_interval: int = 100


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and metrics."""
    enabled: bool = True
    metrics_retention_days: int = 30
    performance_tracking: bool = True
    log_level: str = "INFO"


@dataclass
class OrchestratorConfig:
    """Main configuration for the orchestrator."""
    # Agent configurations
    research: AgentConfig = field(default_factory=AgentConfig)
    analysis: AgentConfig = field(default_factory=AgentConfig)
    writing: AgentConfig = field(default_factory=AgentConfig)
    coordinator: AgentConfig = field(default_factory=AgentConfig)
    validation: AgentConfig = field(default_factory=AgentConfig)
    
    # Core component configurations
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Global settings
    data_directory: str = "data"
    cache_directory: str = "cache"
    log_directory: str = "logs"
    
    def __post_init__(self):
        """Initialize directories after object creation."""
        for directory in [self.data_directory, self.cache_directory, self.log_directory]:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "OrchestratorConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "research": self.research.__dict__,
            "analysis": self.analysis.__dict__,
            "writing": self.writing.__dict__,
            "coordinator": self.coordinator.__dict__,
            "validation": self.validation.__dict__,
            "vector_store": self.vector_store.__dict__,
            "embedding": self.embedding.__dict__,
            "workflow": self.workflow.__dict__,
            "monitoring": self.monitoring.__dict__,
            "data_directory": self.data_directory,
            "cache_directory": self.cache_directory,
            "log_directory": self.log_directory,
        }


def load_config(config_path: Optional[str] = None) -> OrchestratorConfig:
    """Load configuration from file or return default."""
    if config_path and Path(config_path).exists():
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return OrchestratorConfig.from_dict(config_dict)
    return OrchestratorConfig()


def save_config(config: OrchestratorConfig, config_path: str):
    """Save configuration to file."""
    import json
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
