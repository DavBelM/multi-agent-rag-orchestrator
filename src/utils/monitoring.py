"""
Performance Monitoring Utilities

Advanced performance monitoring and metrics collection.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import time
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and aggregates metrics from the system."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.counters: Dict[str, int] = {}
        
    def record_metric(self, name: str, value: float):
        """Record a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
        
    def increment_counter(self, name: str, amount: int = 1):
        """Increment a counter."""
        self.counters[name] = self.counters.get(name, 0) + amount
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        return {
            "metrics": self.metrics,
            "counters": self.counters
        }


class PerformanceTracker:
    """Tracks performance of operations."""
    
    def __init__(self):
        self.timers: Dict[str, float] = {}
        self.durations: Dict[str, List[float]] = {}
        
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.timers[operation] = time.time()
        
    def end_timer(self, operation: str) -> float:
        """End timing and record duration."""
        if operation not in self.timers:
            return 0.0
            
        duration = time.time() - self.timers[operation]
        del self.timers[operation]
        
        if operation not in self.durations:
            self.durations[operation] = []
        self.durations[operation].append(duration)
        
        return duration


class PerformanceMonitor:
    """Advanced performance monitoring for multi-agent systems"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.active_timers: Dict[str, float] = {}
    
    def start_timer(self, operation: str) -> str:
        """Start a performance timer"""
        timer_id = f"{operation}_{int(time.time() * 1000)}"
        self.active_timers[timer_id] = time.time()
        return timer_id
    
    def end_timer(self, timer_id: str, metadata: Optional[Dict[str, Any]] = None) -> float:
        """End a performance timer and record metrics"""
        if timer_id not in self.active_timers:
            return 0.0
        
        duration = time.time() - self.active_timers[timer_id]
        del self.active_timers[timer_id]
        
        # Extract operation name
        operation = timer_id.rsplit('_', 1)[0]
        
        # Record metric
        self.record_metric(operation, duration, metadata)
        
        return duration
    
    def record_metric(self, metric_name: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """Record a performance metric"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        self.metrics[metric_name].append({
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        })
    
    def get_metrics(self, metric_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics"""
        if metric_name:
            return {metric_name: self.metrics.get(metric_name, [])}
        return self.metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {}
        
        for metric_name, measurements in self.metrics.items():
            if measurements:
                values = [m["value"] for m in measurements]
                summary[metric_name] = {
                    "count": len(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "latest": values[-1]
                }
        
        return summary
