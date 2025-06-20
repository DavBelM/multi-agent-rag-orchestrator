"""
Caching Utilities

Advanced caching mechanisms for performance optimization.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import hashlib
import pickle
import logging

logger = logging.getLogger(__name__)


class CacheManager:
    """Advanced cache manager with TTL and size limits"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_size = self.config.get("max_size", 1000)
        self.default_ttl = self.config.get("default_ttl_hours", 24)
        self.cache: Dict[str, Dict[str, Any]] = {}
    
    def set(self, key: str, value: Any, ttl_hours: Optional[int] = None) -> bool:
        """Set a cache entry"""
        try:
            ttl = ttl_hours or self.default_ttl
            expires_at = datetime.now() + timedelta(hours=ttl)
            
            # Evict if cache is full
            if len(self.cache) >= self.max_size:
                self._evict_oldest()
            
            self.cache[key] = {
                "value": value,
                "expires_at": expires_at,
                "created_at": datetime.now(),
                "access_count": 0
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set cache entry {key}: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get a cache entry"""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check expiration
        if datetime.now() > entry["expires_at"]:
            del self.cache[key]
            return None
        
        # Update access statistics
        entry["access_count"] += 1
        entry["last_accessed"] = datetime.now()
        
        return entry["value"]
    
    def delete(self, key: str) -> bool:
        """Delete a cache entry"""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
    
    def _evict_oldest(self):
        """Evict the oldest cache entry"""
        if not self.cache:
            return
        
        oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["created_at"])
        del self.cache[oldest_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": 0.0,  # Would need to track hits/misses
            "total_entries": len(self.cache)
        }
