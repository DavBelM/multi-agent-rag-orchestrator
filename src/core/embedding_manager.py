"""
Embedding Manager

Advanced embedding generation and management for the multi-agent RAG system.
Handles multiple embedding models, caching, and optimization.
"""

import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Union
import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta
import logging
import pickle

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Simple embedding cache for performance optimization"""
    
    def __init__(self, max_size: int = 10000, ttl_hours: int = 24):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
    
    def _get_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key for text and model"""
        content = f"{model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """Get cached embedding"""
        key = self._get_cache_key(text, model_name)
        
        if key in self.cache:
            entry = self.cache[key]
            
            # Check TTL
            if datetime.now() - entry["timestamp"] < self.ttl:
                return entry["embedding"]
            else:
                # Remove expired entry
                del self.cache[key]
        
        return None
    
    def set(self, text: str, model_name: str, embedding: np.ndarray):
        """Cache embedding"""
        # Implement LRU eviction if cache is full
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]
        
        key = self._get_cache_key(text, model_name)
        self.cache[key] = {
            "embedding": embedding.copy(),
            "timestamp": datetime.now()
        }
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "ttl_hours": self.ttl.total_seconds() / 3600
        }


class EmbeddingManager:
    """
    Advanced embedding manager for multiple models and optimization.
    
    Features:
    - Multiple embedding model support
    - Intelligent caching and batching
    - Performance optimization
    - Model comparison and selection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Model configuration
        self.default_model = self.config.get("default_model", "sentence-transformers")
        self.available_models = self.config.get("available_models", {
            "sentence-transformers": {
                "model_name": "all-MiniLM-L6-v2",
                "dimension": 384,
                "max_length": 512
            },
            "openai": {
                "model_name": "text-embedding-ada-002",
                "dimension": 1536,
                "max_length": 8191
            }
        })
        
        # Performance settings
        self.batch_size = self.config.get("batch_size", 32)
        self.max_concurrent = self.config.get("max_concurrent", 4)
        self.cache_enabled = self.config.get("cache_enabled", True)
        
        # Initialize cache
        if self.cache_enabled:
            cache_config = self.config.get("cache", {})
            self.cache = EmbeddingCache(
                max_size=cache_config.get("max_size", 10000),
                ttl_hours=cache_config.get("ttl_hours", 24)
            )
        else:
            self.cache = None
        
        # Model instances (lazy loading)
        self._model_instances = {}
        
        # Statistics
        self.stats = {
            "embeddings_generated": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_texts_processed": 0,
            "average_generation_time": 0.0
        }
    
    async def generate_embedding(
        self,
        text: str,
        model: Optional[str] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """Generate embedding for a single text"""
        embeddings = await self.generate_embeddings([text], model, normalize)
        return embeddings[0]
    
    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        normalize: bool = True,
        batch_size: Optional[int] = None
    ) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        model = model or self.default_model
        batch_size = batch_size or self.batch_size
        
        if not texts:
            return []
        
        # Check cache first
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        if self.cache_enabled and self.cache:
            for i, text in enumerate(texts):
                cached_embedding = self.cache.get(text, model)
                if cached_embedding is not None:
                    embeddings.append((i, cached_embedding))
                    self.stats["cache_hits"] += 1
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
                    self.stats["cache_misses"] += 1
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            start_time = datetime.now()
            
            # Process in batches
            new_embeddings = []
            for i in range(0, len(uncached_texts), batch_size):
                batch_texts = uncached_texts[i:i + batch_size]
                batch_embeddings = await self._generate_batch_embeddings(batch_texts, model)
                new_embeddings.extend(batch_embeddings)
            
            # Update statistics
            generation_time = (datetime.now() - start_time).total_seconds()
            self.stats["embeddings_generated"] += len(new_embeddings)
            self.stats["total_texts_processed"] += len(texts)
            
            # Update average generation time
            if self.stats["embeddings_generated"] > 0:
                self.stats["average_generation_time"] = (
                    self.stats["average_generation_time"] * (self.stats["embeddings_generated"] - len(new_embeddings)) + 
                    generation_time
                ) / self.stats["embeddings_generated"]
            
            # Cache new embeddings
            if self.cache_enabled and self.cache:
                for text, embedding in zip(uncached_texts, new_embeddings):
                    self.cache.set(text, model, embedding)
            
            # Add to results with correct indices
            for i, embedding in enumerate(new_embeddings):
                original_index = uncached_indices[i]
                embeddings.append((original_index, embedding))
        
        # Sort by original order and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        result_embeddings = [embedding for _, embedding in embeddings]
        
        # Normalize if requested
        if normalize:
            result_embeddings = [self._normalize_embedding(emb) for emb in result_embeddings]
        
        return result_embeddings
    
    async def _generate_batch_embeddings(self, texts: List[str], model: str) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts"""
        model_config = self.available_models.get(model, {})
        
        if model == "sentence-transformers":
            return await self._generate_sentence_transformer_embeddings(texts, model_config)
        elif model == "openai":
            return await self._generate_openai_embeddings(texts, model_config)
        else:
            # Fallback to simple embeddings
            return await self._generate_simple_embeddings(texts, model_config)
    
    async def _generate_sentence_transformer_embeddings(
        self,
        texts: List[str],
        model_config: Dict[str, Any]
    ) -> List[np.ndarray]:
        """Generate embeddings using sentence-transformers"""
        try:
            # In a real implementation, this would use the actual sentence-transformers library
            # For now, we'll simulate with random embeddings that have the correct structure
            
            dimension = model_config.get("dimension", 384)
            max_length = model_config.get("max_length", 512)
            
            embeddings = []
            for text in texts:
                # Truncate text if too long
                if len(text) > max_length:
                    text = text[:max_length]
                
                # Generate deterministic "embedding" based on text hash for consistency
                text_hash = hashlib.md5(text.encode()).hexdigest()
                seed = int(text_hash[:8], 16)
                np.random.seed(seed % (2**32))
                
                embedding = np.random.normal(0, 1, dimension).astype(np.float32)
                embeddings.append(embedding)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate sentence-transformer embeddings: {e}")
            # Fallback to simple embeddings
            return await self._generate_simple_embeddings(texts, model_config)
    
    async def _generate_openai_embeddings(
        self,
        texts: List[str],
        model_config: Dict[str, Any]
    ) -> List[np.ndarray]:
        """Generate embeddings using OpenAI API"""
        try:
            # In a real implementation, this would call the OpenAI API
            # For now, we'll simulate with random embeddings
            
            dimension = model_config.get("dimension", 1536)
            max_length = model_config.get("max_length", 8191)
            
            embeddings = []
            for text in texts:
                # Truncate text if too long
                if len(text) > max_length:
                    text = text[:max_length]
                
                # Generate deterministic "embedding" based on text hash
                text_hash = hashlib.md5(text.encode()).hexdigest()
                seed = int(text_hash[:8], 16)
                np.random.seed(seed % (2**32))
                
                embedding = np.random.normal(0, 1, dimension).astype(np.float32)
                embeddings.append(embedding)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate OpenAI embeddings: {e}")
            # Fallback to simple embeddings
            return await self._generate_simple_embeddings(texts, model_config)
    
    async def _generate_simple_embeddings(
        self,
        texts: List[str],
        model_config: Dict[str, Any]
    ) -> List[np.ndarray]:
        """Generate simple hash-based embeddings as fallback"""
        dimension = model_config.get("dimension", 384)
        
        embeddings = []
        for text in texts:
            # Create a simple embedding based on text characteristics
            text_hash = hashlib.md5(text.encode()).hexdigest()
            seed = int(text_hash[:8], 16)
            np.random.seed(seed % (2**32))
            
            # Create embedding with some semantic-like properties
            embedding = np.zeros(dimension, dtype=np.float32)
            
            # Add components based on text features
            words = text.lower().split()
            for i, word in enumerate(words[:min(len(words), dimension // 4)]):
                word_hash = hash(word) % dimension
                embedding[word_hash] += 1.0
            
            # Add random noise
            noise = np.random.normal(0, 0.1, dimension).astype(np.float32)
            embedding += noise
            
            embeddings.append(embedding)
        
        return embeddings
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding to unit vector"""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    
    async def compare_embeddings(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = "cosine"
    ) -> float:
        """Compare two embeddings using specified metric"""
        if metric == "cosine":
            # Cosine similarity
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
        
        elif metric == "dot_product":
            return float(np.dot(embedding1, embedding2))
        
        elif metric == "euclidean":
            distance = np.linalg.norm(embedding1 - embedding2)
            # Convert to similarity
            return float(1.0 / (1.0 + distance))
        
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
    
    async def get_model_info(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a model"""
        model = model or self.default_model
        
        if model not in self.available_models:
            raise ValueError(f"Unknown model: {model}")
        
        model_config = self.available_models[model].copy()
        model_config["is_loaded"] = model in self._model_instances
        model_config["is_default"] = model == self.default_model
        
        return model_config
    
    async def benchmark_models(
        self,
        test_texts: List[str],
        models: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Benchmark different models on test texts"""
        models = models or list(self.available_models.keys())
        results = {}
        
        for model in models:
            start_time = datetime.now()
            
            try:
                # Generate embeddings
                embeddings = await self.generate_embeddings(test_texts, model, normalize=False)
                
                # Calculate metrics
                generation_time = (datetime.now() - start_time).total_seconds()
                avg_time_per_text = generation_time / len(test_texts)
                
                # Calculate embedding statistics
                all_embeddings = np.array(embeddings)
                embedding_stats = {
                    "mean_norm": float(np.mean([np.linalg.norm(emb) for emb in embeddings])),
                    "std_norm": float(np.std([np.linalg.norm(emb) for emb in embeddings])),
                    "dimension": len(embeddings[0]) if embeddings else 0
                }
                
                results[model] = {
                    "success": True,
                    "generation_time": generation_time,
                    "avg_time_per_text": avg_time_per_text,
                    "texts_processed": len(test_texts),
                    "embedding_stats": embedding_stats,
                    "model_config": self.available_models[model]
                }
                
            except Exception as e:
                results[model] = {
                    "success": False,
                    "error": str(e),
                    "generation_time": (datetime.now() - start_time).total_seconds(),
                    "model_config": self.available_models[model]
                }
        
        return results
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get embedding manager statistics"""
        stats = self.stats.copy()
        
        # Add cache statistics
        if self.cache:
            stats["cache_stats"] = self.cache.get_stats()
        
        # Add model information
        stats["available_models"] = list(self.available_models.keys())
        stats["default_model"] = self.default_model
        stats["loaded_models"] = list(self._model_instances.keys())
        
        # Calculate derived statistics
        if stats["cache_hits"] + stats["cache_misses"] > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / (stats["cache_hits"] + stats["cache_misses"])
        else:
            stats["cache_hit_rate"] = 0.0
        
        return stats
    
    async def clear_cache(self):
        """Clear embedding cache"""
        if self.cache:
            self.cache.clear()
            logger.info("Embedding cache cleared")
    
    async def precompute_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """Precompute and cache embeddings for a list of texts"""
        model = model or self.default_model
        batch_size = batch_size or self.batch_size
        
        logger.info(f"Precomputing embeddings for {len(texts)} texts using {model}")
        
        embeddings = await self.generate_embeddings(texts, model, batch_size=batch_size)
        
        # Create mapping
        text_to_embedding = {}
        for text, embedding in zip(texts, embeddings):
            text_to_embedding[text] = embedding
        
        logger.info(f"Precomputed {len(embeddings)} embeddings")
        return text_to_embedding
    
    def add_model(self, model_name: str, model_config: Dict[str, Any]):
        """Add a new model configuration"""
        self.available_models[model_name] = model_config
        logger.info(f"Added model configuration: {model_name}")
    
    def remove_model(self, model_name: str):
        """Remove a model configuration"""
        if model_name in self.available_models:
            del self.available_models[model_name]
            
            # Remove from loaded models if present
            if model_name in self._model_instances:
                del self._model_instances[model_name]
            
            # Update default model if necessary
            if self.default_model == model_name:
                if self.available_models:
                    self.default_model = list(self.available_models.keys())[0]
                else:
                    self.default_model = None
            
            logger.info(f"Removed model: {model_name}")
    
    def set_default_model(self, model_name: str):
        """Set the default model"""
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} is not available")
        
        self.default_model = model_name
        logger.info(f"Set default model to: {model_name}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on embedding manager"""
        health_status = {
            "status": "healthy",
            "issues": [],
            "models_status": {},
            "cache_status": "disabled" if not self.cache_enabled else "enabled"
        }
        
        # Test each model with a simple text
        test_text = "This is a test sentence for health check."
        
        for model_name in self.available_models:
            try:
                start_time = datetime.now()
                embedding = await self.generate_embedding(test_text, model_name)
                generation_time = (datetime.now() - start_time).total_seconds()
                
                health_status["models_status"][model_name] = {
                    "status": "healthy",
                    "generation_time": generation_time,
                    "embedding_dimension": len(embedding)
                }
                
            except Exception as e:
                health_status["models_status"][model_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["issues"].append(f"Model {model_name} failed health check: {e}")
        
        # Check cache if enabled
        if self.cache_enabled and self.cache:
            cache_stats = self.cache.get_stats()
            health_status["cache_status"] = f"enabled ({cache_stats['cache_size']}/{cache_stats['max_size']})"
        
        # Overall status
        if health_status["issues"]:
            health_status["status"] = "degraded" if len(health_status["issues"]) < len(self.available_models) else "unhealthy"
        
        return health_status
