"""
Vector Store

Advanced vector storage and retrieval system for the multi-agent RAG orchestrator.
Handles document embeddings, similarity search, and metadata management.
"""

import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import json
import pickle
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Advanced vector store for document embeddings and similarity search.
    
    Features:
    - Efficient similarity search with multiple algorithms
    - Metadata filtering and indexing
    - Persistent storage and retrieval
    - Batch operations and optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Storage configuration
        self.storage_path = Path(self.config.get("storage_path", "data/vector_store"))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Vector storage
        self.vectors: np.ndarray = None
        self.metadata: List[Dict[str, Any]] = []
        self.document_ids: List[str] = []
        self.id_to_index: Dict[str, int] = {}
        
        # Configuration
        self.dimension = self.config.get("dimension", 384)  # Default for sentence-transformers
        self.similarity_metric = self.config.get("similarity_metric", "cosine")
        self.index_type = self.config.get("index_type", "flat")  # flat, faiss, annoy
        
        # Performance settings
        self.batch_size = self.config.get("batch_size", 100)
        self.cache_size = self.config.get("cache_size", 1000)
        
        # Initialize storage
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize vector storage"""
        try:
            self._load_from_disk()
            logger.info(f"Loaded vector store with {len(self.document_ids)} documents")
        except (FileNotFoundError, EOFError):
            logger.info("Initializing new vector store")
            self.vectors = np.empty((0, self.dimension), dtype=np.float32)
            self.metadata = []
            self.document_ids = []
            self.id_to_index = {}
    
    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        vectors: np.ndarray,
        batch_size: Optional[int] = None
    ) -> List[str]:
        """Add documents with their embeddings to the vector store"""
        batch_size = batch_size or self.batch_size
        
        if len(documents) != len(vectors):
            raise ValueError("Number of documents must match number of vectors")
        
        document_ids = []
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_vectors = vectors[i:i + batch_size]
            
            batch_ids = await self._add_batch(batch_docs, batch_vectors)
            document_ids.extend(batch_ids)
        
        # Save to disk
        await self._save_to_disk()
        
        logger.info(f"Added {len(documents)} documents to vector store")
        return document_ids
    
    async def _add_batch(self, documents: List[Dict[str, Any]], vectors: np.ndarray) -> List[str]:
        """Add a batch of documents"""
        batch_ids = []
        
        for doc, vector in zip(documents, vectors):
            # Generate document ID
            doc_id = doc.get("id", f"doc_{len(self.document_ids)}")
            
            # Ensure unique ID
            original_id = doc_id
            counter = 1
            while doc_id in self.id_to_index:
                doc_id = f"{original_id}_{counter}"
                counter += 1
            
            # Add to storage
            self.document_ids.append(doc_id)
            self.id_to_index[doc_id] = len(self.metadata)
            
            # Store metadata
            metadata = {
                "id": doc_id,
                "content": doc.get("content", ""),
                "source": doc.get("source", "unknown"),
                "timestamp": doc.get("timestamp", datetime.now().isoformat()),
                "metadata": doc.get("metadata", {}),
                "content_length": len(doc.get("content", "")),
                "added_at": datetime.now().isoformat()
            }
            self.metadata.append(metadata)
            
            # Add vector
            if self.vectors.size == 0:
                self.vectors = vector.reshape(1, -1)
            else:
                self.vectors = np.vstack([self.vectors, vector.reshape(1, -1)])
            
            batch_ids.append(doc_id)
        
        return batch_ids
    
    async def similarity_search(
        self,
        query: str = "",
        query_vector: Optional[np.ndarray] = None,
        limit: int = 10,
        threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform similarity search"""
        if query_vector is None and not query:
            raise ValueError("Either query or query_vector must be provided")
        
        if self.vectors.size == 0:
            logger.warning("Vector store is empty")
            return []
        
        # Get query vector (this would normally use the embedding manager)
        if query_vector is None:
            # Placeholder - in real implementation, this would call embedding manager
            query_vector = np.random.rand(self.dimension).astype(np.float32)
        
        # Calculate similarities
        similarities = await self._calculate_similarities(query_vector)
        
        # Apply filters
        if filters:
            similarities = await self._apply_filters(similarities, filters)
        
        # Apply threshold
        similarities = [(idx, score) for idx, score in similarities if score >= threshold]
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Limit results
        similarities = similarities[:limit]
        
        # Build results
        results = []
        for idx, score in similarities:
            metadata = self.metadata[idx]
            result = {
                "id": metadata["id"],
                "content": metadata["content"],
                "source": metadata["source"],
                "score": float(score),
                "metadata": metadata["metadata"]
            }
            results.append(result)
        
        return results
    
    async def _calculate_similarities(self, query_vector: np.ndarray) -> List[Tuple[int, float]]:
        """Calculate similarities between query vector and stored vectors"""
        if self.similarity_metric == "cosine":
            # Cosine similarity
            query_norm = np.linalg.norm(query_vector)
            if query_norm == 0:
                return [(i, 0.0) for i in range(len(self.vectors))]
            
            vector_norms = np.linalg.norm(self.vectors, axis=1)
            # Avoid division by zero
            vector_norms = np.where(vector_norms == 0, 1e-8, vector_norms)
            
            similarities = np.dot(self.vectors, query_vector) / (vector_norms * query_norm)
            similarities = np.clip(similarities, -1.0, 1.0)  # Ensure valid range
            
        elif self.similarity_metric == "dot_product":
            similarities = np.dot(self.vectors, query_vector)
            
        elif self.similarity_metric == "euclidean":
            # Convert to similarity (inverse of distance)
            distances = np.linalg.norm(self.vectors - query_vector, axis=1)
            similarities = 1.0 / (1.0 + distances)
            
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")
        
        return [(i, float(sim)) for i, sim in enumerate(similarities)]
    
    async def _apply_filters(
        self,
        similarities: List[Tuple[int, float]],
        filters: Dict[str, Any]
    ) -> List[Tuple[int, float]]:
        """Apply metadata filters to search results"""
        filtered_similarities = []
        
        for idx, score in similarities:
            metadata = self.metadata[idx]
            
            # Check each filter
            include = True
            for filter_key, filter_value in filters.items():
                if filter_key in metadata:
                    if isinstance(filter_value, list):
                        # List filter - value must be in list
                        if metadata[filter_key] not in filter_value:
                            include = False
                            break
                    elif isinstance(filter_value, dict):
                        # Range filter
                        if "min" in filter_value and metadata[filter_key] < filter_value["min"]:
                            include = False
                            break
                        if "max" in filter_value and metadata[filter_key] > filter_value["max"]:
                            include = False
                            break
                    else:
                        # Exact match
                        if metadata[filter_key] != filter_value:
                            include = False
                            break
                elif filter_key in metadata.get("metadata", {}):
                    # Check nested metadata
                    nested_value = metadata["metadata"][filter_key]
                    if nested_value != filter_value:
                        include = False
                        break
            
            if include:
                filtered_similarities.append((idx, score))
        
        return filtered_similarities
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific document by ID"""
        if document_id not in self.id_to_index:
            return None
        
        idx = self.id_to_index[document_id]
        metadata = self.metadata[idx]
        
        return {
            "id": metadata["id"],
            "content": metadata["content"],
            "source": metadata["source"],
            "metadata": metadata["metadata"],
            "vector": self.vectors[idx].tolist(),
            "timestamp": metadata["timestamp"]
        }
    
    async def update_document(
        self,
        document_id: str,
        content: Optional[str] = None,
        vector: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update an existing document"""
        if document_id not in self.id_to_index:
            return False
        
        idx = self.id_to_index[document_id]
        
        # Update content
        if content is not None:
            self.metadata[idx]["content"] = content
            self.metadata[idx]["content_length"] = len(content)
        
        # Update vector
        if vector is not None:
            self.vectors[idx] = vector
        
        # Update metadata
        if metadata is not None:
            self.metadata[idx]["metadata"].update(metadata)
        
        # Update timestamp
        self.metadata[idx]["updated_at"] = datetime.now().isoformat()
        
        # Save changes
        await self._save_to_disk()
        
        return True
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document from the vector store"""
        if document_id not in self.id_to_index:
            return False
        
        idx = self.id_to_index[document_id]
        
        # Remove from arrays
        self.vectors = np.delete(self.vectors, idx, axis=0)
        del self.metadata[idx]
        del self.document_ids[idx]
        
        # Rebuild index
        self.id_to_index = {doc_id: i for i, doc_id in enumerate(self.document_ids)}
        
        # Save changes
        await self._save_to_disk()
        
        return True
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        if not self.metadata:
            return {
                "total_documents": 0,
                "total_vectors": 0,
                "vector_dimension": self.dimension,
                "storage_size_mb": 0
            }
        
        # Calculate storage size
        storage_size = 0
        for file_path in self.storage_path.glob("*"):
            storage_size += file_path.stat().st_size
        
        # Content statistics
        content_lengths = [meta["content_length"] for meta in self.metadata]
        
        # Source distribution
        sources = [meta["source"] for meta in self.metadata]
        source_counts = {}
        for source in sources:
            source_counts[source] = source_counts.get(source, 0) + 1
        
        return {
            "total_documents": len(self.metadata),
            "total_vectors": len(self.vectors) if self.vectors.size > 0 else 0,
            "vector_dimension": self.dimension,
            "storage_size_mb": storage_size / (1024 * 1024),
            "content_statistics": {
                "average_length": sum(content_lengths) / len(content_lengths),
                "min_length": min(content_lengths),
                "max_length": max(content_lengths),
                "total_characters": sum(content_lengths)
            },
            "source_distribution": source_counts,
            "similarity_metric": self.similarity_metric,
            "index_type": self.index_type
        }
    
    async def _save_to_disk(self):
        """Save vector store to disk"""
        try:
            # Save vectors
            if self.vectors.size > 0:
                np.save(self.storage_path / "vectors.npy", self.vectors)
            
            # Save metadata
            with open(self.storage_path / "metadata.json", "w") as f:
                json.dump(self.metadata, f, indent=2)
            
            # Save document IDs and index
            with open(self.storage_path / "document_ids.json", "w") as f:
                json.dump({
                    "document_ids": self.document_ids,
                    "id_to_index": self.id_to_index
                }, f, indent=2)
            
            # Save configuration
            with open(self.storage_path / "config.json", "w") as f:
                json.dump({
                    "dimension": self.dimension,
                    "similarity_metric": self.similarity_metric,
                    "index_type": self.index_type,
                    "last_updated": datetime.now().isoformat()
                }, f, indent=2)
            
            logger.debug("Vector store saved to disk")
            
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            raise
    
    def _load_from_disk(self):
        """Load vector store from disk"""
        # Load configuration
        with open(self.storage_path / "config.json", "r") as f:
            config = json.load(f)
            self.dimension = config["dimension"]
            self.similarity_metric = config["similarity_metric"]
            self.index_type = config["index_type"]
        
        # Load vectors
        vectors_path = self.storage_path / "vectors.npy"
        if vectors_path.exists():
            self.vectors = np.load(vectors_path)
        else:
            self.vectors = np.empty((0, self.dimension), dtype=np.float32)
        
        # Load metadata
        with open(self.storage_path / "metadata.json", "r") as f:
            self.metadata = json.load(f)
        
        # Load document IDs and index
        with open(self.storage_path / "document_ids.json", "r") as f:
            data = json.load(f)
            self.document_ids = data["document_ids"]
            self.id_to_index = data["id_to_index"]
    
    async def clear(self):
        """Clear all data from the vector store"""
        self.vectors = np.empty((0, self.dimension), dtype=np.float32)
        self.metadata = []
        self.document_ids = []
        self.id_to_index = {}
        
        # Remove files
        for file_path in self.storage_path.glob("*"):
            file_path.unlink()
        
        logger.info("Vector store cleared")
    
    def __len__(self) -> int:
        return len(self.document_ids)
    
    def __contains__(self, document_id: str) -> bool:
        return document_id in self.id_to_index
