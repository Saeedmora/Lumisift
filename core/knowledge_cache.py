"""
Knowledge Cache for TinyLlama Optimization
==========================================
FAISS-based cache for fast retrieval of similar questions/answers.
Reduces LLM inference by caching responses.
"""

import os
import json
import time
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
from datetime import datetime

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available, using numpy fallback")


@dataclass
class CacheEntry:
    """A cached question-answer pair."""
    question: str
    answer: str
    embedding: np.ndarray
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    hits: int = 0
    category: str = "general"
    axes: Dict[str, float] = field(default_factory=dict)


class KnowledgeCache:
    """
    Embedding-based knowledge cache for fast retrieval.
    
    Features:
    - FAISS index for fast similarity search
    - Configurable similarity threshold
    - Cache statistics tracking
    - Persistence to disk
    """
    
    def __init__(
        self,
        embedding_dim: int = 384,
        similarity_threshold: float = 0.85,
        cache_file: str = "knowledge_cache.json"
    ):
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.cache_file = cache_file
        
        # Cache storage
        self.entries: List[CacheEntry] = []
        self.embeddings: Optional[np.ndarray] = None
        
        # FAISS index
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        else:
            self.index = None
        
        # Statistics
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_search_time_ms": 0.0,
            "entries_count": 0
        }
        
        # Load existing cache
        self._load_cache()
        
        print(f"KnowledgeCache initialized: {len(self.entries)} entries, threshold={similarity_threshold}")
    
    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding for cosine similarity."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    
    def search(self, query_embedding: np.ndarray, query_text: str = "") -> Optional[Tuple[str, float]]:
        """
        Search for similar cached entries.
        
        Returns:
            Tuple of (cached_answer, similarity_score) if found, None otherwise.
        """
        start_time = time.time()
        self.stats["total_queries"] += 1
        
        if len(self.entries) == 0:
            self.stats["cache_misses"] += 1
            return None
        
        # Normalize query
        query_norm = self._normalize(query_embedding.astype(np.float32))
        
        if FAISS_AVAILABLE and self.index is not None and self.index.ntotal > 0:
            # FAISS search
            query_norm = query_norm.reshape(1, -1)
            distances, indices = self.index.search(query_norm, 1)
            
            best_score = float(distances[0][0])
            best_idx = int(indices[0][0])
        else:
            # Numpy fallback
            if self.embeddings is None or len(self.embeddings) == 0:
                self.stats["cache_misses"] += 1
                return None
            
            similarities = np.dot(self.embeddings, query_norm)
            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])
        
        # Update search time stats
        search_time = (time.time() - start_time) * 1000
        self.stats["avg_search_time_ms"] = (
            self.stats["avg_search_time_ms"] * 0.9 + search_time * 0.1
        )
        
        # Check threshold
        if best_score >= self.similarity_threshold:
            self.stats["cache_hits"] += 1
            self.entries[best_idx].hits += 1
            return (self.entries[best_idx].answer, best_score)
        
        self.stats["cache_misses"] += 1
        return None
    
    def add(
        self,
        question: str,
        answer: str,
        embedding: np.ndarray,
        category: str = "general",
        axes: Dict[str, float] = None
    ) -> None:
        """Add a new entry to the cache."""
        # Normalize embedding
        embedding_norm = self._normalize(embedding.astype(np.float32))
        
        # Create entry
        entry = CacheEntry(
            question=question,
            answer=answer,
            embedding=embedding_norm,
            category=category,
            axes=axes or {}
        )
        
        self.entries.append(entry)
        
        # Update FAISS index
        if FAISS_AVAILABLE and self.index is not None:
            self.index.add(embedding_norm.reshape(1, -1))
        
        # Update embeddings array
        if self.embeddings is None:
            self.embeddings = embedding_norm.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding_norm])
        
        self.stats["entries_count"] = len(self.entries)
        
        # Auto-save periodically
        if len(self.entries) % 10 == 0:
            self._save_cache()
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        hit_rate = 0.0
        if self.stats["total_queries"] > 0:
            hit_rate = self.stats["cache_hits"] / self.stats["total_queries"]
        
        return {
            **self.stats,
            "hit_rate": round(hit_rate, 3),
            "faiss_available": FAISS_AVAILABLE
        }
    
    def get_top_entries(self, n: int = 5) -> List[Dict]:
        """Get top N most-hit cache entries."""
        sorted_entries = sorted(self.entries, key=lambda e: e.hits, reverse=True)
        return [
            {
                "question": e.question[:80],
                "hits": e.hits,
                "category": e.category
            }
            for e in sorted_entries[:n]
        ]
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            data = {
                "entries": [
                    {
                        "question": e.question,
                        "answer": e.answer,
                        "embedding": e.embedding.tolist(),
                        "timestamp": e.timestamp,
                        "hits": e.hits,
                        "category": e.category,
                        "axes": e.axes
                    }
                    for e in self.entries
                ],
                "stats": self.stats
            }
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def _load_cache(self) -> None:
        """Load cache from disk."""
        if not os.path.exists(self.cache_file):
            return
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for entry_data in data.get("entries", []):
                embedding = np.array(entry_data["embedding"], dtype=np.float32)
                entry = CacheEntry(
                    question=entry_data["question"],
                    answer=entry_data["answer"],
                    embedding=embedding,
                    timestamp=entry_data.get("timestamp", ""),
                    hits=entry_data.get("hits", 0),
                    category=entry_data.get("category", "general"),
                    axes=entry_data.get("axes", {})
                )
                self.entries.append(entry)
                
                # Update FAISS index
                if FAISS_AVAILABLE and self.index is not None:
                    self.index.add(embedding.reshape(1, -1))
                
                # Update embeddings array
                if self.embeddings is None:
                    self.embeddings = embedding.reshape(1, -1)
                else:
                    self.embeddings = np.vstack([self.embeddings, embedding])
            
            self.stats = data.get("stats", self.stats)
            self.stats["entries_count"] = len(self.entries)
            
            print(f"Loaded {len(self.entries)} cache entries from disk")
        except Exception as e:
            print(f"Failed to load cache: {e}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.entries = []
        self.embeddings = None
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_search_time_ms": 0.0,
            "entries_count": 0
        }
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
