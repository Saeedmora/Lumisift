import numpy as np
import pickle
from typing import List, Tuple

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("Warning: faiss not found. Using simple Numpy search.")

class VectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.use_faiss = HAS_FAISS
        
        if self.use_faiss:
            self.index = faiss.IndexFlatL2(dimension)
        else:
            self.vectors = []
            
        self.ids = [] 
        self.id_map = {} 

    def add(self, embeddings: np.ndarray, doc_ids: List[str]):
        """
        Adds embeddings to the index.
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.dimension}, got {embeddings.shape[1]}")
        
        if self.use_faiss:
            start_idx = self.index.ntotal
            self.index.add(embeddings)
            for i, doc_id in enumerate(doc_ids):
                self.id_map[start_idx + i] = doc_id
        else:
            start_idx = len(self.vectors)
            for vec in embeddings:
                self.vectors.append(vec)
            for i, doc_id in enumerate(doc_ids):
                self.id_map[start_idx + i] = doc_id
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, List[str]]:
        """
        Searches for the k nearest neighbors.
        """
        if self.use_faiss:
            distances, indices = self.index.search(query_embedding, k)
            results_ids = []
            for idx_row in indices:
                row_ids = [self.id_map.get(idx) for idx in idx_row if idx != -1]
                results_ids.append(row_ids)
            return distances, results_ids[0]
        else:
            # Simple L2 search using numpy
            if not self.vectors:
                return np.array([]), []
            
            vecs = np.array(self.vectors)
            # Compute L2 distance: ||u - v||^2
            # Here query_embedding is (1, dim)
            diff = vecs - query_embedding
            dists = np.sum(diff**2, axis=1)
            
            # Get top k
            k = min(k, len(dists))
            indices = np.argsort(dists)[:k]
            
            results_ids = [self.id_map.get(idx) for idx in indices]
            return np.array([dists[indices]]), results_ids

    def save(self, filepath: str):
        if self.use_faiss:
            faiss.write_index(self.index, filepath)
        else:
            with open(filepath + ".npy", 'wb') as f:
                np.save(f, np.array(self.vectors))
                
        with open(filepath + ".map", 'wb') as f:
            pickle.dump(self.id_map, f)

    def load(self, filepath: str):
        if self.use_faiss:
            self.index = faiss.read_index(filepath)
        else:
            try:
                with open(filepath + ".npy", 'rb') as f:
                    self.vectors = list(np.load(f))
            except FileNotFoundError:
                pass
                
        with open(filepath + ".map", 'rb') as f:
            self.id_map = pickle.load(f)
