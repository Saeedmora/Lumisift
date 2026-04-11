import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence_transformers not found. Using Mock Embeddings.")

class EmbeddingService:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.use_mock = not HAS_SENTENCE_TRANSFORMERS
        if not self.use_mock:
            print(f"Loading embedding model: {model_name}...")
            try:
                self.model = SentenceTransformer(model_name)
                print("Model loaded.")
            except Exception as e:
                print(f"Failed to load model: {e}. Switching to Mock mode.")
                self.use_mock = True
        
        if self.use_mock:
            self.dim = 384

    def embed(self, text: str) -> np.ndarray:
        """
        Generates an embedding for the given text.
        """
        if self.use_mock:
            # Deterministic mock embedding based on hash
            np.random.seed(hash(text) % (2**32))
            return np.random.rand(self.dim).astype(np.float32)
        return self.model.encode(text)

    def embed_many(self, texts: list[str]) -> np.ndarray:
        """
        Generates embeddings for a list of texts.
        """
        if self.use_mock:
            return np.array([self.embed(t) for t in texts])
        return self.model.encode(texts)
