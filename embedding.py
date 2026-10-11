"""EmbeddingManager: SentenceTransformer wrapper."""

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME


class EmbeddingManager:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model: SentenceTransformer | None = None

    def load_model(self) -> None:
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(
                f"Model loaded. Embedding dimension: "
                f"{self.model.get_embedding_dimension()}"
            )
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
