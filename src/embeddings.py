from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class LocalEmbeddings:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True),
            dtype="float32",
        )

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(
            self.model.encode([text], show_progress_bar=False, normalize_embeddings=True),
            dtype="float32",
        )
