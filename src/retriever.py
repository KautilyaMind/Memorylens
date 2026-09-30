from __future__ import annotations

from typing import Any

from src.embeddings import LocalEmbeddings
from src.vectorstore import VectorStore


class DenseRetriever:
    def __init__(self, store: VectorStore, embeddings: LocalEmbeddings, top_k: int = 5):
        self.store = store
        self.embeddings = embeddings
        self.top_k = top_k

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            raise ValueError("Question cannot be empty")
        return self.store.search(self.embeddings.embed_query(query), self.top_k)
