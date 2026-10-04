"""Index vectoriel FAISS avec embeddings locaux (fastembed, multilingue)."""
import logging
from functools import lru_cache

import faiss
import numpy as np
from fastembed import TextEmbedding

from utils import config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def _embedder(model_name: str) -> TextEmbedding:
    """Charge le modèle une seule fois par processus (~220 Mo)."""
    logger.info("Chargement du modèle d'embedding %s", model_name)
    return TextEmbedding(model_name=model_name)


class VectorStoreManager:
    """Index FAISS en similarité cosine (produit scalaire sur vecteurs normalisés)."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self.model_name = model_name
        self.index = None
        self.document_chunks: list[str] = []
        self.metadata_chunks: list[dict] = []

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        vectors = np.array(list(_embedder(self.model_name).embed(texts)), dtype=np.float32)
        faiss.normalize_L2(vectors)
        return vectors

    def build_index(self, chunks: list[dict]) -> None:
        texts, metas = [], []
        for chunk in chunks:
            text = chunk.get("content") or chunk.get("page_content") or ""
            if text.strip():
                texts.append(text)
                metas.append(chunk.get("metadata", {}))

        if not texts:
            logger.warning("Aucun chunk à indexer.")
            return

        logger.info("Génération des embeddings pour %d chunks...", len(texts))
        vectors = self.generate_embeddings(texts)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.document_chunks = texts
        self.metadata_chunks = metas
        logger.info("Index FAISS créé : %d vecteurs.", len(texts))

    def search(self, query: str, k: int = 10, min_score: float = 0.0) -> list[dict]:
        """Renvoie les chunks les plus proches, triés par score cosine décroissant."""
        if self.index is None or not self.document_chunks:
            return []

        query_vec = self.generate_embeddings([query])
        scores, indices = self.index.search(query_vec, min(k, len(self.document_chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < min_score:
                continue
            results.append({
                "content": self.document_chunks[idx],
                "metadata": self.metadata_chunks[idx],
                "score": float(score),
            })
        return results
