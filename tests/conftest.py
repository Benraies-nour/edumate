"""Fixtures communes : aucun appel réseau (ni Groq, ni Hugging Face)."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeEmbedder:
    """Embedding déterministe par sac de mots hachés : assez pour tester la recherche."""

    def embed(self, texts):
        for text in texts:
            vec = np.zeros(64, dtype=np.float32)
            for word in text.lower().split():
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1
            yield vec


@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch):
    import utils.vector_store as vs
    monkeypatch.setattr(vs, "_embedder", lambda name: FakeEmbedder())
