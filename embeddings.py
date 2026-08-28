import hashlib
import re
from collections import Counter

import numpy as np
from langchain_core.embeddings import Embeddings


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


class LocalHashEmbeddings(Embeddings):
    """Deterministic local embeddings for small assessment PDFs."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        words = [token.lower() for token in TOKEN_PATTERN.findall(text)]
        bigrams = [f"{left}_{right}" for left, right in zip(words, words[1:])]
        tokens = words + bigrams
        counts = Counter(tokens)

        for token, count in counts.items():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            vector[index] += 1.0 + np.log1p(count)

        norm = np.linalg.norm(vector)
        if norm:
            vector = vector / norm

        return vector.tolist()
