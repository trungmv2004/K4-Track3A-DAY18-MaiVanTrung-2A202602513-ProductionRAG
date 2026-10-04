from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os
import re
import sys
import unicodedata
from functools import lru_cache

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BM25_TOP_K,
    COLLECTION_NAME,
    DENSE_TOP_K,
    EMBEDDING_MODEL,
    HYBRID_TOP_K,
    QDRANT_HOST,
    QDRANT_PORT,
)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    from underthesea import word_tokenize

    return word_tokenize(unicodedata.normalize("NFC", text), format="text").replace("_", " ")


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", segment_vietnamese(text).casefold())


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi

        self.documents = list(chunks)
        self.corpus_tokens = [_tokens(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens) if any(self.corpus_tokens) else None
        if self.bm25 is not None:
            # Standard positive IDF prevents negative scores on tiny corpora.
            import math

            for term in self.bm25.idf:
                df = sum(term in tokens for tokens in self.corpus_tokens)
                self.bm25.idf[term] = math.log(1 + (len(chunks) - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or top_k <= 0 or not query.strip():
            return []
        scores = self.bm25.get_scores(_tokens(query))
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            SearchResult(
                self.documents[i]["text"], float(scores[i]), self.documents[i].get("metadata", {}), "bm25"
            )
            for i in indices
            if scores[i] > 0
        ]


@lru_cache(maxsize=2)
def _load_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer

    from config import MODEL_CACHE_DIR

    return SentenceTransformer(model_name, cache_folder=MODEL_CACHE_DIR)


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient

        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
            self.client.get_collections()
        except Exception:
            self.client = QdrantClient(":memory:")
            print("  Qdrant server unavailable: using in-memory Qdrant.", flush=True)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            self._encoder = _load_encoder(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, PointStruct, VectorParams

        encoder = self._get_encoder()
        dimension = encoder.get_sentence_embedding_dimension()
        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(
            collection, vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
        )
        for start in range(0, len(chunks), 64):
            batch = chunks[start : start + 64]
            vectors = encoder.encode(
                [c["text"] for c in batch], normalize_embeddings=True, batch_size=16, show_progress_bar=False
            )
            points = [
                PointStruct(
                    id=start + i,
                    vector=v.tolist(),
                    payload={"text": c["text"], "metadata": c.get("metadata", {})},
                )
                for i, (c, v) in enumerate(zip(batch, vectors))
            ]
            self.client.upsert(collection_name=collection, points=points, wait=True)

    def search(
        self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME
    ) -> list[SearchResult]:
        """Search using dense vectors."""
        if top_k <= 0 or not query.strip() or not self.client.collection_exists(collection):
            return []
        vector = self._get_encoder().encode(query, normalize_embeddings=True).tolist()
        response = self.client.query_points(collection_name=collection, query=vector, limit=top_k)
        return [
            SearchResult(pt.payload["text"], float(pt.score), pt.payload.get("metadata", {}), "dense")
            for pt in response.points
        ]


def reciprocal_rank_fusion(
    results_list: list[list[SearchResult]], k: int = 60, top_k: int = HYBRID_TOP_K
) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    if k < 0:
        raise ValueError("k must be non-negative")
    if top_k <= 0:
        return []
    scores, documents = {}, {}
    for results in results_list:
        seen = set()
        for rank, result in enumerate(results):
            identity = result.metadata.get("chunk_id") or (result.metadata.get("source", ""), result.text)
            if identity in seen:
                continue
            seen.add(identity)
            documents.setdefault(identity, result)
            scores[identity] = scores.get(identity, 0.0) + 1 / (k + rank + 1)
    ranked = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [
        SearchResult(documents[key].text, scores[key], documents[key].metadata, "hybrid") for key in ranked
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""

    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print("Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
