"""Retrieval only (no LLM), to check that the right articles come up.
Usage:  python -m src.rag.search [--role juriste] "Que doit faire une entreprise en cas de fuite de données ?"
"""
import sys
from functools import lru_cache

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from . import config
from .roles import qdrant_filter


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(config.EMBEDDING_MODEL)


def search(question: str, k: int = 5, role: str = "grand_public") -> list[dict]:
    model = _model()
    vector = model.encode(f"query: {question}", normalize_embeddings=True).tolist()
    client = QdrantClient(url=config.QDRANT_URL)
    hits = client.query_points(collection_name=config.QDRANT_COLLECTION, query=vector, query_filter=qdrant_filter(role), limit=k).points
    return [{"score": h.score, **h.payload} for h in hits]


if __name__ == "__main__":
    args = sys.argv[1:]
    role = "grand_public"
    if args[:1] == ["--role"]:
        role, args = args[1], args[2:]
    for r in search(" ".join(args), role=role):
        print(f"{r['score']:.3f}  Art. {r['article']} {r['title']}  (p.{r['page']})")
