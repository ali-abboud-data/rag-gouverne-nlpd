"""Fedlex PDF -> articles -> passages -> embeddings -> Qdrant.

Run (Docker and Qdrant started):  python -m src.rag.ingest
E5 models need the prefix "passage: " on documents and "query: " on questions.
"""
import re
from pathlib import Path

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from . import config
from .roles import audience_of
from .parse import parse

MAX_CHARS = 1400  # an E5 small model reads about 512 tokens; longer text is truncated


def split_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split a long article at sentence or paragraph ends, never mid-word."""
    if len(text) <= max_chars:
        return [text]
    parts, current = [], ""
    for piece in re.split(r"(?<=[.;:])\s+", text):
        if current and len(current) + len(piece) + 1 > max_chars:
            parts.append(current)
            current = piece
        else:
            current = f"{current} {piece}".strip()
    if current:
        parts.append(current)
    return parts


def build_passages(pdf: Path) -> list[dict]:
    passages = []
    for art in parse(pdf):
        for i, chunk in enumerate(split_text(art.text)):
            passages.append({
                "article": art.number,
                "title": art.title,
                "chapter": art.chapter,
                "page": art.page,
                "part": i + 1,
                "text": chunk,
                "audience": audience_of(art.number),
                "source": pdf.name,
            })
    return passages


def main() -> None:
    pdf = next(Path("data/raw").glob("*.pdf"))
    passages = build_passages(pdf)
    print(f"{len(passages)} passages issus de {pdf.name}")

    model = SentenceTransformer(config.EMBEDDING_MODEL)
    vectors = model.encode([f"passage: Art. {p['article']} {p['title']}. {p['text']}" for p in passages],
                           normalize_embeddings=True, show_progress_bar=True)

    client = QdrantClient(url=config.QDRANT_URL)
    if client.collection_exists(config.QDRANT_COLLECTION):
        client.delete_collection(config.QDRANT_COLLECTION)
    client.create_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.COSINE),
    )
    client.create_payload_index(config.QDRANT_COLLECTION, field_name="audience", field_schema=models.PayloadSchemaType.KEYWORD)
    client.upsert(
        collection_name=config.QDRANT_COLLECTION,
        points=[models.PointStruct(id=i, vector=v.tolist(), payload=p)
                for i, (v, p) in enumerate(zip(vectors, passages))],
    )
    print("Indexing done.")


if __name__ == "__main__":
    main()
