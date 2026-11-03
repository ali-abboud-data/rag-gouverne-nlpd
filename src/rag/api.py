"""API FastAPI.

Lancer :  uvicorn src.rag.api:app --reload      puis ouvrir http://localhost:8000/docs

Caller role:
- if the API_KEYS environment variable is set ("key1:juriste,key2:grand_public"), the role is DERIVED from the key
  sent in the X-API-Key header; the "role" field of the request body is then ignored
  (a caller cannot grant itself a wider role);
- otherwise (development mode), the role is read from the request body, "grand_public" by default.

Logging: role, duration, refusal, tokens and question length. The question text is never written to the logs.
"""
import hmac
import logging
import os
import time

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from . import answer as answer_module
from . import config
from .roles import ROLES

log = logging.getLogger("rag.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

app = FastAPI(
    title="Governed RAG - FADP",
    description="Sourced answers on the Swiss Federal Act on Data Protection, with refusal and role-based access.",
)


def _api_keys() -> dict[str, str]:
    """Read API_KEYS ("key:role,key:role") on every call; reject an unknown role."""
    keys: dict[str, str] = {}
    for item in filter(None, (s.strip() for s in os.getenv("API_KEYS", "").split(","))):
        key, _, role = item.partition(":")
        if role not in ROLES:
            raise RuntimeError(f"API_KEYS: unknown role {role!r} (valid roles: {sorted(ROLES)})")
        keys[key] = role
    return keys


def resolve_role(body_role: str | None, api_key: str | None) -> str:
    keys = _api_keys()
    if keys:
        match = next((role for k, role in keys.items() if api_key and hmac.compare_digest(k, api_key)), None)
        if match is None:
            raise HTTPException(status_code=401, detail="Missing or invalid API key (X-API-Key header).")
        return match
    role = body_role or "grand_public"
    if role not in ROLES:
        raise HTTPException(status_code=422, detail=f"Unknown role: {role!r}. Valid roles: {sorted(ROLES)}.")
    return role


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    role: str | None = Field(default=None, description="Ignored if API keys are configured (the role then comes from the key).")
    rewrite: bool = Field(default=True, description="Reformuler la question avec le vocabulaire de la loi avant la recherche.")


class Passage(BaseModel):
    article: str
    score: float


class AskResponse(BaseModel):
    role: str
    answer: str
    refused: bool
    cited_articles: list[str]
    unverified_citations: list[str]
    indirect_citations: list[str]
    retrieved: list[Passage]
    rewritten_query: str | None
    usage: dict


@app.get("/health")
def health() -> dict:
    try:
        qdrant_ok = QdrantClient(url=config.QDRANT_URL).collection_exists(config.QDRANT_COLLECTION)
    except Exception:
        qdrant_ok = False
    return {"status": "ok" if qdrant_ok else "degraded", "qdrant_collection": qdrant_ok,
            "auth": "api-key" if _api_keys() else "dev (role read from the request body)"}


@app.get("/roles")
def roles() -> dict:
    return {role: sorted(audiences) for role, audiences in ROLES.items()}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, x_api_key: str | None = Header(default=None)) -> AskResponse:
    role = resolve_role(req.role, x_api_key)
    t0 = time.time()
    try:  # def (not async def): FastAPI runs this blocking call in a separate thread
        r = answer_module.answer(req.question, role=role, rewrite=req.rewrite)
    except Exception:
        log.exception("answer failed (role=%s)", role)
        raise HTTPException(status_code=502, detail="Internal error: model or database unavailable.")
    log.info("ask role=%s seconds=%.2f refused=%s question_chars=%d tokens=%s",
             role, time.time() - t0, r["refused"], len(req.question), r.get("usage"))
    return AskResponse(
        role=role, answer=r["answer"], refused=r["refused"], cited_articles=r["cited_articles"],
        unverified_citations=r["unverified_citations"], indirect_citations=r.get("indirect_citations", []),
        retrieved=[Passage(article=a, score=s) for a, s in r["retrieved"]],
        rewritten_query=r.get("rewritten_query"), usage=r.get("usage", {}),
    )
