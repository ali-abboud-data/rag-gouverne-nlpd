"""Retrieval + generation with sources and refusal.

Rules:
- the model answers only from the supplied passages;
- chaque affirmation cite son article ;
- no answer in the passages: explicit refusal;
- post-hoc check: any citation of an article absent from the retrieved passages is flagged.

Lancer :  python -m src.rag.answer [--role juriste] "Quelles sont les sanctions en cas de violation du devoir d'informer ?"
Requires ANTHROPIC_API_KEY in .env.
"""
import re
import sys

from pydantic import BaseModel
from pydantic_ai import Agent

from . import config
from .search import search

SYSTEM = """Tu es un assistant qui répond aux questions sur la loi fédérale suisse sur la protection des données (LPD).
Réponds UNIQUEMENT à partir des passages fournis. N'utilise aucune autre connaissance.
Cite l'article pour chaque affirmation, sous la forme « (art. 24) ».
Si les passages ne permettent pas de répondre, mets refused=true et explique en une phrase que la réponse n'est pas dans les textes indexés.
Tu ne donnes pas de conseil juridique : tu restitues le contenu de la loi."""


REWRITE_SYSTEM = """Tu aides à chercher dans le texte de la loi fédérale suisse sur la protection des données (LPD).
Réécris la question de l'utilisateur en une seule phrase au vocabulaire de la loi, pour une recherche par le sens.
Exemples de termes de la loi : responsable du traitement, sous-traitant, personne concernée, données personnelles, droit d'accès, devoir d'informer, PFPDT, communication à l'étranger, violation de la sécurité des données.
Garde le sens exact de la question, n'ajoute aucun fait, ne réponds pas à la question."""


class Rewrite(BaseModel):
    query: str


class Answer(BaseModel):
    answer: str
    cited_articles: list[str]
    refused: bool


agent = Agent(config.LLM_MODEL, output_type=Answer, system_prompt=SYSTEM)
rewriter = Agent(config.LLM_MODEL, output_type=Rewrite, system_prompt=REWRITE_SYSTEM)


def _merge(primary: list[dict], secondary: list[dict], k: int) -> list[dict]:
    """Interleave two result lists (no duplicate passage) up to k passages."""
    merged, seen = [], set()
    for pair in zip(primary, secondary):
        for h in pair:
            key = (h["article"], h["part"])
            if key not in seen:
                seen.add(key)
                merged.append(h)
    return merged[:k]


def referenced_articles(text: str) -> set[str]:
    """Articles that the text itself cites ("art. 16, al. 2", "art. 25 à 27"), to validate internal cross-references.

    Paragraphs and letters ("al. 1 et 2", "let. b") are removed, then the list of numbers after "art." is read
    up to the first word that is neither "et" nor "à" ("de la loi du 22 juin 2007" is thus ignored).
    """
    found: set[str] = set()
    for m in re.finditer(r"\bart\.\s+", text):
        span = text[m.end(): m.end() + 90].split(";")[0]
        span = re.sub(r"\b(?:al|let|ch)\.\s*[\w]+(?:\s*(?:,|et|à)\s*(?:\d+[a-z]?|[a-z]\b))*", " ", span)
        lead = re.match(r"((?:\d+[a-z]{0,3}|[,\s]|\bet\b|\bà\b)+)", span)
        if not lead:
            continue
        chunk = lead.group(1)
        for a, b in re.findall(r"(\d+)\s+à\s+(\d+)", chunk):
            if int(a) < int(b) <= 74:
                found.update(str(n) for n in range(int(a), int(b) + 1))
        found.update(n for n in re.findall(r"\d+[a-z]{0,3}", chunk) if int(re.match(r"\d+", n).group()) <= 74)
    return found


def _normalize(ref: str) -> str:
    """"art. 60", "Art. 60 al. 1" or "60" -> "60" (the article number, with its letter if any)."""
    m = re.search(r"(\d+[a-z]{0,3})", ref)
    return m.group(1) if m else ref


def answer(question: str, role: str = "grand_public", k: int = 6, rewrite: bool = True) -> dict:
    rewritten, rewrite_tokens = None, {"input_tokens": 0, "output_tokens": 0}
    hits = search(question, k=k, role=role)
    if rewrite:
        try:
            rr = rewriter.run_sync(question)
            rewritten = rr.output.query
            rewrite_tokens = {"input_tokens": rr.usage.input_tokens or 0, "output_tokens": rr.usage.output_tokens or 0}
            hits = _merge(search(rewritten, k=k, role=role), hits, k)
        except Exception:  # if rewriting fails, keep the original search
            rewritten = None
    context = "\n\n".join(f"[Art. {h['article']} - {h['title']}]\n{h['text']}" for h in hits)
    result = agent.run_sync(f"Passages :\n{context}\n\nQuestion : {question}")
    out = result.output
    retrieved = {h["article"] for h in hits}
    cited = [_normalize(a) for a in out.cited_articles]
    referenced = set().union(*(referenced_articles(h["text"]) for h in hits)) if hits else set()
    unknown = [a for a in cited if a not in retrieved and a not in referenced]  # citation sans appui dans les passages
    indirect = [a for a in cited if a not in retrieved and a in referenced]  # cited via an internal cross-reference: legitimate
    try:
        u = result.usage
        usage = {"input_tokens": (u.input_tokens or 0) + rewrite_tokens["input_tokens"],
                 "output_tokens": (u.output_tokens or 0) + rewrite_tokens["output_tokens"]}
    except Exception:  # l'interface d'usage peut varier selon la version de pydantic-ai
        usage = {}
    return {
        "usage": usage,
        "role": role,
        "rewritten_query": rewritten,
        "answer": out.answer,
        "cited_articles": cited,
        "refused": out.refused,
        "unverified_citations": unknown,  # citations sans appui dans les passages : signal de risque
        "indirect_citations": indirect,   # articles cited via a cross-reference in a retrieved passage
        "retrieved": [(h["article"], round(h["score"], 3)) for h in hits],
    }


if __name__ == "__main__":
    args, role = sys.argv[1:], "grand_public"
    if args[:1] == ["--role"]:
        role, args = args[1], args[2:]
    r = answer(" ".join(args), role=role)
    print("Rewritten query:", r["rewritten_query"])
    print(r["answer"])
    print("\nCited articles:", r["cited_articles"], "| refused:", r["refused"])
    if r["unverified_citations"]:
        print("ATTENTION citations hors passages :", r["unverified_citations"])
    print("Retrieved passages:", r["retrieved"])
