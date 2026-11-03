"""Role-based access.

Each indexed passage carries an "audience" field; a role only sees the audiences open to it.
The filter is applied INSIDE the Qdrant query: a restricted passage is never retrieved, hence never sent to the LLM
(the model cannot cite or summarise it, even by mistake).

WARNING: the split below is a DEMONSTRATION policy. The law is entirely public;
two access levels are simulated here to show the mechanism, as one would do for internal documents.
  - public     : principles, rights of individuals, authority, sanctions (everything except the following)
  - technique  : internal compliance obligations (art. 10 to 18: adviser, codes of conduct, register,
                 certification, representative, disclosure abroad, publication; art. 22 and 23: impact assessment
                 and prior consultation)
"""
import re

from qdrant_client import models

TECHNIQUE = set(range(10, 19)) | {22, 23}

ROLES: dict[str, set[str]] = {
    "grand_public": {"public"},
    "juriste": {"public", "technique"},
}


def audience_of(article: str) -> str:
    n = int(re.match(r"\d+", article).group())
    return "technique" if n in TECHNIQUE else "public"


def allowed_audiences(role: str) -> set[str]:
    if role not in ROLES:
        raise ValueError(f"Unknown role: {role!r} (valid roles: {sorted(ROLES)})")
    return ROLES[role]


def can_see(role: str, article: str) -> bool:
    return audience_of(article) in allowed_audiences(role)


def qdrant_filter(role: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="audience", match=models.MatchAny(any=sorted(allowed_audiences(role))))])
