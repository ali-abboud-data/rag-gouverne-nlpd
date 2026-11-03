"""API tests without calling the model or Qdrant (the answer function is replaced by a stub).

Run:  python -m tests.test_api      (or: pytest, if installed)
"""
import os

from fastapi.testclient import TestClient

from src.rag import answer as answer_module
from src.rag.api import app

client = TestClient(app)


def _fake_answer(question, role="grand_public", k=6, rewrite=True):
    return {"role": role, "answer": f"answer for role {role}", "refused": False, "cited_articles": ["25"],
            "unverified_citations": [], "indirect_citations": [], "retrieved": [("25", 0.9)],
            "rewritten_query": None, "usage": {"input_tokens": 1, "output_tokens": 1}}


def setup_function(_=None):
    answer_module.answer = _fake_answer
    os.environ.pop("API_KEYS", None)


def test_dev_mode_default_role_is_grand_public():
    r = client.post("/ask", json={"question": "Quel est le but de la loi ?"})
    assert r.status_code == 200 and r.json()["role"] == "grand_public"


def test_dev_mode_accepts_known_role_and_rejects_unknown():
    assert client.post("/ask", json={"question": "Quel est le but ?", "role": "juriste"}).json()["role"] == "juriste"
    assert client.post("/ask", json={"question": "Quel est le but ?", "role": "admin"}).status_code == 422


def test_question_length_is_validated():
    assert client.post("/ask", json={"question": "?"}).status_code == 422
    assert client.post("/ask", json={"question": "x" * 501}).status_code == 422


def test_api_key_mode_requires_a_valid_key():
    os.environ["API_KEYS"] = "key-juriste:juriste,key-public:grand_public"
    assert client.post("/ask", json={"question": "Quel est le but ?"}).status_code == 401
    assert client.post("/ask", json={"question": "Quel est le but ?"}, headers={"X-API-Key": "wrong-key"}).status_code == 401
    ok = client.post("/ask", json={"question": "Quel est le but ?"}, headers={"X-API-Key": "key-juriste"})
    assert ok.status_code == 200 and ok.json()["role"] == "juriste"


def test_api_key_mode_ignores_the_role_claimed_in_the_body():
    os.environ["API_KEYS"] = "key-public:grand_public"
    r = client.post("/ask", json={"question": "Quel est le but ?", "role": "juriste"}, headers={"X-API-Key": "key-public"})
    assert r.status_code == 200 and r.json()["role"] == "grand_public"


def test_roles_endpoint_lists_audiences():
    assert client.get("/roles").json()["juriste"] == ["public", "technique"]


def test_model_failure_returns_502_without_details():
    def boom(*a, **k):
        raise RuntimeError("secret key sk-ant-xxx")
    answer_module.answer = boom
    r = client.post("/ask", json={"question": "Quel est le but ?"})
    assert r.status_code == 502 and "sk-ant" not in r.text


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        setup_function()
        fn()
        print("OK  ", name)
    print(f"{len(tests)} tests passed")
