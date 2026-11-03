"""Evaluate the system on eval/questions.yaml, once or several times.

Measures (per run, then mean and spread over several runs):
- retrieval_hit : an expected article is among the retrieved passages (retrieval quality) ;
- citation_hit  : an expected article is cited AND the system did not refuse (generation quality) ;
- refus : wrongful refusals on an answerable question; correct refusals on out-of-scope questions;
- unsupported_citations : articles cited without support in the passages (neither retrieved nor cited by a passage);
- indirect_citations : articles cited via an internal cross-reference of a retrieved passage (legitimate);
- access_violations : retrieved passages the role should not have seen (expected: 0);
- duration and tokens (the cost in francs is not computed: prices to check with the provider).

Usage :  python -m eval.run_eval [--runs 3] [--limit N] [--no-rewrite]
Writes eval/results/<date>-<variant>.json and eval/results/latest-<variant>.md.
Never edits the README: copy the figures by hand, after reading them.
"""
import argparse
import json
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

import yaml

from src.rag import config
from src.rag.answer import answer
from src.rag.roles import can_see


def expected_articles(value) -> set[str]:
    return set(re.findall(r"\d+[a-z]{0,3}", str(value or "")))


def run_once(questions: list[dict], rewrite: bool, label: str) -> list[dict]:
    rows = []
    for q in questions:
        role = q.get("role", "grand_public")
        t0 = time.time()
        try:
            r, err = answer(q["question"], role=role, rewrite=rewrite), None
        except Exception as e:  # an API error must not stop the whole evaluation
            r, err = None, f"{type(e).__name__}: {e}"
        row = {"id": q["id"], "question": q["question"], "role": role, "should_refuse": q["should_refuse"],
               "seconds": round(time.time() - t0, 2), "error": err}
        if r:
            retrieved = {a for a, _ in r["retrieved"]}
            cited = set(r["cited_articles"])
            exp = expected_articles(q.get("expected_article"))
            row.update(
                refused=r["refused"],
                refusal_ok=(r["refused"] == q["should_refuse"]),
                retrieval_hit=bool(exp & retrieved) if exp else None,
                citation_hit=(bool(exp & cited) and not r["refused"]) if exp else None,
                unsupported=r["unverified_citations"],
                indirect=r.get("indirect_citations", []),
                access_violations=[a for a in retrieved if not can_see(role, a)],
                cited=sorted(cited),
                usage=r.get("usage", {}),
                rewritten_query=r.get("rewritten_query"),
                answer=r["answer"],
            )
        rows.append(row)
        ok = row.get("refusal_ok") and row.get("citation_hit") in (True, None)
        print(f"[{label}] {q['id']} {'ERREUR' if err else ('OK' if ok else 'À VOIR'):7} {row['seconds']:5.1f}s  {q['question'][:60]}")
    return rows


def metrics(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r["error"]]
    ans = [r for r in ok if not r["should_refuse"]]
    ref = [r for r in ok if r["should_refuse"]]
    pct = lambda n, d: round(100 * n / d, 1) if d else None
    return {
        "errors": len(rows) - len(ok),
        "retrieval_hit_pct": pct(sum(bool(r["retrieval_hit"]) for r in ans), len(ans)),
        "citation_hit_pct": pct(sum(bool(r["citation_hit"]) for r in ans), len(ans)),
        "wrong_refusals": sum(bool(r["refused"]) for r in ans),
        "correct_refusal_pct": pct(sum(bool(r["refused"]) for r in ref), len(ref)),
        "unsupported_citations": sum(len(r["unsupported"]) for r in ok),
        "indirect_citations": sum(len(r["indirect"]) for r in ok),
        "access_violations": sum(len(r["access_violations"]) for r in ok),
        "avg_seconds": round(sum(r["seconds"] for r in ok) / len(ok), 2) if ok else None,
        "input_tokens": sum((r.get("usage") or {}).get("input_tokens") or 0 for r in ok),
        "output_tokens": sum((r.get("usage") or {}).get("output_tokens") or 0 for r in ok),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1, help="number of runs (mean and spread)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-rewrite", action="store_true", help="disable query rewriting (the \"before\" measurement)")
    args = ap.parse_args()

    questions = yaml.safe_load(Path("eval/questions.yaml").read_text(encoding="utf8"))
    if args.limit:
        questions = questions[: args.limit]

    runs = [run_once(questions, not args.no_rewrite, f"{i + 1}/{args.runs}") for i in range(args.runs)]
    per_run = [metrics(r) for r in runs]

    agg = {}
    for key in per_run[0]:
        vals = [m[key] for m in per_run if m[key] is not None]
        agg[key] = {"mean": round(statistics.mean(vals), 2), "min": min(vals), "max": max(vals)} if vals else None

    unstable = []
    for i, q in enumerate(questions):
        outcomes = []
        for rows in runs:
            r = rows[i]
            outcomes.append(bool(not r["error"] and r.get("refusal_ok") and r.get("citation_hit") in (True, None)))
        if len(set(outcomes)) > 1 or not all(outcomes):
            unstable.append({"id": q["id"], "question": q["question"], "ok_in_runs": f"{sum(outcomes)}/{len(outcomes)}"})

    summary = {
        "date": datetime.now().isoformat(timespec="minutes"),
        "llm_model": config.LLM_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "query_rewriting": not args.no_rewrite,
        "temperature": "not settable: the model does not accept this parameter (checked), hence the mean over several runs",
        "questions": len(questions),
        "runs": args.runs,
        "metrics": agg,
        "questions_to_examine": unstable,
    }
    variant = "sans-reformulation" if args.no_rewrite else "reformulation"
    out = Path("eval/results")
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M") + f"-{variant}-{args.runs}run"
    (out / f"{stamp}.json").write_text(json.dumps({"summary": summary, "per_run": per_run, "runs": runs}, ensure_ascii=False, indent=2), encoding="utf8")

    lines = [f"# Results ({summary['date']})", "",
             f"Model: {summary['llm_model']} | query rewriting: {summary['query_rewriting']} | "
             f"{summary['questions']} questions | {args.runs} run(s)", "",
             "| Measure | Mean | Min | Max |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['mean']} | {v['min']} | {v['max']} |" for k, v in agg.items() if v]
    lines += ["", "## Questions to review (failed or unstable)", ""]
    lines += [f"- {u['id']} : {u['question']} (passed in {u['ok_in_runs']} runs)" for u in unstable] or ["- none"]
    (out / f"latest-{variant}.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
