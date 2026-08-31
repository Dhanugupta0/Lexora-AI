#!/usr/bin/env python3
"""Measure the pipeline instead of eyeballing it.

Runs a labelled question set against a freshly ingested corpus and reports the
metrics that actually matter for RAG:

    Recall@k        did the passage containing the answer make it into the
                    retrieved set at all? (the ceiling on everything downstream)
    MRR             how high did it rank?
    Precision@k     what fraction of what we retrieved was relevant
    Answer accuracy did the final answer contain the expected fact
    Groundedness    fraction of answers the verifier passed
    Abstention      did the system correctly decline the unanswerable questions
                    (a system that never abstains is hallucinating, not scoring)

Usage
    PYTHONPATH=. python scripts/evaluate.py
    PYTHONPATH=. python scripts/evaluate.py --dataset my_set.json --top-k 8
    PYTHONPATH=. python scripts/evaluate.py --compare      # hybrid vs dense vs sparse

The dataset is a JSON file: {"documents": [{"filename", "text"}],
"questions": [{"question", "expects": [...], "answerable": bool}]}.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BUILTIN_DATASET: Dict[str, Any] = {
    "documents": [
        {
            "filename": "acme_handbook.md",
            "text": """# Acme Corp Employee Handbook 2024

## Leave Policy
Full-time employees accrue 22 days of paid annual leave per calendar year.
Leave requests must be submitted at least 10 business days in advance.
Unused leave carries over up to a maximum of 5 days.

## Remote Work
Employees may work remotely up to 3 days per week with manager approval.
Fully remote arrangements require approval from the VP of People.

## Expenses
Expense claims must be filed within 30 days of the expense date.
Meals during business travel are reimbursed up to $75 per day.
Claims above $2,000 require pre-approval from a director.

## Security
All employees must enable two-factor authentication on corporate accounts.
Laptops must be encrypted using policy STD-441 before customer data is stored.
""",
        },
        {
            "filename": "vertex_annual_review.md",
            "text": """# Vertex Labs Annual Review FY2024

## Financials
Revenue reached 122.4 million euros, an increase of 19 percent year over year.
The hardware division contributed 71.8 million euros.
The services division contributed 50.6 million euros.

## Operations
Manufacturing yield improved to 94.2 percent after the Dresden retooling.
The Dresden retooling cost 18.3 million euros against a budget of 20 million.
Warranty returns fell to 0.8 percent.

## Governance
The audit committee is chaired by Dr. Ingrid Halvorsen.
The board approved resolution VX-2291 authorising a share buyback of up to 15 million euros.
""",
        },
    ],
    "questions": [
        {"question": "How many days of annual leave do full-time employees get?",
         "expects": ["22"], "relevant": ["Leave Policy"], "answerable": True},
        {"question": "How far in advance must leave be requested?",
         "expects": ["10"], "relevant": ["Leave Policy"], "answerable": True},
        {"question": "How many days a week can I work remotely?",
         "expects": ["3"], "relevant": ["Remote Work"], "answerable": True},
        {"question": "What is the daily meal reimbursement limit?",
         "expects": ["75"], "relevant": ["Expenses"], "answerable": True},
        {"question": "Above what amount does an expense claim need pre-approval?",
         "expects": ["2,000", "2000"], "relevant": ["Expenses"], "answerable": True},
        {"question": "What does policy STD-441 require?",
         "expects": ["encrypt"], "relevant": ["Security"], "answerable": True},
        {"question": "What was Vertex Labs revenue and how much did it grow?",
         "expects": ["122.4", "19"], "relevant": ["Financials"], "answerable": True},
        {"question": "How much did the hardware division contribute?",
         "expects": ["71.8"], "relevant": ["Financials"], "answerable": True},
        {"question": "What was the manufacturing yield after the Dresden retooling?",
         "expects": ["94.2"], "relevant": ["Operations"], "answerable": True},
        {"question": "What did the Dresden retooling cost against budget?",
         "expects": ["18.3", "20"], "relevant": ["Operations"], "answerable": True},
        {"question": "Who chairs the audit committee?",
         "expects": ["Halvorsen"], "relevant": ["Governance"], "answerable": True},
        {"question": "What does resolution VX-2291 authorise?",
         "expects": ["buyback", "15"], "relevant": ["Governance"], "answerable": True},
        # Unanswerable -- the system should abstain rather than invent an answer.
        {"question": "What is the parental leave allowance?", "expects": [], "answerable": False},
        {"question": "Who is the CEO of Vertex Labs?", "expects": [], "answerable": False},
        {"question": "What was Vertex Labs revenue in 2019?", "expects": [], "answerable": False},
    ],
}


def ingest_corpus(documents: List[Dict[str, str]]) -> List[str]:
    from app.database import init_db
    from app.rag.pipeline import ingest_document

    init_db()
    ids = []
    for spec in documents:
        doc_id = str(uuid.uuid4())
        suffix = Path(spec["filename"]).suffix.lstrip(".") or "md"
        result = ingest_document(doc_id, spec["text"].encode(), suffix, spec["filename"])
        ids.append(doc_id)
        print(f"  ingested {spec['filename']}: {result.chunk_count} chunks in {result.seconds}s")
    return ids


def contains_any(text: str, needles: List[str]) -> bool:
    lowered = re.sub(r"\s+", " ", text).lower()
    return any(n.lower() in lowered for n in needles)


def evaluate(dataset: Dict[str, Any], top_k: int, mode: str | None) -> Dict[str, Any]:
    from app.config import get_settings
    from app.rag.pipeline import answer_question

    settings = get_settings()
    original_mode = settings.RETRIEVAL_MODE
    if mode:
        settings.RETRIEVAL_MODE = mode

    questions = dataset["questions"]
    answerable = [q for q in questions if q.get("answerable", True)]

    hits = reciprocal_ranks = precision_sum = 0.0
    correct = grounded = abstained_correctly = 0
    latencies: List[int] = []
    rows: List[Dict[str, Any]] = []

    try:
        for spec in questions:
            started = time.time()
            result = answer_question(spec["question"], top_k=top_k, include_trace=False)
            latencies.append(int((time.time() - started) * 1000))

            row = {"question": spec["question"], "answerable": spec.get("answerable", True),
                   "confidence": result.confidence, "abstained": result.abstained}

            if not spec.get("answerable", True):
                # Success here means declining, not answering.
                declined = result.abstained or not contains_any(result.answer, [" "]) and (
                    result.abstained or result.confidence < 0.5 or _reads_as_refusal(result.answer)
                )
                abstained_correctly += 1 if declined else 0
                row["outcome"] = "declined" if declined else "HALLUCINATED"
                rows.append(row)
                continue

            wanted = [s.lower() for s in spec.get("relevant", [])]
            rank = None
            relevant_retrieved = 0
            for position, source in enumerate(result.sources, start=1):
                section = (source.section or "").lower()
                if wanted and any(w in section for w in wanted):
                    relevant_retrieved += 1
                    if rank is None:
                        rank = position

            if rank is not None:
                hits += 1
                reciprocal_ranks += 1.0 / rank
            if result.sources:
                precision_sum += relevant_retrieved / len(result.sources)

            answered = contains_any(result.answer, spec["expects"]) if spec["expects"] else False
            correct += 1 if answered else 0
            grounded += 1 if result.grounded else 0

            row.update(rank=rank, correct=answered, grounded=result.grounded)
            row["outcome"] = "ok" if answered else "MISS"
            rows.append(row)
    finally:
        settings.RETRIEVAL_MODE = original_mode

    n = max(len(answerable), 1)
    unanswerable = max(len(questions) - len(answerable), 1)
    latencies.sort()
    return {
        "mode": mode or original_mode,
        "questions": len(questions),
        f"recall@{top_k}": hits / n,
        "mrr": reciprocal_ranks / n,
        f"precision@{top_k}": precision_sum / n,
        "answer_accuracy": correct / n,
        "groundedness": grounded / n,
        "correct_abstention": abstained_correctly / unanswerable,
        "latency_p50_ms": latencies[len(latencies) // 2] if latencies else 0,
        "latency_p95_ms": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0,
        "rows": rows,
    }


_REFUSAL_WORDS = ("not contain", "no information", "could not find", "couldn't find",
                  "does not mention", "not mentioned", "not specified", "no passage",
                  "not provide", "unable to")


def _reads_as_refusal(answer: str) -> bool:
    return any(word in answer.lower() for word in _REFUSAL_WORDS)


def report(results: Dict[str, Any], verbose: bool) -> None:
    print(f"\n{'=' * 62}\n  RESULTS -- {results['mode']} retrieval\n{'=' * 62}")
    for key, value in results.items():
        if key in ("rows", "mode", "questions"):
            continue
        if isinstance(value, float):
            print(f"  {key:<22} {value:6.1%}" if value <= 1 else f"  {key:<22} {value:6.2f}")
        else:
            print(f"  {key:<22} {value:>6}")
    if verbose:
        print(f"\n  {'-' * 58}")
        for row in results["rows"]:
            mark = {"ok": "PASS", "declined": "PASS", "MISS": "FAIL",
                    "HALLUCINATED": "FAIL"}[row["outcome"]]
            rank = f"rank {row['rank']}" if row.get("rank") else "not retrieved" if row["answerable"] else "-"
            print(f"  [{mark}] {row['outcome']:<13} conf {row['confidence']:.2f}  {rank:<14} {row['question'][:52]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the LexoraAI RAG pipeline.")
    parser.add_argument("--dataset", type=Path, help="JSON dataset (defaults to the built-in set)")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--compare", action="store_true",
                        help="Run hybrid, dense and sparse retrieval and compare them")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Reuse whatever is already indexed")
    parser.add_argument("--quiet", action="store_true", help="Summary only")
    parser.add_argument("--json", type=Path, help="Also write the raw results here")
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text()) if args.dataset else BUILTIN_DATASET

    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set -- generation will run in extractive fallback mode.\n")

    if not args.skip_ingest:
        print("Ingesting evaluation corpus...")
        ingest_corpus(dataset["documents"])

    modes = ["hybrid", "dense", "sparse"] if args.compare else [None]
    all_results = []
    for mode in modes:
        results = evaluate(dataset, args.top_k, mode)
        report(results, verbose=not args.quiet)
        all_results.append(results)

    if args.compare:
        print(f"\n{'=' * 62}\n  COMPARISON\n{'=' * 62}")
        keys = [f"recall@{args.top_k}", "mrr", "answer_accuracy", "groundedness"]
        print(f"  {'mode':<10}" + "".join(f"{k:>18}" for k in keys))
        for results in all_results:
            print(f"  {results['mode']:<10}" + "".join(f"{results[k]:>17.1%}" for k in keys))

    if args.json:
        args.json.write_text(json.dumps(all_results, indent=2))
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
