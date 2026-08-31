"""Verify the answer against the passages it claims to come from.

Telling a model "only use the context" reduces hallucination. It does not
eliminate it. This module checks the output rather than trusting the prompt:

1. **Citation validity.** Parse every `[S#]` marker. Any pointing at a passage
   number that was never in the prompt is a fabricated citation -- the most
   damaging failure mode, because it looks verified. Those get stripped.

2. **Per-claim support.** Split the answer into claim sentences and score each
   against its cited passages with a hybrid signal: embedding cosine (catches
   paraphrase) blended with weighted lexical overlap that leans on numbers,
   dates and proper nouns (catches the digit-swap error embeddings miss).

3. **Optional LLM adjudication.** Claims that land in the ambiguous band get a
   second opinion from an NLI-style verifier call.

4. **Abstention.** If overall support falls under the floor, the answer is
   replaced with an honest "I couldn't verify this in your documents" instead
   of being shipped with a confident tone.

Every stage fails open in the safe direction: if verification itself breaks, the
answer is returned unmodified and flagged `verified: false` rather than dropped.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.config import get_settings
from app.rag import prompts
from app.rag.chunker import split_sentences
from app.rag.keyword import tokenize
from app.rag.llm import get_llm

logger = logging.getLogger(__name__)
settings = get_settings()

CITATION = re.compile(r"\[S(\d+)\]")
# Models reach for full-width or CJK brackets often enough that we normalise
# them before parsing -- an unrecognised citation reads as an uncited claim and
# would wrongly tank the confidence score.
_BRACKET_ALIASES = str.maketrans({"\u3010": "[", "\u3011": "]", "\uff3b": "[", "\uff3d": "]",
                                  "\u2039": "[", "\u203a": "]"})
# Requires an explicit S/Source marker so ordinary parentheticals like "(2024)"
# are never mistaken for citations.
_LOOSE_CITATION = re.compile(
    r"[\[(]\s*(?:S|Source|source|SOURCE)\s*\.?\s*(\d+(?:\s*[,;&]\s*(?:S|Source|source)?\s*\.?\s*\d+)*)\s*[\])]"
)
# Tokens where an exact match matters far more than a semantic one.
_HIGH_VALUE = re.compile(r"^(\d[\d,.%/-]*|[A-Z][A-Za-z0-9.-]{2,})$")

ABSTENTION_TEXT = (
    "I could not verify an answer to this in your uploaded documents. "
    "The passages I retrieved touch on the topic but do not actually state the answer, "
    "so I would rather tell you that than guess.\n\n"
    "You could try rephrasing the question with wording closer to the document, narrowing it "
    "to a specific document, or uploading the file that covers it."
)


_REFUSAL = re.compile(
    r"\b(could\s*n[o']?t|cannot|can\s*not|do(?:es)?\s*n[o']?t|no|not)\b[^.]{0,60}?"
    r"\b(find|found|contain|mention|state|specify|include|appear|available|provided)\b",
    re.I,
)


def looks_like_refusal(answer: str) -> bool:
    """True when the model honestly reported that the context lacks the answer.

    Such a reply has no citations by design, so the claim checker would score it
    as unsupported and abstain over it -- replacing a specific, useful non-answer
    with a generic one.
    """
    text = (answer or "").strip()
    return bool(text) and len(text) < 700 and not parse_citations(text) and bool(_REFUSAL.search(text))


@dataclass
class ClaimCheck:
    text: str
    citations: List[int]
    support: float
    verdict: str                     # supported | partial | unsupported
    uncited: bool = False            # no [S#] marker, so the reader cannot check it


@dataclass
class GroundingReport:
    verified: bool = False
    confidence: float = 0.0          # 0..1 -- overall faithfulness
    support_ratio: float = 0.0       # fraction of claims judged supported
    claims: List[ClaimCheck] = field(default_factory=list)
    cited_sources: List[int] = field(default_factory=list)
    invalid_citations: List[int] = field(default_factory=list)
    uncited_claims: int = 0
    abstained: bool = False
    method: str = "none"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verified": self.verified,
            "confidence": round(self.confidence, 3),
            "support_ratio": round(self.support_ratio, 3),
            "claims_checked": len(self.claims),
            "unsupported_claims": sum(1 for c in self.claims if c.verdict == "unsupported"),
            "uncited_claims": self.uncited_claims,
            "cited_sources": self.cited_sources,
            "invalid_citations": self.invalid_citations,
            "abstained": self.abstained,
            "method": self.method,
            "notes": self.notes,
        }


# --------------------------------------------------------------------------- #
# Citation handling
# --------------------------------------------------------------------------- #
def normalize_citations(text: str) -> str:
    """Rewrite every citation variant a model might emit into canonical `[S#]`.

    Handles full-width brackets, `(S2)`, `[Source 3]` and grouped forms like
    `[S1, S2]`, which would otherwise be silently dropped by the strict parser.
    """
    if not text:
        return text
    text = text.translate(_BRACKET_ALIASES)

    def rewrite(match: re.Match) -> str:
        numbers = re.findall(r"\d+", match.group(1))
        if not numbers:
            return match.group(0)
        return "".join(f"[S{int(n)}]" for n in numbers)

    text = _LOOSE_CITATION.sub(rewrite, text)
    # Models routinely write "... year over year. [S1]" -- the citation belongs
    # to that sentence, so pull it inside the terminator before we split claims.
    text = re.sub(r"([.!?])[ \t]*((?:\[S\d+\])+)", r" \2\1", text)
    text = re.sub(r"(?<=[^\s(\[])(\[S\d+\])", r" \1", text)     # always space before a citation
    return re.sub(r"[ \t]{2,}", " ", text)


def parse_citations(text: str) -> List[int]:
    return sorted({int(n) for n in CITATION.findall(text or "")})


def strip_invalid_citations(answer: str, valid_count: int) -> tuple[str, List[int]]:
    """Remove `[S#]` markers pointing outside the passages we actually supplied."""
    invalid: List[int] = []

    def replace(match: re.Match) -> str:
        number = int(match.group(1))
        if 1 <= number <= valid_count:
            return match.group(0)
        invalid.append(number)
        return ""

    cleaned = CITATION.sub(replace, answer or "")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    return cleaned.strip(), sorted(set(invalid))


def split_claims(answer: str) -> List[str]:
    """Sentences that assert something -- headings, bullets markers and
    conversational filler are not worth fact-checking."""
    claims: List[str] = []
    for line in (answer or "").splitlines():
        line = re.sub(r"^\s*([-*•]|\d+[.)])\s+", "", line).strip()
        if not line or line.startswith("#"):
            continue
        for sentence in split_sentences(line):
            sentence = sentence.strip()
            words = re.findall(r"[A-Za-z0-9]+", CITATION.sub("", sentence))
            if len(words) >= 3 and len(sentence) >= 12:
                claims.append(sentence)
    return claims


# --------------------------------------------------------------------------- #
# Support scoring
# --------------------------------------------------------------------------- #
def _lexical_support(claim: str, evidence: str) -> float:
    """Weighted overlap -- numbers and proper nouns count several times over."""
    claim_tokens = tokenize(CITATION.sub("", claim))
    if not claim_tokens:
        return 0.0
    evidence_tokens = set(tokenize(evidence))
    if not evidence_tokens:
        return 0.0

    # Map each normalised token back to its surface form so we can spot
    # numbers and capitalised entities, which must match exactly.
    surface = {}
    for word in CITATION.sub("", claim).split():
        parts = tokenize(word)
        if parts:
            surface.setdefault(parts[0], word)

    matched = weight_total = 0.0
    for token in set(claim_tokens):
        original = surface.get(token, token)
        weight = 3.0 if _HIGH_VALUE.match(original.strip(".,;:()")) else 1.0
        weight_total += weight
        if token in evidence_tokens:
            matched += weight
    return matched / weight_total if weight_total else 0.0


def _semantic_support(claims: Sequence[str], evidences: Sequence[str]) -> List[float]:
    try:
        from app.rag.embedder import get_embedder
        embedder = get_embedder()
        # Claim vs evidence is a symmetric comparison -- both sides are
        # statements, so neither should be embedded as a search query.
        vectors = embedder.embed_similarity([CITATION.sub("", c) for c in claims] + list(evidences))
    except Exception as exc:                                 # noqa: BLE001
        logger.warning("Semantic grounding unavailable (%s) -- lexical only", exc)
        return [0.0] * len(claims)

    claim_vectors = vectors[: len(claims)]
    evidence_vectors = vectors[len(claims) :]

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return 0.0 if na == 0 or nb == 0 else dot / (na * nb)

    return [max((cosine(cv, ev) for ev in evidence_vectors), default=0.0) for cv in claim_vectors]


# --------------------------------------------------------------------------- #
# Verifier
# --------------------------------------------------------------------------- #
def verify_answer(
    answer: str,
    passages: Sequence[str],
    *,
    use_llm: bool = False,
) -> GroundingReport:
    """Score how faithfully `answer` follows `passages`."""
    report = GroundingReport()
    if not answer or not passages:
        report.notes.append("nothing to verify")
        return report

    report.cited_sources = [n for n in parse_citations(answer) if 1 <= n <= len(passages)]

    claims = split_claims(answer)
    if not claims:
        report.notes.append("no checkable claims")
        report.verified = True
        report.confidence = 0.6
        report.support_ratio = 1.0
        report.method = "trivial"
        return report

    # Pair every claim with the passages it cites (or all of them, if uncited).
    evidence_per_claim: List[List[str]] = []
    citations_per_claim: List[List[int]] = []
    for claim in claims:
        cited = [n for n in parse_citations(claim) if 1 <= n <= len(passages)]
        citations_per_claim.append(cited)
        evidence_per_claim.append([passages[n - 1] for n in cited] if cited else list(passages))

    semantic = _semantic_support(claims, list(passages))
    flat_evidence = ["\n".join(ev) for ev in evidence_per_claim]
    semantic_targeted = _semantic_support(claims, flat_evidence) if any(citations_per_claim) else semantic

    checks: List[ClaimCheck] = []
    for i, claim in enumerate(claims):
        evidence = "\n".join(evidence_per_claim[i])
        lexical = _lexical_support(claim, evidence)
        # targeted cosine is per-claim-vs-its-own-evidence; fall back to global max
        sem = max(semantic_targeted[i] if i < len(semantic_targeted) else 0.0,
                  semantic[i] if i < len(semantic) else 0.0)
        support = 0.55 * sem + 0.45 * lexical

        if support >= settings.GROUNDING_MIN_SUPPORT + 0.15:
            verdict = "supported"
        elif support >= settings.GROUNDING_MIN_SUPPORT:
            verdict = "partial"
        else:
            verdict = "unsupported"

        if not citations_per_claim[i]:
            # The evidence may well back it, but an uncited claim is unverifiable
            # by the reader, so it can never be rated better than "partial".
            support *= 0.9
            verdict = "partial" if verdict == "supported" else verdict

        checks.append(ClaimCheck(claim, citations_per_claim[i], round(support, 3), verdict,
                                 uncited=not citations_per_claim[i]))

    report.method = "hybrid-lexical-semantic"

    # Second opinion on the borderline ones only -- one extra call at most.
    if use_llm:
        ambiguous = [i for i, c in enumerate(checks) if c.verdict in ("partial", "unsupported")]
        if ambiguous:
            verdicts = _llm_verdicts(
                [checks[i].text for i in ambiguous],
                ["\n".join(evidence_per_claim[i]) for i in ambiguous],
            )
            for position, index in enumerate(ambiguous):
                verdict = verdicts.get(position + 1)
                if verdict:
                    checks[index].verdict = verdict
                    checks[index].support = max(
                        checks[index].support,
                        {"supported": 0.85, "partial": 0.55, "unsupported": 0.1}[verdict],
                    )
            report.method = "hybrid+llm-nli"

    report.claims = checks
    report.uncited_claims = sum(1 for c in checks if c.uncited)
    solid = sum(1 for c in checks if c.verdict == "supported")
    partial = sum(1 for c in checks if c.verdict == "partial")
    report.support_ratio = (solid + 0.5 * partial) / len(checks)

    citation_health = 1.0 if report.cited_sources else 0.55
    mean_support = sum(c.support for c in checks) / len(checks)
    report.confidence = round(
        min(1.0, 0.5 * report.support_ratio + 0.35 * mean_support + 0.15 * citation_health), 3
    )
    report.verified = report.support_ratio >= 0.5 and report.confidence >= settings.ABSTAIN_THRESHOLD
    return report


def _llm_verdicts(claims: Sequence[str], evidences: Sequence[str]) -> Dict[int, str]:
    llm = get_llm()
    if not llm.available:
        return {}

    evidence_block = "\n\n".join(
        f"[E{i}] {CITATION.sub('', e)[:900]}" for i, e in enumerate(evidences, start=1)
    )
    claim_block = "\n".join(f"{i}. {CITATION.sub('', c)}" for i, c in enumerate(claims, start=1))

    payload = llm.complete_json(
        [
            {"role": "system", "content": prompts.VERIFY_SYSTEM},
            {"role": "user", "content": prompts.VERIFY_USER.format(
                evidence=evidence_block, claims=claim_block)},
        ],
        temperature=0.0,
        max_tokens=900,
        reasoning_effort="low",
        tag="verify",
        default={},
    )

    out: Dict[int, str] = {}
    for entry in payload.get("verdicts") or []:
        if not isinstance(entry, dict):
            continue
        try:
            index = int(entry.get("id", 0))
        except (TypeError, ValueError):
            continue
        verdict = str(entry.get("verdict", "")).strip().lower()
        if verdict in ("supported", "partial", "unsupported"):
            out[index] = verdict
    return out


def apply_grounding(
    answer: str,
    passages: Sequence[str],
    *,
    use_llm: bool = False,
) -> tuple[str, GroundingReport]:
    """Clean citations, verify, and abstain when support is too weak."""
    if not settings.ENABLE_GROUNDING_CHECK:
        return answer, GroundingReport(verified=True, confidence=0.5, method="disabled")

    answer = normalize_citations(answer)

    if looks_like_refusal(answer):
        report = GroundingReport(verified=True, confidence=0.5, support_ratio=1.0,
                                 method="refusal")
        report.notes.append("model reported the answer is not in the context")
        return answer, report

    invalid: List[int] = []
    if settings.STRIP_INVALID_CITATIONS:
        answer, invalid = strip_invalid_citations(answer, len(passages))

    try:
        report = verify_answer(answer, passages, use_llm=use_llm)
    except Exception as exc:                                 # noqa: BLE001
        logger.warning("Grounding check failed (%s) -- returning the answer unverified", exc)
        report = GroundingReport(verified=False, confidence=0.0, method="failed")
        report.notes.append(str(exc))
        report.invalid_citations = invalid
        return answer, report

    report.invalid_citations = invalid
    if invalid:
        report.notes.append(f"removed {len(invalid)} fabricated citation(s)")
        report.confidence = max(0.0, report.confidence - 0.1 * len(invalid))

    if report.confidence < settings.ABSTAIN_THRESHOLD:
        report.abstained = True
        report.notes.append(
            f"confidence {report.confidence:.2f} below the {settings.ABSTAIN_THRESHOLD:.2f} floor"
        )
        return ABSTENTION_TEXT, report

    return answer, report
