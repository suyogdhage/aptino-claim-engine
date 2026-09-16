import json
import os
import sys
import time
from typing import Dict, List, Any, Optional
from pydantic import ValidationError

from app.models.claim import ClaimCase, DecisionResponse
from app.graph.workflow import ClaimEngine
from app.config import settings
from app.evaluation.expected import EXPECTED_OUTCOMES_FILE, GOLD_EVIDENCE_FILE, ExpectedOutcome, GoldEvidence


def load_cases(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_decision(d: str) -> str:
    return d.upper().strip()


def recall_at_k(retrieved_pages: List[int], gold_pages: List[int], k_values: List[int]) -> Dict[int, float]:
    """Fraction of gold pages covered among the top-k retrieved pages."""
    gold = set(gold_pages)
    if not gold:
        return {}
    result = {}
    for k in sorted(k_values):
        top_k = set(retrieved_pages[:k])
        result[k] = round(len(top_k & gold) / len(gold), 3)
    return result


def citation_precision(citations: List, gold_pages: List[int]) -> float:
    """Fraction of cited pages that are among the gold pages for the case."""
    gold = set(gold_pages)
    if not citations:
        return 0.0
    if not gold:
        return 0.0
    correct = sum(1 for c in citations if c.page in gold)
    return round(correct / len(citations), 3)


def evaluate_single(
    engine: ClaimEngine,
    case_data: dict,
    expected: ExpectedOutcome,
    gold: Optional[GoldEvidence] = None,
) -> dict:
    case_id = case_data.get("case_id", "UNKNOWN")
    result: DecisionResponse = engine.analyze(ClaimCase(**case_data))

    predicted = normalize_decision(result.decision.value)
    expected_dec = normalize_decision(expected.decision)
    correct = predicted == expected_dec

    # Abstention: NEEDS_REVIEW matches expected NEEDS_REVIEW
    abstained = predicted == "NEEDS_REVIEW"

    # Retrieval metrics from trace
    retrieval_counts = {}
    retrieved_pages: List[int] = []
    for t in result.trace:
        if t.agent == "PolicyEvidence":
            metadata = t.metadata if isinstance(t.metadata, dict) else {}
            retrieval_counts = metadata.get("counts", metadata)
            retrieved_pages = metadata.get("retrieved_pages", [])

    total_evidence = sum(retrieval_counts.values()) if retrieval_counts else 0

    citation_count = len(result.citations)
    gold_pages = gold.gold_pages if gold else []
    retrieval_recall_at_k = recall_at_k(retrieved_pages, gold_pages, [1, 2, 4, 8])
    citation_hit_rate = citation_precision(result.citations, gold_pages)

    return {
        "case_id": case_id,
        "predicted_decision": predicted,
        "expected_decision": expected_dec,
        "correct": correct,
        "confidence": result.confidence,
        "confidence_ok": result.confidence >= expected.confidence_min,
        "abstained": abstained,
        "evidence_per_dimension": retrieval_counts,
        "total_evidence_retrieved": total_evidence,
        "retrieved_pages": retrieved_pages,
        "gold_pages": gold_pages,
        "retrieval_recall_at_k": retrieval_recall_at_k,
        "citation_count": citation_count,
        "citation_hit_rate": citation_hit_rate,
        "citation_precision": citation_hit_rate,
        "validation_status": result.validation.status.value,
        "unsupported_claims": result.validation.unsupported_claims,
        "missing_evidence": result.missing_evidence,
        "trace": [t.model_dump() for t in result.trace],
        "full_result": result.model_dump()
    }


def run_evaluation(data_dir: str = "./data") -> dict:
    engine = ClaimEngine()
    with open(EXPECTED_OUTCOMES_FILE, "r") as f:
        expected_map = {k: ExpectedOutcome(**v) for k, v in json.load(f).items()}

    from app.evaluation.expected import load_gold_evidence
    gold_map = load_gold_evidence()

    case_sets = [
        ("public", os.path.join(data_dir, "public_test_cases.json")),
        ("custom", os.path.join(data_dir, "custom_test_cases.json")),
    ]

    all_results = []
    per_set = {}

    for set_name, case_path in case_sets:
        if not os.path.exists(case_path):
            continue
        cases = load_cases(case_path)
        set_results = []
        for case_data in cases:
            case_id = case_data.get("case_id")
            expected = expected_map.get(case_id)
            if not expected:
                print(f"[SKIP] {case_id}: no expected outcome defined")
                continue
            print(f"[RUN ] {case_id}")
            try:
                res = evaluate_single(engine, case_data, expected, gold_map.get(case_id))
            except Exception as e:
                print(f"[ERR ] {case_id}: {e}")
                status_code = getattr(e, "status_code", None)
                skipped = status_code == 429 or "rate_limit" in str(e).lower()
                res = {
                    "case_id": case_id,
                    "predicted_decision": "ERROR",
                    "expected_decision": expected.decision,
                    "correct": False,
                    "confidence": 0.0,
                    "confidence_ok": False,
                    "abstained": False,
                    "evidence_per_dimension": {},
                    "total_evidence_retrieved": 0,
                    "retrieved_pages": [],
                    "gold_pages": gold_map.get(case_id).gold_pages if gold_map.get(case_id) else [],
                    "retrieval_recall_at_k": {},
                    "citation_count": 0,
                    "citation_hit_rate": 0.0,
                    "citation_precision": 0.0,
                    "validation_status": "FAIL",
                    "unsupported_claims": [],
                    "missing_evidence": [],
                    "trace": [],
                    "full_result": None,
                    "error": str(e),
                    "skipped": skipped,
                }
            set_results.append(res)
            all_results.append(res)
        per_set[set_name] = set_results

    # Aggregate metrics
    evaluated_results = [r for r in all_results if not r.get("skipped", False)]
    total = len(evaluated_results)
    correct = sum(1 for r in evaluated_results if r["correct"])
    abstained = sum(1 for r in evaluated_results if r["abstained"])
    cites = sum(1 for r in evaluated_results if r["citation_count"] > 0)
    avg_conf = sum(r["confidence"] for r in evaluated_results) / max(total, 1)

    # Identify abstention cases (cases where expected is NEEDS_REVIEW)
    expected_abstain = [
        r["case_id"] for r in all_results
        if r["expected_decision"] == "NEEDS_REVIEW"
    ]
    abstained_correctly = [
        r["case_id"] for r in all_results
        if r["expected_decision"] == "NEEDS_REVIEW" and r["predicted_decision"] == "NEEDS_REVIEW"
    ]

    # Aggregate retrieval + citation metrics
    with_gold = [r for r in evaluated_results if r.get("gold_pages")]
    recall_at_1 = recall_at_2 = recall_at_4 = recall_at_8 = 0.0
    cite_precisions = []
    if with_gold:
        recall_at_1 = round(sum(r["retrieval_recall_at_k"].get(1, 0.0) for r in with_gold) / len(with_gold), 3)
        recall_at_2 = round(sum(r["retrieval_recall_at_k"].get(2, 0.0) for r in with_gold) / len(with_gold), 3)
        recall_at_4 = round(sum(r["retrieval_recall_at_k"].get(4, 0.0) for r in with_gold) / len(with_gold), 3)
        recall_at_8 = round(sum(r["retrieval_recall_at_k"].get(8, 0.0) for r in with_gold) / len(with_gold), 3)
        cite_precisions = [r["citation_precision"] for r in evaluated_results]

    citation_precision_avg = round(sum(cite_precisions) / max(len(cite_precisions), 1), 3) if cite_precisions else 0.0

    summary = {
        "total_cases": total,
        "skipped_cases": [r["case_id"] for r in all_results if r.get("skipped", False)],
        "correct_decisions": correct,
        "accuracy": round(correct / max(total, 1), 3),
        "avg_confidence": round(avg_conf, 3),
        "abstention_rate": round(abstained / max(total, 1), 3),
        "expected_abstain_cases": expected_abstain,
        "abstained_correctly": abstained_correctly,
        "citation_coverage": round(cites / max(total, 1), 3),
        "citation_precision": citation_precision_avg,
        "retrieval_recall_at_1": recall_at_1,
        "retrieval_recall_at_2": recall_at_2,
        "retrieval_recall_at_4": recall_at_4,
        "retrieval_recall_at_8": recall_at_8,
        "per_set": {k: {
            "total": len([r for r in v if not r.get("skipped", False)]),
            "skipped": sum(1 for r in v if r.get("skipped", False)),
            "correct": sum(1 for r in v if r["correct"] and not r.get("skipped", False)),
            "accuracy": round(sum(1 for r in v if r["correct"] and not r.get("skipped", False)) / max(len([r for r in v if not r.get("skipped", False)]), 1), 3)
        } for k, v in per_set.items()}
    }

    report = {"summary": summary, "results": all_results}

    os.makedirs("evaluation_results", exist_ok=True)
    report_path = os.path.join("evaluation_results", "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps(summary, indent=2))
    print(f"Report written to {report_path}")
    return summary


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    run_evaluation(data_dir)
