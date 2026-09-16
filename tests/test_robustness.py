import json
import unittest

from app.llm import MockResponse, safe_parse_message
from app.models.claim import (
    ClaimCase, Citation, DecisionStatus, KeyFinding, ValidationStatus,
)
from app.models.state import (
    ClaimState, CoverageFinding, CoverageFindings, DecisionDraft,
    InvestigationDimension,
)
from app.agents.case_analysis import CaseAnalysisAgent
from app.agents.decision import DecisionAgent
from app.agents.validation import ValidationAgent
from app.evaluation.evaluate import recall_at_k, citation_precision


def _case() -> ClaimCase:
    return ClaimCase.model_validate({
        "case_id": "TEST-001", "policy_id": "TEST", "policy_start_date": "2025-01-01",
        "claim_date": "2026-03-01", "sum_insured_inr": 500000,
        "continuous_coverage_months": 14, "patient": {"age": 30},
        "hospital": {"name": "Test Hospital", "network_provider": True},
        "treatment": {
            "type": "inpatient", "admission_hours": 24,
            "diagnosis": "Acute appendicitis", "procedure": "Appendectomy",
            "pre_existing": False, "experimental": False,
        },
        "expenses_inr": {}, "documents": ["claim_form"], "task": "Assess claim",
    })


def _state(case=None, coverage=None, retrieved=None) -> ClaimState:
    return {
        "case": case or _case(),
        "investigation_plan": None,
        "retrieved_evidence": retrieved or {},
        "coverage_findings": coverage,
        "decision_draft": None,
        "final_decision": None,
        "validation_result": None,
        "trace": [],
        "retry_count": 0,
        "error": None,
    }


class _FakeClient:
    def __init__(self, content: str):
        self.chat = _FakeChat(content)


class _FakeChat:
    def __init__(self, content: str):
        self.completions = _FakeCompletions(content)


class _FakeCompletions:
    def __init__(self, content: str):
        self._resp = MockResponse(content)

    def create(self, **kwargs):
        return self._resp


class SafeParseMessageTests(unittest.TestCase):
    def test_returns_empty_dict_for_non_json_string(self):
        self.assertEqual(safe_parse_message(MockResponse("totally-not-json")), {})

    def test_returns_empty_dict_for_json_list(self):
        self.assertEqual(safe_parse_message(MockResponse("[1, 2, 3]")), {})

    def test_returns_dict_for_valid_json_object(self):
        self.assertEqual(safe_parse_message(MockResponse(json.dumps({"a": 1}))), {"a": 1})


class CaseAnalysisGuardTests(unittest.TestCase):
    def _run(self, content: str):
        state = _state()
        CaseAnalysisAgent(_FakeClient(content)).analyze(state)
        return state["investigation_plan"]

    def test_filters_invalid_dimensions(self):
        plan = self._run(json.dumps({
            "dimensions": ["waiting_period", "nonexistent_dim"],
            "missing_fields": [], "checklist": [],
            "priority_order": ["nonexistent_dim"],
        }))
        self.assertEqual(plan.dimensions, [InvestigationDimension.WAITING_PERIOD])

    def test_defaults_to_coverage_scope_when_all_dims_invalid(self):
        plan = self._run(json.dumps({
            "dimensions": [], "missing_fields": [], "checklist": [],
            "priority_order": [],
        }))
        self.assertEqual(plan.dimensions, [InvestigationDimension.COVERAGE_SCOPE])

    def test_handles_malformed_json_without_crashing(self):
        plan = self._run("{bad json [")
        self.assertEqual(plan.dimensions, [InvestigationDimension.COVERAGE_SCOPE])


def _make_citation() -> Citation:
    return Citation(claim="", source="policy.pdf", page=1, section="TEST", chunk_id="c1")


class DecisionGuardTests(unittest.TestCase):
    def _run(self, content: str):
        finding = CoverageFinding(
            dimension=InvestigationDimension.COVERAGE_SCOPE,
            finding="Covered.", supported=True, citations=[_make_citation()],
        )
        state = _state(coverage=CoverageFindings(findings=[finding], overall_admissible=True))
        DecisionAgent(_FakeClient(content)).decide(state)
        return state["decision_draft"]

    def test_unknown_status_falls_back_to_needs_review(self):
        draft = self._run(json.dumps({
            "decision": "CLEARLY_INVALID_STATUS",
            "confidence": 0.9,
            "reasoning": "test",
            "key_findings": [],
            "applicable_limits": [],
            "missing_evidence": [],
        }))
        self.assertEqual(draft.decision, DecisionStatus.NEEDS_REVIEW)
        self.assertAlmostEqual(draft.confidence, 0.9)

    def test_malformed_json_abstains_and_promotes_coverage_findings(self):
        draft = self._run("not json at all")
        self.assertEqual(draft.decision, DecisionStatus.NEEDS_REVIEW)
        self.assertTrue(len(draft.key_findings) >= 1)


class ValidationGuardTests(unittest.TestCase):
    def _run(self, content: str):
        draft = DecisionDraft(
            decision=DecisionStatus.ADMISSIBLE,
            confidence=0.9,
            key_findings=[KeyFinding(finding="X", supported=True, citations=[_make_citation()])],
            applicable_limits=[],
            missing_evidence=[],
            reasoning="test",
        )
        state = _state()
        state["decision_draft"] = draft
        ValidationAgent(_FakeClient(content)).validate(state)
        return state["validation_result"]

    def test_unknown_status_falls_back_to_fail(self):
        val = self._run(json.dumps({"status": "UNKNOWN_STATUS", "unsupported_claims": []}))
        self.assertEqual(val.status, ValidationStatus.FAIL)


class RetrievalMetricTests(unittest.TestCase):
    def test_recall_at_k_covers_known_pages(self):
        result = recall_at_k([1, 2, 3, 4], [1, 2], [4])
        self.assertEqual(result[4], 1.0)

    def test_recall_at_k_zero_when_no_overlap(self):
        result = recall_at_k([5, 6], [1, 2], [2])
        self.assertEqual(result[2], 0.0)

    def test_citation_precision_half_correct(self):
        result = citation_precision(
            [type("_C", (), {"page": 1})(), type("_C", (), {"page": 3})()],
            [1, 2],
        )
        self.assertAlmostEqual(result, 0.5)

    def test_citation_precision_empty_citations(self):
        self.assertEqual(citation_precision([], [1, 2]), 0.0)

    def test_citation_precision_empty_gold(self):
        self.assertEqual(citation_precision([type("_C", (), {"page": 1})()], []), 0.0)


if __name__ == "__main__":
    unittest.main()
