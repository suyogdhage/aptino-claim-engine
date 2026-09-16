import json
import unittest

from app.agents.decision import DecisionAgent
from app.llm import MockLLM
from app.models.claim import ClaimCase, Citation, DecisionStatus
from app.models.state import (
    ClaimState, CoverageFinding, CoverageFindings, Evidence,
    InvestigationDimension,
)


class EvidenceContractTests(unittest.TestCase):
    def setUp(self):
        self.case = ClaimCase.model_validate({
            "case_id": "TEST-001", "policy_id": "TEST", "policy_start_date": "2025-01-01",
            "claim_date": "2026-03-01", "sum_insured_inr": 500000,
            "continuous_coverage_months": 14, "patient": {"age": 30},
            "hospital": {"name": "Test Hospital", "network_provider": True},
            "treatment": {"type": "inpatient", "admission_hours": 24,
                          "diagnosis": "Acute appendicitis", "procedure": "Appendectomy",
                          "pre_existing": False, "experimental": False},
            "expenses_inr": {}, "documents": ["claim_form"], "task": "Assess claim",
        })

    def test_decision_promotes_specialist_citations_when_provider_returns_no_findings(self):
        citation = Citation(claim="", source="policy.pdf", page=7,
                            section="WHAT WE COVER", chunk_id="evidence-1")
        finding = CoverageFinding(
            dimension=InvestigationDimension.COVERAGE_SCOPE,
            finding="Hospitalization is covered subject to policy terms.",
            supported=True, citations=[citation],
        )
        state: ClaimState = {
            "case": self.case, "investigation_plan": None, "retrieved_evidence": {},
            "coverage_findings": CoverageFindings(findings=[finding], overall_admissible=True),
            "decision_draft": None, "final_decision": None, "validation_result": None,
            "trace": [], "retry_count": 0, "error": None,
        }
        DecisionAgent(MockLLM()).decide(state)
        self.assertEqual(state["decision_draft"].decision, DecisionStatus.ADMISSIBLE)
        self.assertEqual(len(state["decision_draft"].key_findings), 1)
        self.assertEqual(state["decision_draft"].key_findings[0].citations[0].chunk_id, "evidence-1")

    def test_mock_validation_rejects_uncited_material_finding(self):
        prompt = """You are a Validation Agent
KEY FINDINGS:
[] A material claim
APPLICABLE LIMITS:
None
RETRIEVED EVIDENCE:
None"""
        result = json.loads(MockLLM().chat._respond(prompt))
        self.assertEqual(result["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
