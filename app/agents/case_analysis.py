import time
import json
from typing import Dict, List, Any
from groq import Groq
from app.models.state import (
    ClaimState, InvestigationPlan, InvestigationDimension,
    ClaimCase, EvidenceContext, ExpenseTiming, PriorPolicy
)
from app.models.claim import TreatmentType
from app.config import settings
from app.llm import groq_json_completion


DIMENSION_QUERIES = {
    InvestigationDimension.WAITING_PERIOD: "initial waiting period 30 days pre-existing disease waiting period 24 48 months",
    InvestigationDimension.PRE_EXISTING: "pre-existing disease definition waiting period coverage exclusion",
    InvestigationDimension.COVERAGE_SCOPE: "scope of cover inpatient hospitalization day care domiciliary treatment covered expenses",
    InvestigationDimension.EXCLUSIONS: "exclusions cosmetic experimental unproven treatment not covered",
    InvestigationDimension.HOSPITAL_DEFINITION: "hospital definition registered minimum criteria 10 beds 24 hour nursing",
    InvestigationDimension.DOMICILIARY_CONDITIONS: "domiciliary treatment conditions home treatment hospital room unavailable patient cannot be moved",
    InvestigationDimension.DAY_CARE_QUALIFICATION: "day care treatment less than 24 hours hospitalization qualification",
    InvestigationDimension.CATEGORY_LIMITS: "sub-limit category limit specific disease cap room rent ICU limit",
    InvestigationDimension.PRE_POST_HOSPITALIZATION: "pre-hospitalization post-hospitalization expenses 30 days 60 days same condition",
    InvestigationDimension.PORTABILITY: "portability continuous coverage prior insurer waiting period reduction",
    InvestigationDimension.EXPERIMENTAL_TREATMENT: "experimental unproven investigational treatment exclusion",
    InvestigationDimension.EVIDENCE_SUFFICIENCY: "documents required claim form discharge summary itemized bill medical records",
}


class CaseAnalysisAgent:
    def __init__(self, groq_client: Groq, model: str = None):
        self.client = groq_client
        self.model = model or settings.GROQ_MODEL

    def _build_prompt(self, case: ClaimCase) -> str:
        case_json = case.model_dump_json(indent=2)
        return f"""You are a Claim Case Analysis Agent. Analyze the following health insurance claim case and create an investigation plan.

CLAIM CASE:
{case_json}

Your task:
1. Extract key facts from the case
2. Identify which decision dimensions are relevant (from the list below)
3. Detect any missing fields or evidence gaps
4. Create a prioritized investigation checklist

DECISION DIMENSIONS:
- waiting_period: Initial 30-day waiting period, pre-existing disease waiting periods (24/48 months)
- pre_existing: Whether condition is pre-existing and applicable waiting period
- coverage_scope: Whether treatment type (inpatient/day-care/domiciliary) is covered
- exclusions: Cosmetic, experimental, unproven, or other excluded treatments
- hospital_definition: Whether facility meets policy definition of Hospital
- domiciliary_conditions: Conditions for domiciliary treatment coverage
- day_care_qualification: Whether <24hr treatment qualifies as covered hospitalization
- category_limits: Disease-specific sub-limits, room rent caps, ICU limits
- pre_post_hospitalization: Pre/post hospitalization expense windows and conditions
- portability: Prior continuous coverage with another Indian insurer
- experimental_treatment: Whether treatment is experimental/unproven
- evidence_sufficiency: Whether submitted documents are sufficient for decision

Return ONLY a valid JSON object with this structure:
{{
  "dimensions": ["dimension1", "dimension2", ...],
  "missing_fields": ["field1", "field2", ...],
  "checklist": ["item1", "item2", ...],
  "priority_order": ["dimension1", "dimension2", ...]
}}

Order dimensions by priority - waiting periods and coverage scope first, then exclusions, then limits."""

    def analyze(self, state: ClaimState) -> ClaimState:
        start = time.time()
        case = state["case"]

        prompt = self._build_prompt(case)
        response = groq_json_completion(self.client, self.model, prompt,
                                        temperature=0.1, max_tokens=1500)

        result = json.loads(response.choices[0].message.content)

        dimensions = [InvestigationDimension(d) for d in result.get("dimensions", [])]
        priority_order = [InvestigationDimension(d) for d in result.get("priority_order", [])]

        plan = InvestigationPlan(
            dimensions=dimensions,
            missing_fields=result.get("missing_fields", []),
            checklist=result.get("checklist", []),
            priority_order=priority_order if priority_order else dimensions
        )

        state["investigation_plan"] = plan
        state["trace"].append({
            "agent": "CaseAnalysis",
            "action": f"Identified {len(dimensions)} investigation dimensions, {len(plan.missing_fields)} missing fields",
            "duration_ms": int((time.time() - start) * 1000),
            "evidence_count": 0,
            "metadata": {"dimensions": [d.value for d in dimensions]}
        })

        return state


def case_analysis_node(state: ClaimState, groq_client: Groq) -> ClaimState:
    agent = CaseAnalysisAgent(groq_client)
    return agent.analyze(state)