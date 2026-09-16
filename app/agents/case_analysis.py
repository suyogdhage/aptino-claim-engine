import time
from groq import Groq
from app.models.state import (
    ClaimState, InvestigationPlan, InvestigationDimension,
    ClaimCase, EvidenceContext, ExpenseTiming, PriorPolicy
)
from app.config import settings
from app.llm import groq_json_completion, safe_parse_message


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

        result = safe_parse_message(response)

        def parse_dimensions(raw) -> list:
            if not isinstance(raw, list):
                return []
            dims = []
            for d in raw:
                if isinstance(d, str):
                    try:
                        dims.append(InvestigationDimension(d))
                    except ValueError:
                        continue
            return dims

        dimensions = parse_dimensions(result.get("dimensions"))
        priority_order = parse_dimensions(result.get("priority_order"))

        # Never run the pipeline with a null investigation plan; fall back to
        # the coverage-scope dimension when the provider output is unusable.
        if not dimensions:
            dimensions = [InvestigationDimension.COVERAGE_SCOPE]
        if not priority_order:
            priority_order = dimensions

        plan = InvestigationPlan(
            dimensions=dimensions,
            missing_fields=result.get("missing_fields", [])
            if isinstance(result.get("missing_fields"), list) else [],
            checklist=result.get("checklist", [])
            if isinstance(result.get("checklist"), list) else [],
            priority_order=priority_order
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