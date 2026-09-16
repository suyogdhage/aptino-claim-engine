import time
from typing import Dict, List
from app.models.state import ClaimState, InvestigationPlan, InvestigationDimension, Evidence
from app.retrieval.hybrid_retriever import HybridRetriever


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


class PolicyEvidenceAgent:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def build_queries(self, plan: InvestigationPlan, case) -> Dict[InvestigationDimension, str]:
        queries = {}
        for dim in plan.dimensions:
            base_query = DIMENSION_QUERIES.get(dim, dim.value)
            case_context = f"Diagnosis: {case.treatment.diagnosis}. Procedure: {case.treatment.procedure}. "
            case_context += f"Treatment type: {case.treatment.type.value}. "
            case_context += f"Pre-existing: {case.treatment.pre_existing}. "
            case_context += f"Experimental: {case.treatment.experimental}. "
            case_context += f"Hospital: {case.hospital.name}, network: {case.hospital.network_provider}. "
            case_context += f"Admission hours: {case.treatment.admission_hours}. "
            case_context += f"Continuous coverage months: {case.continuous_coverage_months}. "
            if case.prior_policy:
                case_context += f"Prior policy: {case.prior_policy.continuous_years} years continuous. "
            queries[dim] = case_context + base_query
        return queries

    def retrieve(self, state: ClaimState) -> ClaimState:
        start = time.time()
        plan = state["investigation_plan"]
        case = state["case"]

        queries = self.build_queries(plan, case)
        retrieved = self.retriever.retrieve(queries)

        total_evidence = sum(len(v) for v in retrieved.values())

        state["retrieved_evidence"] = retrieved
        all_pages: List[int] = []
        for ev_list in retrieved.values():
            for ev in ev_list:
                all_pages.append(ev.page)

        state["trace"].append({
            "agent": "PolicyEvidence",
            "action": f"Retrieved evidence for {len(queries)} dimensions",
            "duration_ms": int((time.time() - start) * 1000),
            "evidence_count": total_evidence,
            "metadata": {
                "counts": {d.value: len(v) for d, v in retrieved.items()},
                "retrieved_pages": all_pages,
            }
        })

        return state


def policy_evidence_node(state: ClaimState, retriever: HybridRetriever) -> ClaimState:
    agent = PolicyEvidenceAgent(retriever)
    return agent.retrieve(state)