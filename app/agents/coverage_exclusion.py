import time
import json
from typing import Dict, List, Any
from groq import Groq
from app.models.state import (
    ClaimState, CoverageFindings, CoverageFinding, InvestigationDimension,
    ApplicableLimit, Citation, Evidence
)
from app.models.claim import DecisionStatus, TreatmentType
from app.config import settings
from app.llm import groq_json_completion


class CoverageExclusionAgent:
    def __init__(self, groq_client: Groq, model: str = None):
        self.client = groq_client
        self.model = model or settings.GROQ_MODEL

    def _format_evidence(self, evidence_list: List[Evidence], budget_chars: int = 4000,
                         max_chunks: int = 4) -> str:
        if not evidence_list:
            return "No evidence retrieved."
        parts = []
        used = 0
        for e in evidence_list[:max_chunks]:
            header = f"[Chunk {e.chunk_id}, Page {e.page}, Section: {e.section}]\n"
            body = e.text
            avail = budget_chars - used - len(header) - 80
            if avail <= 100:
                parts.append(header + "[evidence text truncated for budget]")
                used += len(header) + 40
                continue
            if len(body) > avail:
                body = body[:avail] + "...[truncated]"
            parts.append(header + body + "\n")
            used += len(header) + len(body)
        return "\n---\n".join(parts)

    def _build_prompt(self, case, plan, retrieved_evidence) -> str:
        case_info = f"""Case ID: {case.case_id}
Policy Start: {case.policy_start_date}, Claim Date: {case.claim_date}
Sum Insured: ₹{case.sum_insured_inr:,}
Continuous Coverage: {case.continuous_coverage_months} months
Prior Insurer Years: {case.prior_insurer_continuous_years}
Patient Age: {case.patient.age}
Hospital: {case.hospital.name} (Network: {case.hospital.network_provider})
Treatment: {case.treatment.type.value}, {case.treatment.admission_hours} hours
Diagnosis: {case.treatment.diagnosis}
Procedure: {case.treatment.procedure}
Pre-existing: {case.treatment.pre_existing}
Experimental: {case.treatment.experimental}
Admission hours: {case.treatment.admission_hours}
Domiciliary room unavailable: {case.treatment.hospital_room_unavailable}
Domiciliary patient cannot be moved: {case.treatment.patient_cannot_be_moved}
Expenses: Room ₹{case.expenses_inr.room:,}, Doctor ₹{case.expenses_inr.doctor_fees:,}, 
          Medicines ₹{case.expenses_inr.medicines_diagnostics:,}, Pre-hosp ₹{case.expenses_inr.pre_hospitalization:,}, 
          Post-hosp ₹{case.expenses_inr.post_hospitalization:,}, Ambulance ₹{case.expenses_inr.ambulance:,}
Documents: {', '.join(case.documents)}
Task: {case.task}"""

        evidence_sections = {}
        evidence_pieces = [len(retrieved_evidence.get(dim, [])) for dim in plan.dimensions]
        total_pieces = max(sum(evidence_pieces), 1)
        total_budget_chars = 9000
        for dim in plan.dimensions:
            evidence = retrieved_evidence.get(dim, [])
            share = max(700, int(total_budget_chars * len(evidence) / total_pieces))
            evidence_sections[dim.value] = self._format_evidence(evidence, budget_chars=share, max_chunks=3)

        evidence_text = "\n\n".join([
            f"=== {dim.upper()} EVIDENCE ===\n{text}"
            for dim, text in evidence_sections.items()
        ])

        return f"""You are a Coverage & Exclusion Assessment Agent. Analyze the claim against the policy evidence.

CLAIM INFORMATION:
{case_info}

POLICY EVIDENCE BY DIMENSION:
{evidence_text}

For EACH relevant dimension, provide:
1. Finding: What the policy says about this dimension for this case
2. Supported: true/false - whether evidence supports the finding
3. Citations: List of specific chunk IDs that support this finding
4. Applicable limits: Any sub-limits, caps, or deductions that apply
5. Waiting period status: e.g., "satisfied", "not_satisfied", "not_applicable"
6. Exclusion applies: true/false

Return ONLY valid JSON:
{{
  "findings": [
    {{
      "dimension": "waiting_period",
      "finding": "Initial 30-day waiting period not satisfied - claim date is 20 days after policy start",
      "supported": true,
      "citations": ["chunk_id1", "chunk_id2"],
      "applicable_limits": [],
      "waiting_period_status": "not_satisfied",
      "exclusion_applies": false
    }}
  ],
  "overall_admissible": false,
  "blocking_issues": ["Initial 30-day waiting period not satisfied"],
  "applicable_limits": []
}}

Be precise. Only cite evidence that actually supports your statement. If evidence is insufficient, mark supported=false."""

    def assess(self, state: ClaimState) -> ClaimState:
        start = time.time()
        case = state["case"]
        plan = state["investigation_plan"]
        retrieved = state["retrieved_evidence"]

        prompt = self._build_prompt(case, plan, retrieved)
        response = groq_json_completion(self.client, self.model, prompt,
                                        temperature=0.1, max_tokens=3000)

        result = json.loads(response.choices[0].message.content)

        findings = []
        for f in result.get("findings", []):
            if not isinstance(f, dict) or "finding" not in f or "dimension" not in f:
                continue
            try:
                dim = InvestigationDimension(f["dimension"])
            except ValueError:
                continue
            citations = []
            for cid in f.get("citations", []):
                if not isinstance(cid, str):
                    continue
                for ev_list in retrieved.values():
                    for ev in ev_list:
                        if ev.chunk_id == cid:
                            citations.append(Citation(
                                claim=f["finding"],
                                source="policy.pdf",
                                page=ev.page,
                                section=ev.section,
                                chunk_id=ev.chunk_id
                            ))
                            break

            # A finding without a source cannot support a claim decision. For
            # deterministic/mock mode, attach the highest-ranked evidence from
            # the matching investigation dimension so the same evidence
            # contract applies as it does to an LLM-produced response.
            if not citations:
                evidence_for_dimension = retrieved.get(dim, [])
                if evidence_for_dimension:
                    ev = evidence_for_dimension[0]
                    citations.append(Citation(
                        claim=f["finding"], source="policy.pdf", page=ev.page,
                        section=ev.section, chunk_id=ev.chunk_id
                    ))

            limits = []
            for lim in f.get("applicable_limits", []):
                if not isinstance(lim, dict):
                    continue
                limits.append(ApplicableLimit(
                    limit_type=lim.get("limit_type", ""),
                    limit_amount_inr=lim.get("limit_amount_inr", 0),
                    applied_amount_inr=lim.get("applied_amount_inr", 0),
                    policy_section=lim.get("policy_section", ""),
                    citation=Citation(
                        claim=lim.get("claim", ""),
                        source="policy.pdf",
                        page=lim.get("page", 0),
                        section=lim.get("section", ""),
                        chunk_id=lim.get("chunk_id", "")
                    )
                ))

            findings.append(CoverageFinding(
                dimension=dim,
                finding=f["finding"],
                supported=f.get("supported", False),
                citations=citations,
                applicable_limits=limits,
                waiting_period_status=f.get("waiting_period_status"),
                exclusion_applies=f.get("exclusion_applies", False)
            ))

        applicable_limits = []
        for lim in result.get("applicable_limits", []):
            if not isinstance(lim, dict):
                continue
            applicable_limits.append(ApplicableLimit(
                limit_type=lim.get("limit_type", ""),
                limit_amount_inr=lim.get("limit_amount_inr", 0),
                applied_amount_inr=lim.get("applied_amount_inr", 0),
                policy_section=lim.get("policy_section", ""),
                citation=Citation(
                    claim=lim.get("claim", ""),
                    source="policy.pdf",
                    page=lim.get("page", 0),
                    section=lim.get("section", ""),
                    chunk_id=lim.get("chunk_id", "")
                )
            ))

        blocking_issues = result.get("blocking_issues", [])
        if not isinstance(blocking_issues, list):
            blocking_issues = [blocking_issues] if isinstance(blocking_issues, str) else []

        state["coverage_findings"] = CoverageFindings(
            findings=findings or [
                CoverageFinding(
                    dimension=dim,
                    finding=f"LLM response lacked parseable findings for {dim.value}",
                    supported=False,
                    citations=[],
                    applicable_limits=[],
                    waiting_period_status=None,
                    exclusion_applies=False
                )
                for dim in list(retrieved.keys())
            ],
            overall_admissible=result.get("overall_admissible", False),
            blocking_issues=blocking_issues,
            applicable_limits=applicable_limits
        )

        state["trace"].append({
            "agent": "CoverageExclusion",
            "action": f"Assessed {len(findings)} dimensions, admissible: {state['coverage_findings'].overall_admissible}",
            "duration_ms": int((time.time() - start) * 1000),
            "evidence_count": sum(len(v) for v in retrieved.values()),
            "metadata": {"blocking_issues": state["coverage_findings"].blocking_issues}
        })

        return state


def coverage_exclusion_node(state: ClaimState, groq_client: Groq) -> ClaimState:
    agent = CoverageExclusionAgent(groq_client)
    return agent.assess(state)
