import time
import json
from typing import Dict, List
from groq import Groq
from app.models.state import (
    ClaimState, DecisionDraft, CoverageFindings, InvestigationDimension,
    KeyFinding, ApplicableLimit, Citation, Evidence
)
from app.models.claim import DecisionStatus
from app.config import settings
from app.llm import groq_json_completion, safe_parse_message


class DecisionAgent:
    def __init__(self, groq_client: Groq, model: str = None):
        self.client = groq_client
        self.model = model or settings.GROQ_MODEL

    def _build_prompt(self, state: ClaimState) -> str:
        case = state["case"]
        coverage = state["coverage_findings"]

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
Admission hours: {case.treatment.admission_hours}
Domiciliary room unavailable: {case.treatment.hospital_room_unavailable}
Domiciliary patient cannot be moved: {case.treatment.patient_cannot_be_moved}
Pre-existing: {case.treatment.pre_existing}
Experimental: {case.treatment.experimental}
Patient age: {case.patient.age}
Expenses: Room ₹{case.expenses_inr.room:,}, Doctor ₹{case.expenses_inr.doctor_fees:,}, Medicines ₹{case.expenses_inr.medicines_diagnostics:,}, Pre-hosp ₹{case.expenses_inr.pre_hospitalization:,}, Post-hosp ₹{case.expenses_inr.post_hospitalization:,}, Ambulance ₹{case.expenses_inr.ambulance:,}
Total Expenses: ₹{sum(case.expenses_inr.model_dump().values()):,}"""
        if case.evidence_context is not None:
            ev = case.evidence_context
            case_info += (f"\nEvidence: hospital_registered={ev.hospital_registered}, "
                          f"medical_necessity_confirmed={ev.medical_necessity_confirmed}")
        if case.expense_timing is not None:
            case_info += (
                f"\nPre/Post same condition: {case.expense_timing.same_condition_confirmed}"
            )

        findings_text = "\n".join([
            f"- [{f.dimension.value}] supported={f.supported} wp={f.waiting_period_status} excl={f.exclusion_applies}: {f.finding}"
            + (f" CITED_CHUNKS=[{', '.join(c.chunk_id for c in f.citations)}]" if f.citations else "")
            for f in coverage.findings
        ])

        evidence_text = self._format_evidence(state.get("retrieved_evidence", {}), budget_chars=4200)

        limits_text = "\n".join([
            f"- {l.limit_type}: cap ₹{l.limit_amount_inr:,}, applied ₹{l.applied_amount_inr:,} ({l.policy_section})"
            for l in coverage.applicable_limits
        ]) if coverage.applicable_limits else "None"

        blocking_text = "\n".join([
            f"- {b}" for b in coverage.blocking_issues
        ]) if coverage.blocking_issues else "None"

        return f"""You are a Decision Agent. Combine the coverage findings into a final claim decision.

CLAIM:
{case_info}

COVERAGE FINDINGS:
{findings_text}

BLOCKING ISSUES:
{blocking_text}

APPLICABLE LIMITS:
{limits_text}

AVAILABLE POLICY EVIDENCE:
{evidence_text}

DECISION STATUSES:
- ADMISSIBLE: Fully covered, no material limit/deduction
- ADMISSIBLE_WITH_LIMITS: Covered but limits/caps/waiting-period effects reduce payable
- PARTIALLY_ADMISSIBLE: Part of claim not supported
- NOT_ADMISSIBLE: Policy supports rejection/exclusion
- NEEDS_REVIEW: Evidence insufficient for safe final decision

Return ONLY valid JSON:
{{
  "decision": "ADMISSIBLE_WITH_LIMITS",
  "confidence": 0.85,
  "reasoning": "Concise explanation of the decision logic",
  "key_findings": [
    {{
      "finding": "Statement about the claim",
      "supported": true,
      "citations": ["chunk_id"]
    }}
  ],
  "applicable_limits": [
    {{
      "limit_type": "Disease-specific sub-limit",
      "limit_amount_inr": 250000,
      "applied_amount_inr": 250000,
      "policy_section": "e.g., Specific Diseases",
      "chunk_id": "chunk_id"
    }}
  ],
  "missing_evidence": ["item requested but not provided"]
}}

Rules:
- Assign NEEDS_REVIEW when blocking_issues indicate uncertainty or evidence gaps.
- Every key finding and limit must reference at least one chunk_id from AVAILABLE POLICY EVIDENCE (only those "Chunk" ids).
- Confidence reflects certainty, not optimism."""

    def _format_evidence(self, retrieved_evidence: Dict, budget_chars: int = 4200) -> str:
        parts = []
        used = 0
        for dim, ev_list in retrieved_evidence.items():
            header = f"=== {dim.value.upper()} ===\n"
            if used + len(header) > budget_chars:
                break
            parts.append(header)
            used += len(header)
            for e in ev_list[:3]:
                block = f"[Chunk {e.chunk_id}] Page {e.page}, {e.section}: {e.text[:220]}"
                if len(block) > 240:
                    block = block[:240] + "..."
                if used + len(block) > budget_chars:
                    break
                parts.append(block)
                used += len(block)
        return "\n".join(parts) if parts else "No evidence retrieved."

    def decide(self, state: ClaimState) -> ClaimState:
        start = time.time()
        coverage = state["coverage_findings"]

        prompt = self._build_prompt(state)
        response = groq_json_completion(self.client, self.model, prompt,
                                        temperature=0.1, max_tokens=2500)

        result = safe_parse_message(response)

        try:
            decision_status = DecisionStatus(result.get("decision", "NEEDS_REVIEW"))
        except ValueError:
            decision_status = DecisionStatus.NEEDS_REVIEW
        try:
            confidence = min(1.0, max(0.0, float(result.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5

        key_findings = []
        for f in result.get("key_findings", []):
            if not isinstance(f, dict) or "finding" not in f:
                continue
            citations = self._resolve_citations(f.get("citations", []), state)
            key_findings.append(KeyFinding(
                finding=f["finding"],
                supported=f.get("supported", True),
                citations=citations
            ))

        # A decision must never be emitted as an uncited conclusion. Some
        # deterministic providers intentionally return only a decision label;
        # promote the already structured specialist findings into concise,
        # traceable decision findings in that case.
        if not key_findings:
            for finding in coverage.findings:
                if not finding.citations:
                    continue
                key_findings.append(KeyFinding(
                    finding=finding.finding,
                    supported=finding.supported,
                    citations=finding.citations,
                ))

        applicable_limits = []
        for lim in result.get("applicable_limits", []):
            if not isinstance(lim, dict):
                continue
            chunk_id = lim.get("chunk_id", "")
            citation = self._resolve_citations([chunk_id], state)
            applicable_limits.append(ApplicableLimit(
                limit_type=lim.get("limit_type", ""),
                limit_amount_inr=lim.get("limit_amount_inr", 0),
                applied_amount_inr=lim.get("applied_amount_inr", 0),
                policy_section=lim.get("policy_section", ""),
                citation=citation[0] if citation else Citation(
                    claim=lim.get("limit_type", ""), source="policy.pdf",
                    page=0, section="", chunk_id=""
                )
            ))

        state["decision_draft"] = DecisionDraft(
            decision=decision_status,
            confidence=confidence,
            key_findings=key_findings,
            applicable_limits=applicable_limits,
            missing_evidence=result.get("missing_evidence", []),
            reasoning=result.get("reasoning", "")
        )

        state["trace"].append({
            "agent": "Decision",
            "action": f"Decision: {decision_status.value}, confidence {confidence:.2f}",
            "duration_ms": int((time.time() - start) * 1000),
            "evidence_count": len(key_findings),
            "metadata": {}
        })

        return state

    def _resolve_citations(self, chunk_ids: List[str], state: ClaimState) -> List[Citation]:
        if not chunk_ids:
            return []

        chunk_map = {}
        for ev_list in state.get("retrieved_evidence", {}).values():
            for ev in ev_list:
                chunk_map[ev.chunk_id] = ev

        citations = []
        for cid in chunk_ids:
            ev = chunk_map.get(cid)
            if ev:
                citations.append(Citation(
                    claim="", source="policy.pdf",
                    page=ev.page, section=ev.section, chunk_id=ev.chunk_id
                ))
        return citations


def decision_node(state: ClaimState, groq_client: Groq) -> ClaimState:
    agent = DecisionAgent(groq_client)
    return agent.decide(state)
