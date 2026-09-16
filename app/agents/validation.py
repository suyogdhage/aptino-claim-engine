import time
import json
from typing import Dict, List
from groq import Groq
from app.models.state import (
    ClaimState, Evidence, DecisionDraft, InvestigationDimension
)
from app.models.claim import ValidationResult, ValidationStatus
from app.config import settings
from app.llm import groq_json_completion, safe_parse_message


class ValidationAgent:
    def __init__(self, groq_client: Groq, model: str = None):
        self.client = groq_client
        self.model = model or settings.GROQ_MODEL

    def _evidence_text_map(self, state: ClaimState) -> Dict[str, Evidence]:
        chunk_map = {}
        for ev_list in state.get("retrieved_evidence", {}).values():
            for ev in ev_list:
                chunk_map[ev.chunk_id] = ev
        return chunk_map

    def _build_prompt(self, state: ClaimState) -> str:
        draft = state["decision_draft"]
        chunk_map = self._evidence_text_map(state)

        # Collect all citations used in the decision
        cited_chunks = set()
        findings_lines = []
        for f in draft.key_findings:
            cids = [c.chunk_id for c in f.citations]
            cited_chunks.update(cids)
            findings_lines.append(f"[{', '.join(cids)}] {f.finding}")

        limits_lines = []
        for l in draft.applicable_limits:
            cids = [l.citation.chunk_id]
            cited_chunks.update(cids)
            limits_lines.append(f"[{', '.join(cids)}] {l.limit_type}: cap ₹{l.limit_amount_inr:,} applied ₹{l.applied_amount_inr:,}")

        evidence_texts = []
        for cid in sorted(cited_chunks):
            ev = chunk_map.get(cid)
            if ev:
                evidence_texts.append(f"[Chunk {cid}] Page {ev.page}, Section {ev.section}\n{ev.text[:800]}")
            else:
                evidence_texts.append(f"[Chunk {cid}] ** NOT RETRIEVED — citation has no supporting evidence **")

        return f"""You are a Validation Agent. Verify that each material decision statement is actually supported by the cited policy evidence.

FINAL DECISION: {draft.decision.value}
CONFIDENCE: {draft.confidence}

KEY FINDINGS:
{chr(10).join(findings_lines)}

APPLICABLE LIMITS:
{chr(10).join(limits_lines) if limits_lines else "None"}

RETRIEVED EVIDENCE:
{chr(10).join(evidence_texts)}

For each key finding and limit, determine whether the cited evidence genuinely supports the claim. Check for:
1. Citation exists and matches content
2. Evidence actually supports the statement (no overreach)
3. No hallucinated policy content

Return ONLY valid JSON:
{{
  "status": "PASS" or "FAIL",
  "unsupported_claims": [
    {{
      "claim": "the unsupported statement",
      "chunk_ids": ["cited chunk"],
      "reason": "why unsupported"
    }}
  ]
}}"""

    def validate(self, state: ClaimState) -> ClaimState:
        start = time.time()
        draft = state["decision_draft"]

        prompt = self._build_prompt(state)
        response = groq_json_completion(self.client, self.model, prompt,
                                        temperature=0.0, max_tokens=1500)

        result = safe_parse_message(response)

        unsupported = []
        for finding in draft.key_findings:
            if not finding.citations:
                unsupported.append(f"{finding.finding} (chunks: ) — material finding has no policy citation")
        for u in result.get("unsupported_claims", []):
            if not isinstance(u, dict):
                continue
            unsupported.append(f"{u.get('claim', '')} (chunks: {', '.join(u.get('chunk_ids', []))}) — {u.get('reason', '')}")

        try:
            status = ValidationStatus(result.get("status", "FAIL"))
        except ValueError:
            status = ValidationStatus.FAIL

        validation = ValidationResult(
            status=status,
            unsupported_claims=unsupported
        )

        state["validation_result"] = validation
        state["trace"].append({
            "agent": "Validation",
            "action": f"Validation: {validation.status.value}, {len(unsupported)} unsupported claims",
            "duration_ms": int((time.time() - start) * 1000),
            "evidence_count": len(unsupported),
            "metadata": {"unsupported_claims": unsupported}
        })

        return state


def validation_node(state: ClaimState, groq_client: Groq) -> ClaimState:
    agent = ValidationAgent(groq_client)
    return agent.validate(state)
