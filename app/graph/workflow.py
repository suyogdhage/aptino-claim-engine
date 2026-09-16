import json
from typing import Optional, List, Dict, Any
from groq import Groq
from app.models.state import ClaimState, InvestigationDimension, CoverageFindings, CoverageFinding, KeyFinding, ApplicableLimit, Citation, DecisionDraft, Evidence
from app.models.claim import DecisionStatus, ValidationStatus, ValidationResult, DecisionResponse, ClaimCase
from app.retrieval.hybrid_retriever import HybridRetriever

from langgraph.graph import StateGraph, END

from app.agents.case_analysis import case_analysis_node
from app.agents.policy_evidence import policy_evidence_node
from app.agents.coverage_exclusion import coverage_exclusion_node
from app.agents.decision import decision_node
from app.agents.validation import validation_node
from app.config import settings


def create_initial_state(case: ClaimCase) -> ClaimState:
    return ClaimState(
        case=case,
        investigation_plan=None,
        retrieved_evidence={},
        coverage_findings=None,
        decision_draft=None,
        final_decision=None,
        validation_result=None,
        trace=[],
        retry_count=0,
        error=None
    )


def build_workflow(groq_client: Groq, retriever: HybridRetriever):
    graph = StateGraph(ClaimState)

    def case_analysis_step(state: ClaimState) -> ClaimState:
        return case_analysis_node(state, groq_client)

    def policy_evidence_step(state: ClaimState) -> ClaimState:
        return policy_evidence_node(state, retriever)

    def coverage_exclusion_step(state: ClaimState) -> ClaimState:
        return coverage_exclusion_node(state, groq_client)

    def decision_step(state: ClaimState) -> ClaimState:
        retry = state.get("retry_count", 0)
        state = decision_node(state, groq_client)
        state["retry_count"] = retry + 1
        return state

    def validation_step(state: ClaimState) -> ClaimState:
        return validation_node(state, groq_client)

    def finalize_step(state: ClaimState) -> ClaimState:
        draft = state["decision_draft"]
        validation = state["validation_result"]

        # A decision that remains unsupported after the bounded retry loop must
        # not be presented as a final adjudication.
        if validation.status != ValidationStatus.PASS:
            draft.decision = DecisionStatus.NEEDS_REVIEW
            draft.confidence = min(draft.confidence, 0.5)
            if "Decision validation failed; manual review required." not in draft.missing_evidence:
                draft.missing_evidence.append("Decision validation failed; manual review required.")

        citations = []
        for f in draft.key_findings:
            for c in f.citations:
                c.claim = f.finding
                citations.append(c)
        for lim in draft.applicable_limits:
            citations.append(lim.citation)

        state["final_decision"] = DecisionResponse(
            case_id=state["case"].case_id,
            decision=draft.decision,
            confidence=draft.confidence,
            key_findings=draft.key_findings,
            applicable_limits=draft.applicable_limits,
            missing_evidence=draft.missing_evidence,
            citations=citations,
            validation=validation,
            trace=state.get("trace", [])
        )
        return state

    def validation_router(state: ClaimState) -> str:
        validation = state["validation_result"]
        retry = state["retry_count"]
        if validation.status == ValidationStatus.PASS or retry >= 2:
            return "finalize"
        return "decision"

    graph.add_node("case_analysis", case_analysis_step)
    graph.add_node("policy_evidence", policy_evidence_step)
    graph.add_node("coverage_exclusion", coverage_exclusion_step)
    graph.add_node("decision", decision_step)
    graph.add_node("validation", validation_step)
    graph.add_node("finalize", finalize_step)

    graph.set_entry_point("case_analysis")
    graph.add_edge("case_analysis", "policy_evidence")
    graph.add_edge("policy_evidence", "coverage_exclusion")
    graph.add_edge("coverage_exclusion", "decision")
    graph.add_edge("decision", "validation")
    graph.add_conditional_edges(
        "validation",
        validation_router,
        {"decision": "decision", "finalize": "finalize"}
    )
    graph.add_edge("finalize", END)

    return graph.compile()


class ClaimEngine:
    def __init__(
        self,
        groq_client=None,
        retriever: Optional[HybridRetriever] = None
    ):
        from app.llm import get_llm_client
        self.groq_client = groq_client or get_llm_client()
        self.retriever = retriever or self._build_retriever()
        self.workflow = build_workflow(self.groq_client, self.retriever)

    def _build_retriever(self) -> HybridRetriever:
        import os
        from app.retrieval.chroma_store import ChromaStore
        from app.retrieval.bm25_index import BM25Index
        from app.retrieval.hybrid_retriever import index_status

        ready, detail = index_status(settings.POLICY_PDF_PATH, settings.CHROMA_PERSIST_DIR)
        if not ready:
            raise RuntimeError(f"Retrieval index is not ready: {detail}")

        chroma = ChromaStore(persist_dir=settings.CHROMA_PERSIST_DIR)
        bm25 = BM25Index(index_path=os.path.join(settings.CHROMA_PERSIST_DIR, "bm25_index.pkl"))
        if not bm25.load():
            raise RuntimeError("BM25 index could not be loaded")

        return HybridRetriever(chroma_store=chroma, bm25_index=bm25)

    def analyze(self, case: ClaimCase) -> DecisionResponse:
        state = create_initial_state(case)
        final_state = self.workflow.invoke(state)
        return final_state["final_decision"]
