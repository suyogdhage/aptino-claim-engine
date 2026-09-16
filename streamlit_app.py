import json
import sys
import os
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.models.claim import ClaimCase, DecisionResponse


def format_inr(value: int) -> str:
    return f"\u20b9{value:,}"


@st.cache_resource
def load_engine():
    from app.config import settings
    from app.retrieval.hybrid_retriever import build_indices, index_status

    ready, _ = index_status(settings.POLICY_PDF_PATH, settings.CHROMA_PERSIST_DIR)
    if not ready:
        build_indices(settings.POLICY_PDF_PATH, settings.CHROMA_PERSIST_DIR)

    from app.graph.workflow import ClaimEngine
    return ClaimEngine()


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def main():
    st.set_page_config(page_title="Aptino Claim Decision Engine", layout="wide")
    st.title("Aptino Policy-Aware Claim Decision Engine")
    st.caption("Multi-Agent RAG system analyzing health insurance claims against supplied policy. Runs entirely in-app\u2014no separate backend needed.")

    public_cases = []
    path = os.path.join(DATA_DIR, "public_test_cases.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            public_cases = json.load(f)

    source = st.radio("Case source:", ["Public Test Case", "Upload JSON", "Paste JSON"], horizontal=True)

    case_data = None
    if source == "Public Test Case":
        if public_cases:
            selected = st.selectbox("Select a case:", [c["case_id"] for c in public_cases])
            case_data = next((c for c in public_cases if c["case_id"] == selected), None)
        else:
            st.warning("No public test cases found.")
    elif source == "Upload JSON":
        uploaded = st.file_uploader("Upload claim case JSON", type=["json"])
        if uploaded:
            case_data = json.loads(uploaded.read())
    else:
        pasted = st.text_area("Paste claim case JSON", height=300)
        if pasted:
            try:
                case_data = json.loads(pasted)
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

    if case_data:
        try:
            case = ClaimCase(**case_data)
        except Exception as e:
            st.error(f"Validation error: {e}")
            return

        st.subheader(f"Case: {case.case_id}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Policy Start", str(case.policy_start_date))
        col2.metric("Claim Date", str(case.claim_date))
        col3.metric("Sum Insured", format_inr(case.sum_insured_inr))
        col4.metric("Coverage (months)", case.continuous_coverage_months)

        with st.expander("Full Case Details"):
            st.json(case_data)

        if st.button("Analyze Claim", type="primary"):
            with st.spinner("Running multi-agent workflow... this takes ~60-120s"):
                try:
                    engine = load_engine()
                    result = engine.analyze(case)
                    st.session_state["result"] = result
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    st.stop()

    if "result" in st.session_state:
        result = st.session_state["result"]
        display_result(result)


def display_result(result):
    decision = result.decision.value
    confidence = result.confidence
    validation = {"status": result.validation.status.value, "unsupported_claims": result.validation.unsupported_claims}

    color_map = {
        "ADMISSIBLE": "#14532d",
        "ADMISSIBLE_WITH_LIMITS": "#14532d",
        "PARTIALLY_ADMISSIBLE": "#7c2d12",
        "NOT_ADMISSIBLE": "#7f1d1d",
        "NEEDS_REVIEW": "#7c2d12",
    }
    bg = color_map.get(decision, "#1e293b")

    st.subheader("Decision")
    st.markdown(
        f"<div style='background:{bg};color:white;padding:8px 16px;border-radius:8px;"
        f"font-size:20px;font-weight:bold;display:inline-block'>{decision}</div>",
        unsafe_allow_html=True,
    )
    st.write(f"**Confidence:** {confidence:.0%}")

    if decision == "NEEDS_REVIEW":
        st.warning("Insufficient evidence \u2014 the system abstains from a final decision.")
    elif validation.get("status") == "FAIL":
        st.error("Validation FAILED \u2014 some claims are not supported by cited evidence.")
        for unsupported in validation.get("unsupported_claims", []):
            st.caption(f"Validation detail: {unsupported}")

    st.subheader("Key Findings")
    for f in result.key_findings:
        with st.expander(f.finding):
            st.write(f"Supported: {f.supported}")
            for cit in f.citations:
                st.caption(f"Page {cit.page} \u00b7 Section {cit.section} \u00b7 {cit.chunk_id}")

    st.subheader("Applicable Limits / Deductions")
    if result.applicable_limits:
        for lim in result.applicable_limits:
            st.write(
                f"**{lim.limit_type}**: cap {format_inr(lim.limit_amount_inr)}, "
                f"applied {format_inr(lim.applied_amount_inr)} "
                f"({lim.policy_section})"
            )
    else:
        st.caption("No limits applied.")

    st.subheader("Missing Evidence")
    if result.missing_evidence:
        for m in result.missing_evidence:
            st.warning(m)
    else:
        st.caption("None.")

    st.subheader("Citations")
    for cit in result.citations:
        st.caption(f"[{cit.chunk_id}] Page {cit.page} \u00b7 {cit.section}")

    st.subheader("Execution Trace")
    for t in result.trace:
        st.caption(f"**{t.agent}**: {t.action} \u2014 {t.duration_ms}ms, {t.evidence_count} evidence")

    st.download_button(
        "Download Result JSON",
        json.dumps({
            "case_id": result.case_id,
            "decision": result.decision.value,
            "confidence": result.confidence,
            "key_findings": [{"finding": f.finding, "supported": f.supported,
                              "citations": [{"chunk_id": c.chunk_id, "page": c.page, "section": c.section} for c in f.citations]}
                             for f in result.key_findings],
            "missing_evidence": result.missing_evidence,
        }, indent=2),
        file_name=f"{result.case_id}_result.json",
    )


if __name__ == "__main__":
    main()
