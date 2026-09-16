import json
import os
import sys
from typing import List, Optional, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import httpx
from app.models.claim import ClaimCase, DecisionResponse


API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
DATA_DIR = "./data"


def load_public_cases():
    path = os.path.join(DATA_DIR, "public_test_cases.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def format_inr(value: int) -> str:
    return f"₹{value:,}"


def main():
    st.set_page_config(page_title="Aptino Claim Decision Engine", layout="wide")
    st.title("Aptino Policy-Aware Claim Decision Engine")
    st.caption("Multi-Agent RAG system analyzing health insurance claims against supplied policy.")

    public_cases = load_public_cases()

    source = st.radio("Case source:", ["Public Test Case", "Upload JSON", "Paste JSON"], horizontal=True)

    case_data = None
    if source == "Public Test Case":
        if public_cases:
            labels = {c["case_id"]: c for c in public_cases}
            selected = st.selectbox("Select a case:", list(labels.keys()))
            case_data = labels[selected]
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
            with st.spinner("Running multi-agent workflow..."):
                try:
                    resp = httpx.post(
                        f"{API_BASE_URL}/analyze",
                        json=case_data,
                        timeout=300
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    display_result(result)
                except httpx.HTTPStatusError as e:
                    detail = "The API could not complete this request."
                    try:
                        detail = e.response.json().get("detail", detail)
                    except ValueError:
                        pass
                    st.error(detail)
                    st.info("Check the API readiness endpoint, then retry the analysis.")
                except httpx.RequestError:
                    st.error("Cannot reach the API. Confirm API_BASE_URL and that the backend is running.")
                    st.info("Once the backend is ready, click Analyze Claim again.")
    else:
        st.info("Select or provide a claim case to analyze.")


def display_result(result: Dict[str, Any]):
    decision = result.get("decision", "NEEDS_REVIEW")
    confidence = result.get("confidence", 0.0)
    validation = result.get("validation", {})

    color_map = {
        "ADMISSIBLE": "normal",
        "ADMISSIBLE_WITH_LIMITS": "normal",
        "PARTIALLY_ADMISSIBLE": "normal",
        "NOT_ADMISSIBLE": "exception",
        "NEEDS_REVIEW": "warning",
    }

    st.subheader("Decision")

    badge_colors = {
        "ADMISSIBLE": "#14532d",
        "ADMISSIBLE_WITH_LIMITS": "#14532d",
        "PARTIALLY_ADMISSIBLE": "#7c2d12",
        "NOT_ADMISSIBLE": "#7f1d1d",
        "NEEDS_REVIEW": "#7c2d12",
    }
    bg = badge_colors.get(decision, "#1e293b")
    st.markdown(
        f"<div style='background:{bg};color:white;padding:8px 16px;border-radius:8px;"
        f"font-size:20px;font-weight:bold;display:inline-block'>{decision}</div>",
        unsafe_allow_html=True,
    )
    st.write(f"**Confidence:** {confidence:.0%}")

    if decision == "NEEDS_REVIEW":
        st.warning("Insufficient evidence — the system abstains from a final decision.")
    elif validation.get("status") == "FAIL":
        st.error("Validation FAILED — some claims are not supported by cited evidence.")
        for unsupported in validation.get("unsupported_claims", []):
            st.caption(f"Validation detail: {unsupported}")

    st.subheader("Key Findings")
    for f in result.get("key_findings", []):
        with st.expander(f["finding"]):
            st.write(f"Supported: {f.get('supported', True)}")
            for cit in f.get("citations", []):
                st.caption(f"Page {cit.get('page')} · Section {cit.get('section')} · {cit.get('chunk_id')}")

    st.subheader("Applicable Limits / Deductions")
    limits = result.get("applicable_limits", [])
    if limits:
        for lim in limits:
            st.write(
                f"**{lim['limit_type']}**: cap {format_inr(lim.get('limit_amount_inr', 0))}, "
                f"applied {format_inr(lim.get('applied_amount_inr', 0))} "
                f"({lim.get('policy_section', '')})"
            )
    else:
        st.caption("No limits applied.")

    st.subheader("Missing Evidence")
    missing = result.get("missing_evidence", [])
    if missing:
        for m in missing:
            st.warning(m)
    else:
        st.caption("None.")

    st.subheader("Citations")
    for cit in result.get("citations", []):
        st.caption(
            f"[{cit.get('chunk_id', '')}] Page {cit.get('page')} · {cit.get('section', '')}"
        )

    st.subheader("Execution Trace")
    for t in result.get("trace", []):
        st.caption(
            f"**{t.get('agent')}**: {t.get('action')} — "
            f"{t.get('duration_ms', 0)}ms, {t.get('evidence_count', 0)} evidence"
        )

    if st.download_button("Download Result JSON", json.dumps(result, indent=2),
                          file_name=f"{result.get('case_id', 'result')}.json"):
        pass


if __name__ == "__main__":
    main()
