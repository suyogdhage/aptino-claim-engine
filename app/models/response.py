from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from app.models.claim import (
    ClaimCase, DecisionResponse, HealthResponse, DecisionStatus,
    ValidationStatus, Citation, KeyFinding, ApplicableLimit, AgentTrace
)
from app.models.state import InvestigationPlan, CoverageFindings, DecisionDraft, Evidence


class AnalyzeRequest(BaseModel):
    case: ClaimCase


class AnalyzeResponse(BaseModel):
    result: DecisionResponse


class IngestRequest(BaseModel):
    force_reindex: bool = False


class IngestResponse(BaseModel):
    success: bool
    chunks_indexed: int
    message: str


class EvaluationCase(BaseModel):
    case_id: str
    case: ClaimCase
    expected_decision: DecisionStatus
    expected_confidence_min: float = 0.5
    expected_key_findings: List[str] = []
    should_abstain: bool = False


class EvaluationResult(BaseModel):
    case_id: str
    predicted: DecisionResponse
    expected_decision: DecisionStatus
    correct: bool
    confidence_calibrated: bool
    retrieval_recall_at_k: Dict[int, float]
    citation_correctness: float
    notes: str


class EvaluationSummary(BaseModel):
    total_cases: int
    correct_decisions: int
    accuracy: float
    avg_confidence: float
    abstention_rate: float
    retrieval_metrics: Dict[str, float]
    citation_correctness_avg: float
    failure_cases: List[Dict[str, Any]]