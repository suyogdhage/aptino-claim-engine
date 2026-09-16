from typing import List, Dict, Any, Optional, TypedDict
from enum import Enum
from pydantic import BaseModel
from app.models.claim import (
    ClaimCase, DecisionStatus, ValidationStatus, Citation,
    KeyFinding, ApplicableLimit, AgentTrace, EvidenceContext, ExpenseTiming, PriorPolicy
)


class InvestigationDimension(str, Enum):
    WAITING_PERIOD = "waiting_period"
    PRE_EXISTING = "pre_existing"
    COVERAGE_SCOPE = "coverage_scope"
    EXCLUSIONS = "exclusions"
    HOSPITAL_DEFINITION = "hospital_definition"
    DOMICILIARY_CONDITIONS = "domiciliary_conditions"
    DAY_CARE_QUALIFICATION = "day_care_qualification"
    CATEGORY_LIMITS = "category_limits"
    PRE_POST_HOSPITALIZATION = "pre_post_hospitalization"
    PORTABILITY = "portability"
    EXPERIMENTAL_TREATMENT = "experimental_treatment"
    EVIDENCE_SUFFICIENCY = "evidence_sufficiency"


class InvestigationPlan(BaseModel):
    dimensions: List[InvestigationDimension]
    missing_fields: List[str]
    checklist: List[str]
    priority_order: List[InvestigationDimension]


class Evidence(BaseModel):
    chunk_id: str
    text: str
    page: int
    section: str
    subsection: Optional[str] = None
    clause_id: Optional[str] = None
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rerank_score: float = 0.0
    relevance: float = 0.0
    dimension: Optional[InvestigationDimension] = None


class CoverageFinding(BaseModel):
    dimension: InvestigationDimension
    finding: str
    supported: bool
    citations: List[Citation]
    applicable_limits: List[ApplicableLimit] = []
    waiting_period_status: Optional[str] = None
    exclusion_applies: bool = False


class CoverageFindings(BaseModel):
    findings: List[CoverageFinding]
    overall_admissible: bool
    blocking_issues: List[str] = []
    applicable_limits: List[ApplicableLimit] = []


class DecisionDraft(BaseModel):
    decision: DecisionStatus
    confidence: float
    key_findings: List[KeyFinding]
    applicable_limits: List[ApplicableLimit]
    missing_evidence: List[str]
    reasoning: str


from app.models.claim import DecisionResponse, ValidationResult


class ClaimState(TypedDict):
    case: ClaimCase
    investigation_plan: Optional[InvestigationPlan]
    retrieved_evidence: Dict[InvestigationDimension, List[Evidence]]
    coverage_findings: Optional[CoverageFindings]
    decision_draft: Optional[DecisionDraft]
    final_decision: Optional[DecisionResponse]
    validation_result: Optional[ValidationResult]
    trace: List[AgentTrace]
    retry_count: int
    error: Optional[str]