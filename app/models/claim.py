from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import date
from enum import Enum


class TreatmentType(str, Enum):
    INPATIENT = "inpatient"
    DAY_CARE = "day_care"
    DOMICILIARY = "domiciliary"


class DecisionStatus(str, Enum):
    ADMISSIBLE = "ADMISSIBLE"
    ADMISSIBLE_WITH_LIMITS = "ADMISSIBLE_WITH_LIMITS"
    PARTIALLY_ADMISSIBLE = "PARTIALLY_ADMISSIBLE"
    NOT_ADMISSIBLE = "NOT_ADMISSIBLE"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class ValidationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


class Hospital(BaseModel):
    name: str
    network_provider: bool


class Patient(BaseModel):
    age: int


class Treatment(BaseModel):
    type: TreatmentType
    admission_hours: int
    diagnosis: str
    procedure: str
    pre_existing: bool
    experimental: bool
    hospital_room_unavailable: Optional[bool] = None
    patient_cannot_be_moved: Optional[bool] = None


class ExpensesINR(BaseModel):
    room: int = 0
    doctor_fees: int = 0
    medicines_diagnostics: int = 0
    pre_hospitalization: int = 0
    post_hospitalization: int = 0
    ambulance: int = 0


class EvidenceContext(BaseModel):
    hospital_registered: Optional[bool] = None
    medical_necessity_confirmed: Optional[bool] = None
    hospital_minimum_criteria_documented: Optional[bool] = None


class ExpenseTiming(BaseModel):
    pre_hospitalization_days_before_admission: Optional[int] = None
    post_hospitalization_days_after_discharge: Optional[int] = None
    same_condition_confirmed: Optional[bool] = None


class PriorPolicy(BaseModel):
    insurer_type: str
    continuous_years: int
    database_and_claim_history_received: bool
    previous_sum_insured_inr: int


class ClaimCase(BaseModel):
    case_id: str
    policy_id: str
    policy_start_date: date
    claim_date: date
    sum_insured_inr: int
    continuous_coverage_months: int
    prior_insurer_continuous_years: int = 0
    patient: Patient
    hospital: Hospital
    treatment: Treatment
    expenses_inr: ExpensesINR
    documents: List[str]
    task: str
    evidence_context: Optional[EvidenceContext] = None
    expense_timing: Optional[ExpenseTiming] = None
    prior_policy: Optional[PriorPolicy] = None


class Citation(BaseModel):
    claim: str
    source: str
    page: int
    section: str
    chunk_id: str


class ApplicableLimit(BaseModel):
    limit_type: str
    limit_amount_inr: int
    applied_amount_inr: int
    policy_section: str
    citation: Citation


class KeyFinding(BaseModel):
    finding: str
    supported: bool
    citations: List[Citation]


class ValidationResult(BaseModel):
    status: ValidationStatus
    unsupported_claims: List[str] = []


class AgentTrace(BaseModel):
    agent: str
    action: str
    duration_ms: int
    evidence_count: int = 0
    metadata: Dict[str, Any] = {}


class DecisionResponse(BaseModel):
    case_id: str
    decision: DecisionStatus
    confidence: float = Field(ge=0.0, le=1.0)
    key_findings: List[KeyFinding] = []
    applicable_limits: List[ApplicableLimit] = []
    missing_evidence: List[str] = []
    citations: List[Citation] = []
    validation: ValidationResult
    trace: List[AgentTrace] = []


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    ready: bool = False
    detail: str = ""
