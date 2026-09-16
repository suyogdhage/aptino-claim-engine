import json
import os
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class ExpectedOutcome(BaseModel):
    decision: str
    confidence_min: float = 0.0
    note: str = ""


class GoldEvidence(BaseModel):
    gold_pages: List[int] = []


EXPECTED_OUTCOMES_FILE = os.path.join(os.path.dirname(__file__), "expected_outcomes.json")
GOLD_EVIDENCE_FILE = os.path.join(os.path.dirname(__file__), "gold_evidence.json")


def load_expected_outcomes() -> Dict[str, ExpectedOutcome]:
    with open(EXPECTED_OUTCOMES_FILE, "r") as f:
        raw = json.load(f)
    return {k: ExpectedOutcome(**v) for k, v in raw.items()}


def load_gold_evidence() -> Dict[str, GoldEvidence]:
    if not os.path.exists(GOLD_EVIDENCE_FILE):
        return {}
    with open(GOLD_EVIDENCE_FILE, "r") as f:
        raw = json.load(f)
    return {k: GoldEvidence(**v) for k, v in raw.items()}