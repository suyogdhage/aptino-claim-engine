import json
import os
import time
from typing import List, Dict, Any
from pydantic import BaseModel


class ExpectedOutcome(BaseModel):
    decision: str
    confidence_min: float = 0.0
    note: str = ""


EXPECTED_OUTCOMES_FILE = os.path.join(os.path.dirname(__file__), "expected_outcomes.json")


def load_expected_outcomes() -> Dict[str, ExpectedOutcome]:
    with open(EXPECTED_OUTCOMES_FILE, "r") as f:
        raw = json.load(f)
    return {k: ExpectedOutcome(**v) for k, v in raw.items()}