import json
import unittest

from app.llm import MockChat


def decision_prompt(expenses: str, *, months: int = 14, pre_existing: bool = False) -> str:
    return f"""You are a Decision Agent
CLAIM INFORMATION:
Case ID: TEST
Policy Start: 2025-01-01, Claim Date: 2026-03-01
Sum Insured: ₹500,000
Continuous Coverage: {months} months
Prior Insurer Years: 0
Hospital: Test Hospital (Network: True)
Treatment: inpatient, 24 hours
Diagnosis: Acute appendicitis
Admission hours: 24
Expenses: {expenses}
Pre-existing: {pre_existing}
Experimental: False

BLOCKING ISSUES:
None

APPLICABLE LIMITS:
None
"""


class MockDecisionTests(unittest.TestCase):
    def setUp(self):
        self.chat = MockChat()

    def test_doctor_fee_cap_uses_request_schema_field(self):
        result = json.loads(self.chat._decision(decision_prompt(
            "Room ₹0, Doctor ₹130,000, Medicines ₹0, Pre-hosp ₹0, Post-hosp ₹0, Ambulance ₹0"
        )))
        self.assertEqual(result["decision"], "ADMISSIBLE_WITH_LIMITS")

    def test_medicine_cap_uses_request_schema_field(self):
        result = json.loads(self.chat._decision(decision_prompt(
            "Room ₹0, Doctor ₹0, Medicines ₹210,000, Pre-hosp ₹0, Post-hosp ₹0, Ambulance ₹0"
        )))
        self.assertEqual(result["decision"], "ADMISSIBLE_WITH_LIMITS")

    def test_pre_existing_condition_before_48_months_is_rejected(self):
        result = json.loads(self.chat._decision(decision_prompt(
            "Room ₹0, Doctor ₹0, Medicines ₹0, Pre-hosp ₹0, Post-hosp ₹0, Ambulance ₹0",
            months=47,
            pre_existing=True,
        )))
        self.assertEqual(result["decision"], "NOT_ADMISSIBLE")


if __name__ == "__main__":
    unittest.main()
