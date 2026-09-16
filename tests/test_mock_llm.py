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


def coverage_prompt(post_hosp: int = 0, *, same_condition: object = None) -> str:
    lines = [
        "You are a Coverage & Exclusion Assessment Agent",
        "CLAIM INFORMATION:",
        "Case ID: TEST",
        "Policy Start: 2025-01-01, Claim Date: 2026-03-01",
        "Sum Insured: ₹500,000",
        "Continuous Coverage: 14 months",
        "Prior Insurer Years: 0",
        "Patient Age: 30",
        "Hospital: City Hospital (Network: True)",
        "Treatment: inpatient, 24 hours",
        "Diagnosis: Acute appendicitis",
        "Admission hours: 24",
        "Pre-existing: False",
        "Experimental: False",
        "Expenses: "
        f"Room ₹0, Doctor ₹0, Medicines ₹0, Pre-hosp ₹0, Post-hosp ₹{post_hosp}, Ambulance ₹0",
        "Documents: claim_form",
        "Task: test",
    ]
    if same_condition is not None:
        lines.append(f"Pre/Post same condition: {same_condition}")
    lines.append("POLICY EVIDENCE:")
    return "\n".join(lines)


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

    def test_coverage_flags_out_of_condition_post_hospitalization_expense(self):
        result = json.loads(self.chat._coverage_exclusion(
            coverage_prompt(post_hosp=18000, same_condition="false")))
        self.assertTrue(any("not incurred for the same condition" in b
                            for b in result["blocking_issues"]))
        self.assertTrue(result["overall_admissible"])

    def test_coverage_no_flag_when_same_condition_confirmed(self):
        result = json.loads(self.chat._coverage_exclusion(
            coverage_prompt(post_hosp=18000, same_condition="true")))
        self.assertFalse(any("not incurred for the same condition" in b
                             for b in result["blocking_issues"]))

    def test_partially_admissible_emitted_for_unrelated_post_hospitalization_expense(self):
        result = json.loads(self.chat._decision(
            decision_prompt(
                "Room ₹0, Doctor ₹0, Medicines ₹0, Pre-hosp ₹0, Post-hosp ₹18,000, Ambulance ₹800"
            ).replace(
                "BLOCKING ISSUES:\nNone",
                "BLOCKING ISSUES:\n- Pre/post-hospitalization expenses not payable: "
                "not incurred for the same condition as the admission",
            )
        ))
        self.assertEqual(result["decision"], "PARTIALLY_ADMISSIBLE")


if __name__ == "__main__":
    unittest.main()
