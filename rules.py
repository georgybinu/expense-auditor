from typing import Dict, Optional


def evaluate_expense(amount: float) -> str:
    if amount < 500:
        return "Approved"
    return "Rejected"


def evaluate_expense_rule(amount: Optional[float]) -> Dict[str, str]:
    if amount is None:
        return {
            "status": "Rejected",
            "reason": "Amount could not be determined from the document",
        }

    if amount < 500:
        return {
            "status": "Approved",
            "reason": "Amount is below the approval threshold of 500",
        }

    return {
        "status": "Rejected",
        "reason": "Amount is 500 or more",
    }
