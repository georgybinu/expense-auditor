def evaluate_expense(amount: float) -> str:
    if amount < 500:
        return "Approved"
    return "Rejected"
