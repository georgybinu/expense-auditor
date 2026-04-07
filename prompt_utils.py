from typing import Dict, Optional


def build_auditor_prompt(
    receipt_data: Dict[str, Optional[str]],
    justification: str,
    policy_text: str,
    city: str,
    role: str,
) -> str:
    merchant = receipt_data.get("merchant_name") or "Unknown"
    amount = receipt_data.get("total_amount") or "Unknown"
    date = receipt_data.get("date") or "Unknown"

    return f"""You are an expense auditor.

Use only the retrieved policy chunks provided below to make your decision.
Do not rely on outside assumptions, general knowledge, or unstated company rules.
If the retrieved policy chunks do not contain enough information, return "Flagged" and explain that the policy evidence is insufficient.

Receipt:
Merchant: {merchant}
Amount: {amount}
Date: {date}

Employee:
City: {city}
Role: {role}

Justification:
{justification}

Retrieved Policy Chunks:
{policy_text}

Tasks:
1. Check whether the receipt and justification violate the retrieved policy chunks
2. Check whether the justification is supported by the retrieved policy chunks
3. Return a decision based strictly on the retrieved policy chunks

Output JSON:
{{
  "status": "...",
  "reason": "...",
  "confidence": "..."
}}"""
