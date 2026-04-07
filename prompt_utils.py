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

Receipt:
Merchant: {merchant}
Amount: {amount}
Date: {date}

Employee:
City: {city}
Role: {role}

Justification:
{justification}

Policy:
{policy_text}

Tasks:
1. Check policy violations
2. Check justification validity
3. Return decision

Output JSON:
{{
  "status": "...",
  "reason": "...",
  "confidence": "..."
}}"""
