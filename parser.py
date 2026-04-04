import re
from typing import Dict, Optional


def extract_receipt_details(ocr_text: str) -> Dict[str, Optional[str]]:
    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
    amount_pattern = r"(?:(?:[$€£])|(?:Rs\.?))?\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})"
    merchant_ignore_words = (
        "receipt",
        "invoice",
        "tax",
        "cashier",
        "server",
        "table",
        "order",
        "transaction",
        "approval",
        "auth",
        "card",
        "visa",
        "mastercard",
        "subtotal",
        "total",
        "amount",
        "change",
        "balance",
        "date",
        "time",
        "www",
        "http",
    )

    merchant_name = None
    for line in lines[:8]:
        normalized_line = line.lower()
        if re.search(amount_pattern, line):
            continue
        if any(word in normalized_line for word in merchant_ignore_words):
            continue
        if re.search(r"\d{3,}", line):
            continue
        if len(line) < 3:
            continue
        merchant_name = line
        break

    if merchant_name is None and lines:
        merchant_name = lines[0]

    date_patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
        r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}\b",
        r"\b[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}\b",
    ]
    date_match = None
    for pattern in date_patterns:
        date_match = re.search(pattern, ocr_text, re.IGNORECASE)
        if date_match:
            break

    total_amount = None
    total_keywords = [
        "grand total",
        "amount due",
        "net total",
        "total due",
        "total",
    ]
    excluded_total_words = ("subtotal", "tax", "discount", "change", "balance", "tip")

    for line in lines:
        normalized_line = line.lower()
        if any(word in normalized_line for word in excluded_total_words):
            continue
        if any(keyword in normalized_line for keyword in total_keywords):
            amounts = re.findall(amount_pattern, line)
            if amounts:
                total_amount = amounts[-1].replace(" ", "")
                break

    if total_amount is None:
        amount_matches = []
        for line in lines:
            normalized_line = line.lower()
            if any(word in normalized_line for word in ("subtotal", "tax", "discount", "change")):
                continue
            amount_matches.extend(re.findall(amount_pattern, line))
        if amount_matches:
            total_amount = amount_matches[-1].replace(" ", "")

    return {
        "merchant_name": merchant_name,
        "date": date_match.group(0) if date_match else None,
        "total_amount": total_amount,
    }
