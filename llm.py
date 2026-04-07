import json
import re

import ollama


def query_llama3(prompt: str) -> str:
    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return response["message"]["content"]


def extract_llm_json(response_text: str) -> dict:
    default_result = {
        "status": None,
        "reason": None,
        "confidence": None,
    }

    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not match:
            return default_result

        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return default_result

    return {
        "status": parsed.get("status"),
        "reason": parsed.get("reason"),
        "confidence": parsed.get("confidence"),
    }
