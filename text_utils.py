from typing import List, Optional


def chunk_text(text: str, chunk_size: int = 300) -> List[str]:
    words = text.split()
    chunks = []

    for index in range(0, len(words), chunk_size):
        chunk = words[index:index + chunk_size]
        chunks.append(" ".join(chunk))

    return chunks


def retrieve_relevant_chunks(
    chunks: List[str],
    expense_category: Optional[str] = None,
    city: Optional[str] = None,
    employee_role: Optional[str] = None,
    max_results: int = 3,
) -> List[str]:
    keywords = [
        value.strip().lower()
        for value in (expense_category, city, employee_role)
        if value and value.strip()
    ]

    if not keywords:
        return chunks[:max_results]

    scored_chunks = []
    for chunk in chunks:
        lowered_chunk = chunk.lower()
        score = sum(lowered_chunk.count(keyword) for keyword in keywords)
        if score > 0:
            scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored_chunks[:max_results]]
