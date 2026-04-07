from typing import List


def chunk_text(text: str, chunk_size: int = 300) -> List[str]:
    words = text.split()
    chunks = []

    for index in range(0, len(words), chunk_size):
        chunk = words[index:index + chunk_size]
        chunks.append(" ".join(chunk))

    return chunks
