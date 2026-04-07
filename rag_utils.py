import math
from typing import List, Tuple

from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(chunks: List[str]) -> List[List[float]]:
    embeddings = model.encode(chunks, convert_to_tensor=False)
    return [embedding.tolist() for embedding in embeddings]


def generate_query_embedding(query: str) -> List[float]:
    embedding = model.encode(query, convert_to_tensor=False)
    return embedding.tolist()


def cosine_similarity(vector_a: List[float], vector_b: List[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def semantic_search_chunks(
    query: str,
    chunks: List[str],
    embeddings: List[List[float]],
    max_results: int = 3,
) -> List[Tuple[str, float]]:
    if not query.strip() or not chunks or not embeddings:
        return []

    query_embedding = generate_query_embedding(query)
    scored_chunks = []

    for chunk, embedding in zip(chunks, embeddings):
        score = cosine_similarity(query_embedding, embedding)
        scored_chunks.append((chunk, score))

    scored_chunks.sort(key=lambda item: item[1], reverse=True)
    return scored_chunks[:max_results]
