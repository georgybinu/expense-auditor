import math
from typing import List, Tuple

import faiss
import numpy as np
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


def build_faiss_index(embeddings: List[List[float]]) -> faiss.IndexFlatL2:
    if not embeddings:
        raise ValueError("Embeddings list cannot be empty")

    embedding_array = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatL2(embedding_array.shape[1])
    index.add(embedding_array)
    return index


def search_faiss_index(
    query: str,
    index: faiss.Index,
    chunks: List[str],
    top_k: int = 2,
) -> List[str]:
    if not query.strip() or not chunks:
        return []

    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    _, indices = index.search(query_embedding, top_k)

    results = []
    for chunk_index in indices[0]:
        if 0 <= chunk_index < len(chunks):
            results.append(chunks[chunk_index])

    return results
