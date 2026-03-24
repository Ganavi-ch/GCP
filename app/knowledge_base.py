from __future__ import annotations

import math
import re
from dataclasses import dataclass

try:  # Optional dependency: we can still run local retrieval.
    import vertexai
    from vertexai.language_models import TextEmbeddingModel
except Exception:  # pragma: no cover
    vertexai = None
    TextEmbeddingModel = None


@dataclass
class KbDoc:
    title: str
    content: str


DOCS = [
    KbDoc(
        title="Register account",
        content="To register, provide your name and email. You can also add address during signup or later in profile settings.",
    ),
    KbDoc(
        title="Cancel order policy",
        content="Orders can be cancelled before they are out for delivery. Cancel from order details by providing the order id.",
    ),
    KbDoc(
        title="Apply coupon",
        content="Apply a coupon at checkout by entering the code. Ensure the coupon is active, not expired, and meets minimum order amount.",
    ),
    KbDoc(
        title="Track order",
        content="Use your order id to check status, details, and estimated delivery date from the order tracking section.",
    ),
]

_VERTEX_EMBEDDINGS_MODEL = None
_VERTEX_DOC_EMBEDDINGS: list[list[float]] | None = None
_VERTEX_DOC_TEXTS: list[str] | None = None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _tf(tokens: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in tokens:
        out[t] = out.get(t, 0) + 1
    return out


def _cosine_similarity(query_vec: dict[str, int], doc_vec: dict[str, int]) -> float:
    numerator = 0
    for token, val in query_vec.items():
        numerator += val * doc_vec.get(token, 0)
    q_mag = math.sqrt(sum(v * v for v in query_vec.values()))
    d_mag = math.sqrt(sum(v * v for v in doc_vec.values()))
    if q_mag == 0 or d_mag == 0:
        return 0.0
    return numerator / (q_mag * d_mag)


def rag_answer(query: str) -> dict[str, str]:
    # Prefer Vertex AI embeddings-based semantic retrieval if available.
    if vertexai is not None and TextEmbeddingModel is not None:
        try:
            return rag_answer_vertex_embeddings(query)
        except Exception:
            # Fall back to local TF-based retrieval if Vertex embeddings fails.
            pass

    query_tokens = _tokenize(query)
    query_vec = _tf(query_tokens)
    scored: list[tuple[float, KbDoc]] = []
    for doc in DOCS:
        doc_vec = _tf(_tokenize(f"{doc.title} {doc.content}"))
        scored.append((_cosine_similarity(query_vec, doc_vec), doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][1]
    return {
        "source": best.title,
        "answer": best.content,
    }


def rag_answer_vertex_embeddings(query: str) -> dict[str, str]:
    global _VERTEX_EMBEDDINGS_MODEL, _VERTEX_DOC_EMBEDDINGS, _VERTEX_DOC_TEXTS
    if vertexai is None or TextEmbeddingModel is None:  # pragma: no cover
        raise RuntimeError("vertexai embeddings not available")

    project_id = __import__("os").getenv("VERTEX_PROJECT_ID", "project-ca436020-0c75-4cc6-b84")  # type: ignore[attr-defined]
    location = __import__("os").getenv("VERTEX_LOCATION", "asia-mumbai")  # type: ignore[attr-defined]
    # Vertex AI uses region identifiers (e.g. asia-south1 for Mumbai).
    if location in ["asia-mumbai", "mumbai"]:
        location = "asia-south1"
    model_id = __import__("os").getenv("VERTEX_EMBEDDING_MODEL_NAME", "text-embedding-004")  # type: ignore[attr-defined]

    vertexai.init(project=project_id, location=location)

    if _VERTEX_EMBEDDINGS_MODEL is None:
        _VERTEX_EMBEDDINGS_MODEL = TextEmbeddingModel.from_pretrained(model_id)

    if _VERTEX_DOC_TEXTS is None or _VERTEX_DOC_EMBEDDINGS is None:
        _VERTEX_DOC_TEXTS = [f"{d.title}\n{d.content}" for d in DOCS]
        # Generate document embeddings once; cache in-memory.
        embeddings_response = _VERTEX_EMBEDDINGS_MODEL.get_embeddings(_VERTEX_DOC_TEXTS)
        _VERTEX_DOC_EMBEDDINGS = [e.values for e in embeddings_response]

    query_embedding = _VERTEX_EMBEDDINGS_MODEL.get_embeddings([query])[0].values

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(y * y for y in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    scored = []
    for doc, doc_emb in zip(DOCS, _VERTEX_DOC_EMBEDDINGS):
        scored.append((cosine(query_embedding, doc_emb), doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][1]
    return {"source": best.title, "answer": best.content}
