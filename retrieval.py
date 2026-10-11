"""RAGRetriever + MCQ parsing."""

import re
from typing import Any

from embedding import EmbeddingManager
from vector_store import VectorStore


class RAGRetriever:
    """Query-based retrieval from the vector store."""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def clear_db(self) -> None:
        try:
            all_data = self.vector_store.collection.get()
            if all_data["ids"]:
                self.vector_store.collection.delete(ids=all_data["ids"])
                print(f"Cleared {len(all_data['ids'])} docs from DB")
        except Exception as e:  # noqa: BLE001
            print(f"Clear error: {e}")

    def retrieve(
        self, query: str, top_k: int = 100, score_threshold: float = 0.0
    ) -> list[dict[str, Any]]:
        print(f"Retrieving for: '{query}' | top_k={top_k}")

        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()], n_results=top_k
            )

            retrieved_docs: list[dict[str, Any]] = []
            seen_q_ids: set[str] = set()

            if results["documents"] and results["documents"][0]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                ids = results["ids"][0]

                for doc_id, document, metadata, distance in zip(
                    ids, documents, metadatas, distances
                ):
                    similarity_score = 1 - distance
                    q_id = metadata.get("q_id")

                    if q_id in seen_q_ids:
                        continue

                    if similarity_score >= score_threshold:
                        seen_q_ids.add(q_id)
                        retrieved_docs.append(
                            {
                                "id": doc_id,
                                "content": document,
                                "metadata": metadata,
                                "similarity_score": similarity_score,
                                "distance": distance,
                                "rank": len(retrieved_docs) + 1,
                            }
                        )

                print(f"Retrieved {len(retrieved_docs)} unique documents")
            else:
                print("No documents found")

            return retrieved_docs

        except Exception as e:  # noqa: BLE001
            print(f"Error during retrieval: {e}")
            return []


# ---------- MCQ parsing ----------
def parse_mcq(content: str) -> dict:
    """Raw MCQ text -> {question_text, statements, options}."""
    content = re.sub(r"^Q\d+\.\s*", "", content).strip()

    parts = re.split(r"Options:\s*", content, flags=re.IGNORECASE)
    main_text = parts[0]

    options: list[str] = []
    if len(parts) > 1:
        options = re.findall(
            r"\([A-D]\)\s*(.+?)(?=\s*\([A-D]\)|$)", parts[1], flags=re.DOTALL
        )

    statements = re.findall(r"\([ivxl]+\)\s*[^\n()]+", main_text, flags=re.IGNORECASE)

    clean_statements = []
    for s in statements:
        if "Select the correct" in s:
            s = s.split("Select the correct")[0]
        clean_statements.append(s.strip())

    q_text = re.split(r"\(i\)", main_text, flags=re.IGNORECASE)[0].strip()

    return {
        "question_text": q_text,
        "statements": clean_statements,
        "options": [o.strip() for o in options],
    }


def build_rag_payload(response: list[dict]) -> list[dict]:
    """Retrieved docs -> parsed MCQ payload."""
    final_output = []
    for d in response:
        if not isinstance(d, dict):
            continue
        content = d.get("content") or d.get("page_content")
        if not content:
            continue
        meta = d.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}

        try:
            parsed = parse_mcq(content)
        except Exception as e:  # noqa: BLE001
            print(f"Parse failed for {meta.get('q_id', 'unknown')}: {e}")
            continue

        final_output.append(
            {
                "q_id": meta.get("q_id", "unknown"),
                "year": meta.get("year"),
                "q_number": meta.get("q_number"),
                "source_file": meta.get("source_file"),
                "question_text": parsed.get("question_text", ""),
                "statements": parsed.get("statements", []),
                "options": parsed.get("options", []),
                "similarity_score": d.get("similarity_score", 1.0),
                "distance": d.get("distance", 0.0),
            }
        )
    return final_output
