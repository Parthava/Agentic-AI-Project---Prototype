"""VectorStore: ChromaDB wrapper."""

import os
import uuid
from typing import Any

import chromadb
import numpy as np

from config import COLLECTION_NAME, VECTOR_STORE_DIR


class VectorStore:
    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        persist_directory: str = VECTOR_STORE_DIR,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self) -> None:
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG"},
            )
            print(f"Vector store ready. Collection: {self.collection_name}")
            print(f"Existing docs: {self.collection.count()}\n")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: list[Any], embeddings: np.ndarray) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        print(f"Adding {len(documents)} documents to vector store...")

        ids, metadatas, documents_text, embeddings_list = [], [], [], []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            ids.append(f"doc_{uuid.uuid4().hex[:8]}_{i}")

            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadatas.append(metadata)

            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        try:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text,
            )
        except Exception as e:
            print(f"Error adding documents: {e}")
            raise

    def get_by_q_id(self, q_id: str) -> dict | None:
        """Specific question (SPECIFIC_QA intent)."""
        result = self.collection.get(where={"q_id": q_id})
        if not result["documents"]:
            return None
        return {
            "content": result["documents"][0],
            "metadata": result["metadatas"][0],
        }
