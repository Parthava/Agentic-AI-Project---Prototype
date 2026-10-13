"""
FastAPI app.

Run:
    uvicorn main:app --reload --port 8000

Endpoints:
    GET  /            -> health
    POST /index       -> PDFs se index banao (ek baar)
    POST /query       -> body: {"query": "...", "top_k": 100}
    POST /clear       -> DB clear
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chunking import chunk_qwise
from embedding import EmbeddingManager
from ingestion import process_all_pdfs_qwise
from intents import (
    classify_intent,
    handle_general_chat,
    handle_latest_news,
    handle_rag_search,
    handle_specific_qa,
)
from retrieval import RAGRetriever
from vector_store import VectorStore

# ---------- Globals ----------
embedding_manager: EmbeddingManager | None = None
vector_store: VectorStore | None = None
rag_retriever: RAGRetriever | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: model + store load (once)."""
    global embedding_manager, vector_store, rag_retriever

    print("🚀 Starting up...")
    embedding_manager = EmbeddingManager()
    embedding_manager.load_model()
    vector_store = VectorStore()
    rag_retriever = RAGRetriever(vector_store, embedding_manager)
    print("✅ Ready!")

    yield

    print("🛑 Shutting down...")


app = FastAPI(title="Agentic AI Project Prototype", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # specific domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Schemas ----------
class QueryRequest(BaseModel):
    query: str = Field(..., description="User query from frontend")
    top_k: int = Field(100, ge=1, le=500)


class QueryResponse(BaseModel):
    intent: str
    query: str
    data: dict


class IndexResponse(BaseModel):
    status: str
    chunks_added: int
    total_in_store: int


# ---------- Routes ----------
@app.get("/")
def health():
    return {"status": "ok", "message": "Agentic AI Project Prototype"}


@app.post("/index", response_model=IndexResponse)
def build_index():
    """PDFs ingest -> chunk -> embed -> store."""
    if embedding_manager is None or vector_store is None:
        raise HTTPException(503, "Service not ready")

    try:
        all_docs = process_all_pdfs_qwise("data")
        chunks = chunk_qwise(all_docs)

        texts = [doc.page_content for doc in chunks]
        embeddings = embedding_manager.generate_embeddings(texts)
        vector_store.add_documents(chunks, embeddings)

        return IndexResponse(
            status="success",
            chunks_added=len(chunks),
            total_in_store=vector_store.collection.count(),
        )
    except Exception as e:
        raise HTTPException(500, f"Indexing failed: {e}") from e


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    """Frontend query -> classify -> route to handler."""
    if embedding_manager is None or vector_store is None or rag_retriever is None:
        raise HTTPException(503, "Service not ready")

    try:
        intent = classify_intent(req.query)
        print(f"\n[INTENT] {intent} | query: {req.query}")

        if "RAG_SEARCH" in intent:
            data = handle_rag_search(req.query, rag_retriever, top_k=req.top_k)

        elif "SPECIFIC_QA" in intent:
            data = handle_specific_qa(req.query, vector_store)

        elif "LATEST_NEWS" in intent:
            data = handle_latest_news(req.query)

        else:  # GENERAL_CHAT fallback
            data = handle_general_chat(req.query)

        return QueryResponse(intent=intent, query=req.query, data=data)

    except Exception as e:
        raise HTTPException(500, f"Query failed: {e}") from e


# @app.post("/clear")
# def clear_db():
#     if rag_retriever is None or vector_store is None:
#         raise HTTPException(503, "Service not ready")
#     rag_retriever.clear_db()
#     return {"status": "cleared", "remaining": vector_store.collection.count()}
