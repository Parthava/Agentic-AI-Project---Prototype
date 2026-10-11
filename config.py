"""Shared config: env vars + LLM + Tavily client."""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from tavily import TavilyClient

load_dotenv(override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

model = init_chat_model("groq:openai/gpt-oss-20b", temperature=0.1)
tavily = TavilyClient(api_key=TAVILY_API_KEY)

# Paths / constants
PDF_DIR = "data"
VECTOR_STORE_DIR = "data/vector_store"
COLLECTION_NAME = "pdf_documents"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

FIXED_CURRENT_AFFAIRS_QUERY = (
    "recently 2024 2025 launched held awarded operation summit "
    "budget g20 g7 brics bimstec unfccc unccd cbd "
)
