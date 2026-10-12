# 🎯 Agentic-AI-Project---Prototype

A **Retrieval-Augmented Generation (RAG)** powered API for **Assam Civil Service (APSC)** exam preparation. It lets users:

- 📚 Retrieve previous year MCQ questions filtered by subject (polity, history, geography, etc.)
- 🧠 Solve any specific question (e.g. "Explain Q7 of 2023 paper") with a detailed explanation
- 📰 Fetch latest news & current affairs (India-focused)
- 💬 Chat generally with an AI assistant

Built with **FastAPI + LangChain + ChromaDB + SentenceTransformer + Groq LLM + Tavily**.

---

## 📁 Project Structure

```
project/
│
├── config.py          # Env vars, LLM init, Tavily client, constants
├── ingestion.py       # PDF → question-wise JSON docs
├── chunking.py        # Long questions → chunks (LangChain Documents)
├── embedding.py       # EmbeddingManager (SentenceTransformer wrapper)
├── vector_store.py    # VectorStore (ChromaDB wrapper)
├── retrieval.py       # RAGRetriever + MCQ parser
├── intents.py         # Intent classifier + 4 intent handlers
├── main.py            # FastAPI app (entry point)
├── requirements.txt
├── .env               # API keys (NOT pushed to git)
└── data/              # Place your APSC PDFs here
```

---

## ⚙️ Setup

### 1. Clone & install dependencies

```bash
git clone <your-repo-url>
cd project
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Create `.env` file

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Get your keys:

- Groq → https://console.groq.com/keys
- Tavily → https://app.tavily.com/

### 3. Add PDFs

Place your APSC previous year question paper PDFs inside the `data/` folder.

> **Important:** The filename must contain the year, e.g. `APSC_2023.pdf`, `2022_prelims.pdf`.

> **Note:** For better processing, re-create the question papers so each question sits on a single line. You can do this with the help of Claude: upload the PDF and use this prompt:
>
> *"Process this PDF and write each question on a single line so that there are no issues when I ingest the data for the job. Create a new PDF file."*

---

## 🚀 Running the API

```bash
uvicorn main:app --reload --port 8000
```

- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## 📡 API Endpoints

### 1. `GET /` — Health check

```bash
curl http://localhost:8000/
```

Response:

```json
{ "status": "ok", "message": "Agentic AI Project Prototype" }
```

### 2. `POST /index` — Build / rebuild the vector index

Extracts questions from all PDFs in `data/`, chunks them, embeds them, and stores them in ChromaDB.

```bash
curl -X POST http://localhost:8000/index
```

Response:

```json
{
  "status": "success",
  "chunks_added": 402,
  "total_in_store": 402
}
```

> ⚠️ Run this once after setup, or whenever you add/change PDFs.

### 3. `POST /query` — Main query endpoint

Body:

```json
{
  "query": "Give me all polity questions",
  "top_k": 100
}
```

Example:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Give me all polity questions", "top_k": 100}'
```

The API auto-classifies the query into one of 4 intents and routes it:

| Intent | Trigger Examples | What it does |
| :--- | :--- | :--- |
| `RAG_SEARCH` | "Give me all polity questions", "art and culture questions" | Retrieves MCQ questions filtered by subject |
| `SPECIFIC_QA` | "Explain Q7 of 2023 paper" | Fetches that exact question + LLM generates solution |
| `LATEST_NEWS` | "latest news of today", "current affairs this week" | Tavily news search + LLM summary |
| `GENERAL_CHAT` | "hi", "What is GDP?", "How to hack?" | Helpful chat / safety refusal |

---

## 🧠 How It Works

```
┌─────────────┐
│  PDF Files  │
└──────┬──────┘
       │ ingestion.py
       ▼
┌──────────────────────┐
│ Question-wise Docs   │  (JSON: {content, metadata})
└──────┬───────────────┘
       │ chunking.py
       ▼
┌──────────────────────┐
│ LangChain Documents  │  (chunked if >1000 chars)
└──────┬───────────────┘
       │ embedding.py
       ▼
┌──────────────────────┐
│  Vector Embeddings   │  (BAAI/bge-base-en-v1.5)
└──────┬───────────────┘
       │ vector_store.py
       ▼
┌──────────────────────┐
│   ChromaDB (local)   │
└──────┬───────────────┘
       │
       │   ┌───────────────────────┐
       │   │  User Query (FastAPI) │
       │   └──────────┬────────────┘
       │              │ intents.py
       │              ▼
       │   ┌───────────────────────┐
       │   │  classify_intent()    │
       │   └──────────┬────────────┘
       │              │
       ├──RAG_SEARCH──┤──SPECIFIC_QA──┬──LATEST_NEWS──┬──GENERAL_CHAT
       ▼              ▼               ▼                ▼
   Retrieval    Direct lookup     Tavily API      Groq LLM
   + parse      + LLM solve        + LLM            chat
```

---

## 📝 Notes

- Never commit your `.env` file — add it to `.gitignore`.
- Re-run `POST /index` after adding or changing any PDF in `data/`.
