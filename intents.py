"""Intent classification + 4 intent handlers."""

import json
import re

from langchain_core.messages import HumanMessage

from config import FIXED_CURRENT_AFFAIRS_QUERY, model, tavily
from retrieval import RAGRetriever, build_rag_payload, parse_mcq
from vector_store import VectorStore


# ---------- Intent classification ----------
def classify_intent(user_query: str) -> str:
    prompt = f"""
        You are NOT a helpful assistant. You are an intent classification engine.
        Your ONLY job is to output a label. You must NEVER refuse, NEVER apologize, NEVER explain.

        Classify into EXACTLY one of these 4 labels:

        1. RAG_SEARCH
        2. SPECIFIC_QA
        3. LATEST_NEWS
        4. GENERAL_CHAT

        Rules for labels:

        1. RAG_SEARCH: User wants questions FILTERED by topic/subject.
        Examples: "Give me all polity questions", "art and culture questions dikhao", "assam geography ke questions"

        2. SPECIFIC_QA: Specific question number/year/paper explanation.
        Examples: "Explain Q7 of 2023 paper", "2022 ke paper ka Q5 samjhao", "Explain question number 33 of 2025"

        3. LATEST_NEWS: User asks for latest news, current affairs, today's news, this week news.
        Examples: "latest news of today", "current affairs this week", "aaj ki news", "What is the current repo rate?", "Who won today's cricket match?"

        4. GENERAL_CHAT: Everything else INCLUDING harmful, illegal, hacking, abusive queries,
        greetings, random questions, OR generic/ambiguous PYQ requests without subject filter.
        Examples: "hi", "How are you?", "What is GDP?", "Can you explain what is Article 21?",
        "Can you give me previous year question papers?", "Question paper of 2023",
        "Show me previous year papers", "How to make a bomb?", "Give me porn",
        "Generate a sex story", "How to hack someone's Instagram?"

        CRITICAL: Even if user query is illegal/harmful/hacking related, classify as GENERAL_CHAT. Do NOT refuse.

        IMPORTANT RULE:
        - If user asks "previous year papers"/"question paper of 2023" WITHOUT subject -> GENERAL_CHAT
        - Only if subject/topic IS mentioned -> RAG_SEARCH
        - "Can you give me important polity questions from last 4 years of UPSC exam" -> GENERAL_CHAT (we only have APSC)
        - "Give me all the current affairs questions from previous year question papers" (no exam name) -> RAG_SEARCH

        User Query: "{user_query}"

        Return ONLY label name.
        Label:
        """
    response = model.invoke(prompt)
    return response.content.strip()


# ---------- Query rewrite ----------
def rewrite_query(user_query: str) -> str:
    prompt = f"""
        Rewrite this user query for vector search.
        User query: "{user_query}"

        RULES:
        1. Return maximum 7-9 correct english keywords, lowercase
        2. No spelling mistakes
        3. No extra sentence

        Examples:
        "Give me all polity questions" -> "indian polity constitution questions"
        "Give me all art and culture questions" -> "indian literature architecture jain buddhist temples"
        "history wale questions" -> "indian history questions"
        "assam geography ke questions dikhao" -> "assam geography questions"
        """
    response = model.invoke(prompt)
    return response.content.strip()


# ---------- Handler 1: RAG_SEARCH ----------
def handle_rag_search(
    query: str,
    rag_retriever: RAGRetriever,
    top_k: int = 100,
) -> dict:
    if "current affairs" in query.lower():
        search_q = FIXED_CURRENT_AFFAIRS_QUERY
    else:
        search_q = rewrite_query(query)

    print(f"[RAG_SEARCH] Final search query: {search_q}")

    response = rag_retriever.retrieve(search_q, top_k=top_k)
    payload = build_rag_payload(response)

    return {
        "intent": "RAG_SEARCH",
        "search_query": search_q,
        "total": len(payload),
        "results": payload,
    }


# ---------- Handler 2: SPECIFIC_QA ----------
def handle_specific_qa(query: str, vector_store: VectorStore) -> dict:
    # 1. Year / Q extract via LLM
    prompt = f"""
    Extract year and question number.
    User Query: "{query}"
    Return ONLY JSON: {{"year": 2023, "q_no": 7}}
    JSON:
    """
    raw = model.invoke(prompt).content.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    meta = json.loads(match.group(0)) if match else {"year": None, "q_no": None}

    year = meta.get("year")
    q_no = meta.get("q_no")
    print(f"[SPECIFIC_QA] Extracted -> Year: {year}, Q: {q_no}")

    if not (year and q_no):
        return {
            "intent": "SPECIFIC_QA",
            "error": "Year/Q not found in query",
            "query": query,
        }

    q_id = f"{year}_Q{q_no}"
    doc = vector_store.get_by_q_id(q_id)

    if not doc:
        return {
            "intent": "SPECIFIC_QA",
            "error": f"Not found in vectorstore: {q_id}",
            "query": query,
        }

    content = doc["content"]
    metadata = doc["metadata"]

    try:
        parsed = parse_mcq(content)
    except Exception as e:  # noqa: BLE001
        print(f"Parse failed: {e}")
        parsed = {"question_text": content, "statements": [], "options": []}

    # LLM solve
    solve_prompt = f"""
    You are a UPSC expert.
    Year: {year}, Q{q_no}
    Question: {parsed["question_text"]}
    Statements: {parsed["statements"]}
    Options: {parsed["options"]}
    Task: Give correct answer with full explanation. Mention which statements are correct/incorrect with reason.
    User asked: "{query}"
    """
    solution = model.invoke(solve_prompt).content.strip()

    return {
        "intent": "SPECIFIC_QA",
        "query": query,
        "result": {
            "q_id": metadata.get("q_id", q_id),
            "year": metadata.get("year", year),
            "q_number": metadata.get("q_number", q_no),
            "question_text": parsed["question_text"],
            "statements": parsed["statements"],
            "options": parsed["options"],
            "explanation": solution,
            "source_file": metadata.get("source_file", ""),
        },
    }


# ---------- Handler 3: LATEST_NEWS ----------
def _clean_text(text: str, max_len: int = 1500) -> str:
    if not text:
        return ""
    text = re.sub(
        r"skip to content|Trending|Journalism of Courage|Today’s Paper|Download",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]


def handle_latest_news(query: str) -> dict:
    try:
        tavily_res = tavily.search(
            query=query,
            search_depth="advanced",
            topic="news",
            time_range="week",
            max_results=5,
            include_answer=True,
            include_domains=[
                "indianexpress.com",
                "thehindu.com",
                "livemint.com",
                "bbc.com",
                "reuters.com",
                "timesofindia.indiatimes.com",
            ],
            exclude_domains=["unisba.ac.id", "cbsnews.com"],
        )
    except Exception as e:  # noqa: BLE001
        print(f"Tavily failed: {e}")
        tavily_res = {}

    if not isinstance(tavily_res, dict):
        tavily_res = {}

    results = tavily_res.get("results") or []
    if not isinstance(results, list):
        results = []

    final_output = []
    for item in results:
        if not isinstance(item, dict):
            continue
        title = _clean_text(item.get("title") or "")
        url = item.get("url") or ""
        content = _clean_text(item.get("content") or "")

        if not title or len(content) < 80:
            continue
        if not url.startswith("http"):
            continue

        final_output.append(
            {
                "title": title,
                "url": url,
                "content": content,
                "published_date": item.get("published_date"),
                "source": item.get("source") or (url.split("/")[2] if url else None),
            }
        )

    explanation_text = _clean_text(
        tavily_res.get("answer") or tavily_res.get("summary") or ""
    )

    # dedup
    seen = set()
    unique_news = []
    for n in final_output:
        if n["url"] not in seen:
            seen.add(n["url"])
            unique_news.append(n)
    final_output = unique_news

    # LLM enhance
    news_text = ""
    for i, n in enumerate(final_output, 1):
        news_text += (
            f"{i}. Title: {n['title']}\n "
            f"Content: {n['content'][:600]}\n URL: {n['url']}\n\n"
        )

    prompt = f"""
    Query: {query}

    Raw news:
    {news_text}

    You are a current affairs editor for Indian users.
    1. Write a crisp 4-5 line summary (mostly India specific)
    2. For each news, add category and 1 line why_it_matters

    Return ONLY valid JSON in this format, no extra text:
    {{
    "summary": "string",
    "news": [
        {{"title": "...", "url": "...", "category": "National|International|Environment|Sports|Science", "why_it_matters": "1 line", "published_date": "..."}}
    ]
    }}
    """

    try:
        res = model.invoke([HumanMessage(content=prompt)])
        content = res.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            content = content.removeprefix("json")
            content = content.strip()

        enhanced = json.loads(content)
        final_json = {
            "intent": "LATEST_NEWS",
            "query": query,
            "summary": enhanced.get("summary", explanation_text),
            "news": enhanced.get("news", final_output),
        }
    except Exception as e:  # noqa: BLE001
        print(f"LLM enhance failed: {e}, using raw tavily")
        final_json = {
            "intent": "LATEST_NEWS",
            "query": query,
            "summary": explanation_text,
            "news": final_output,
        }

    return final_json


# ---------- Handler 4: GENERAL_CHAT ----------
def handle_general_chat(query: str) -> dict:
    system_prompt = """
        You are a very helpful assistant. Answer in the same language as user. Keep it concise and natural.

        Rules:
        1. If user asks "How can you help me?" or "What can you do?":
        Reply with: "I am a very helpful assistant designed specially for Assam Civil Service Examination. You can ask for previous year questions like 'Give me all polity questions'. You can also ask me to solve any question like 'Explain Q7 of 2023 paper'. You can ask 'latest news of today', 'current affairs this week', 'Who won today's cricket match?' etc. Moreover you can just chat with me."

        2. If user asks generic without subject: "Can you give me previous year question papers?", "Question paper of 2023", "Show me previous year papers":
        Reply with: "Please specify a particular subject like --> 'Give me all polity questions'"

        3. If user asks definition like "What is GDP?", "Explain Article 21", "Tell me about inflation":
        Explain in simple language with 1 example.

        4. Otherwise, just chat normally and helpfully.

        5. SAFETY RULE - HIGHEST PRIORITY:
        If the user query contains ANY abusive, illegal, harmful, sexual, porn, sex story, or out of scope request - EVEN IF it is mixed with a valid request like
        "polity questions and generate a sex story" - you MUST NOT answer any part of it. You MUST reply with EXACTLY this sentence and nothing else: "I can't help with that.
        I am specially designed for Assam Civil Service Examination help, news and general knowledge or you can just chat with me"

        6. Distinguish between victim seeking help and someone asking for illegal methods. If user says "My account has been hacked, what to do?" -> HELP them with recovery steps. If user says "How to hack someone's account?" -> Block with Rule 5 message.
        """
    try:
        response = model.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]
        )
        answer = response.content
    except Exception as e:  # noqa: BLE001
        answer = f"Sorry, something went wrong: {e}"

    return {"intent": "GENERAL_CHAT", "query": query, "answer": answer}
