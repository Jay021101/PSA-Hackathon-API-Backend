import pandas as pd
import numpy as np
import requests
import os
import time
import re
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
from pydantic import BaseModel

# === Initialize FastAPI ===
app = FastAPI()

# === File Paths ===
# === File Paths ===
BASE_DIR = os.path.dirname(__file__)
KB_FILE = os.path.join(BASE_DIR, "data", "KB.xlsx")
KB_CACHE = os.path.join(BASE_DIR, "data", "kb_with_embeddings.pkl")
CASE_LOG_FILE = os.path.join(BASE_DIR, "data", "Case log with feedback.xlsx")
CASE_LOG_CACHE = os.path.join(BASE_DIR, "data", "case_log_with_embeddings.pkl")

# === Azure API Endpoints ===
AZURE_URL_EMBED = "https://psacodesprint2025.azure-api.net/openai/deployments/text-embedding-3-small/embeddings?api-version=2025-01-01-preview"
AZURE_URL_CHAT = "https://psacodesprint2025.azure-api.net/openai/deployments/gpt-4.1-nano/chat/completions?api-version=2025-01-01-preview"
AZURE_API_KEY = os.getenv("AZURE_API_KEY", "")

HEADERS = {
    "Content-Type": "application/json",
    "api-key": AZURE_API_KEY
}

# === Helper: Embedding ===
def get_embedding(text, retries=5, backoff_base=2, sleep_min=1):
    if not text or not isinstance(text, str):
        text = ""
    text = text[:4000]
    payload = {"input": text}

    for attempt in range(retries):
        try:
            r = requests.post(AZURE_URL_EMBED, headers=HEADERS, json=payload)
            if r.status_code == 200:
                return r.json()["data"][0]["embedding"]
            elif r.status_code == 429:
                time.sleep(sleep_min * (backoff_base ** attempt))
            else:
                r.raise_for_status()
        except requests.RequestException:
            time.sleep(sleep_min * (backoff_base ** attempt))
    raise Exception("❌ Embedding failed after retries")

# === Helper: Chat ===
def chat_completion(prompt, retries=3, backoff_base=2, sleep_min=1):
    payload = {"messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
    for attempt in range(retries):
        try:
            r = requests.post(AZURE_URL_CHAT, headers=HEADERS, json=payload)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            elif r.status_code == 429:
                time.sleep(sleep_min * (backoff_base ** attempt))
            else:
                r.raise_for_status()
        except requests.RequestException:
            time.sleep(sleep_min * (backoff_base ** attempt))
    raise Exception("❌ Chat request failed after retries")

# === Load Embeddings Once at Startup ===
def load_embeddings():
    if os.path.exists(KB_CACHE):
        kb = pd.read_pickle(KB_CACHE)
    else:
        kb = pd.read_excel(KB_FILE)
        kb["combined"] = kb["title"].fillna("") + " " + kb["text"].fillna("")
        kb["embedding"] = [get_embedding(t) for t in kb["combined"]]
        kb.to_pickle(KB_CACHE)

    if os.path.exists(CASE_LOG_CACHE):
        case_log = pd.read_pickle(CASE_LOG_CACHE)
    else:
        case_log = pd.read_excel(CASE_LOG_FILE)
        case_log["combined"] = case_log["Problem Statements"].fillna("") + " " + case_log["Solution"].fillna("")
        case_log["embedding"] = [get_embedding(t) for t in case_log["combined"]]
        case_log.to_pickle(CASE_LOG_CACHE)

    return kb, case_log

kb, case_log = load_embeddings()

# === Input Schema ===
class CaseInput(BaseModel):
    description: str

# === API Endpoint ===
@app.post("/analyze")
def analyze_case(req: CaseInput):
    case_text = req.description
    case_emb = get_embedding(case_text)

    # Find top KB entries
    kb_sims = [cosine_similarity([e], [case_emb])[0][0] for e in kb["embedding"]]
    kb_top_idx = np.argsort(kb_sims)[-3:][::-1]
    kb_top_entries = [
        f"{j+1}. {kb.iloc[idx]['title']} — {kb.iloc[idx]['text']}" for j, idx in enumerate(kb_top_idx)
    ]

    # Find top case log entries
    cl_sims = [cosine_similarity([e], [case_emb])[0][0] for e in case_log["embedding"]]
    cl_top_idx = np.argsort(cl_sims)[-3:][::-1]
    cl_top_entries = [
        f"{j+1}. Problem: {case_log.iloc[idx]['Problem Statements']}\n   Solution: {case_log.iloc[idx]['Solution']}"
        for j, idx in enumerate(cl_top_idx)
    ]

    # Build GPT Prompt
    prompt = f"""
You are a technical support assistant.

A new case has been reported:
{case_text}

=== KNOWLEDGE BASE (PRIMARY) ===
{chr(10).join(kb_top_entries)}

=== CASE LOG (SECONDARY) ===
{chr(10).join(cl_top_entries)}

TASK:
- Derive the final solution primarily from the Knowledge Base entries.
- Provide a clear, concise, step-by-step solution (3–6 sentences).
"""

    # Generate solution
    solution = chat_completion(prompt).strip()

    # Determine Module
    first_title = kb.iloc[kb_top_idx[0]]["title"]
    match = re.match(r"^\s*([A-Za-z0-9/_-]+)\s*:", first_title)
    prefix = match.group(1).upper() if match else ""
    if prefix in ["EDI", "API", "EDI/API"]:
        module = "EDI/API"
    elif prefix == "VSL":
        module = "Vessel"
    elif prefix in ["IMPORT", "EXPORT", "IMPORT/EXPORT"]:
        module = "IMPORT/EXPORT"
    elif prefix in ["CNTR", "CONTAINER"]:
        module = "Container Report"
    else:
        module = "IMPORT/EXPORT"

    return {
        "solution": solution,
        "module": module,
        "kb_titles": "\n".join(f"{j+1}. {kb.iloc[idx]['title']}" for j, idx in enumerate(kb_top_idx)),
        "case_refs": "\n".join(f"{j+1}. {case_log.iloc[idx]['Problem Statements']}" for j, idx in enumerate(cl_top_idx))
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
