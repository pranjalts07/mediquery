<div align="center">

# MediQuery

![Python](https://img.shields.io/badge/Python-3.11-3776ab?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-6c47ff?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Inference_API-ff9d00?style=flat-square&logo=huggingface&logoColor=white)
![Azure](https://img.shields.io/badge/Azure-App_Service-0078d4?style=flat-square&logo=microsoftazure&logoColor=white)

**[Live Demo](https://mediquery-app.azurewebsites.net)**

<br/>

<img src="images/image1.png" alt="MediQuery Screenshot" width="800"/>

</div>

---

A medical question-answering system built on a two-stage retrieval pipeline. Questions are answered using real PubMed abstracts and medical literature — not just what the model already knows.

## How it works

```
User question
     │
     ▼
Safety filter
     │
     ▼
Embed → 384-dim vector  (all-MiniLM-L6-v2)
     │
     ▼
Pinecone top-8 candidates  (cosine similarity ≥ 0.35)
     │
     ▼
Cross-encoder reranks → top 3  (ms-marco-MiniLM-L-6-v2)
     │
     ▼
Llama 3.1-8B generates answer with source citations
```

The bi-encoder retrieves fast but approximate candidates. The cross-encoder sees the full query-passage pair and picks the 3 most relevant chunks, which reduces the hallucination rate by giving the LLM a tighter, more relevant context window.

## Evaluation

Evaluated across 25 questions spanning cardiovascular, metabolic, oncology, immunology, and mental health topics.

| Metric | Score |
|---|---|
| Keyword Recall | 61.4% |
| Ground-Truth Overlap | 95.6% |
| Source-Supported | 41.8% |
| Semantic Similarity | 74.6% |
| **Overall** | **0.684 / B** |
| Median Latency | 1.3s |

Run `python scripts/evaluate_mediquery.py --url <server>` to reproduce against a live instance.

## Features

- Two-stage RAG: bi-encoder recall + cross-encoder reranking
- Streaming responses (SSE) — tokens appear as they're generated
- Conversation memory — references previous messages for follow-ups
- Brief / Detailed mode toggle
- Source citations with similarity score on every answer
- Voice input (Chrome)
- Export conversation to PDF
- Two-tier safety filter — separates general medical questions from active emergencies
- Rate limiting, CSP headers, input sanitization

## Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + gunicorn/uvicorn |
| Vector DB | Pinecone (384-dim, cosine) |
| Embeddings | `all-MiniLM-L6-v2` via HF Inference API |
| Reranker | `ms-marco-MiniLM-L-6-v2` via HF Inference API |
| LLM | `Llama-3.1-8B-Instruct` via Cerebras (HF) |
| Deployment | Azure App Service |

No LangChain — the full pipeline is about 200 lines of Python.

## Local setup

```bash
git clone https://github.com/pranjalts07/mediquery.git
cd mediquery
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:

```env
HF_API_TOKEN=hf_...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=mediquery
PINECONE_HOST=https://mediquery-xxx.svc.aped-xxxx-xxxx.pinecone.io
HF_LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct:cerebras
```

Keys: [HuggingFace](https://huggingface.co/settings/tokens) · [Pinecone](https://app.pinecone.io) (create index: dim=384, metric=cosine)

**Ingest the knowledge base:**

```bash
# Parse a PDF into chunks
python scripts/ingest_pdf.py path/to/medical_reference.pdf data/knowledge.jsonl

# Embed and push to Pinecone
python scripts/ingest.py --data data/knowledge.jsonl

# Or fetch PubMed abstracts directly (no PDF needed)
python scripts/fetch_pubmed.py
python scripts/ingest.py --data data/pubmed_knowledge.jsonl
```

**Run:**

```bash
make dev
# or: uvicorn app.main:app --reload --port 8000
```

## API

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the symptoms of type 2 diabetes?"}'

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is ibuprofen?", "mode": "short"}'
```

Docs at `/docs`.

## Project structure

```
mediquery/
├── app/
│   ├── main.py              # FastAPI routes, rate limiting, security middleware
│   ├── rag.py               # Embed → retrieve → rerank → generate
│   ├── safety.py            # Two-tier emergency detection
│   └── config.py            # Settings from env vars
├── templates/
│   └── chat.html            # Chat UI
├── scripts/
│   ├── ingest.py            # Embed and upsert documents to Pinecone
│   ├── ingest_pdf.py        # PDF to JSONL chunks
│   ├── fetch_pubmed.py      # Pull abstracts from NIH PubMed API
│   └── evaluate_mediquery.py
├── tests/
│   ├── test_safety.py
│   └── test_scoring.py
├── notebooks/
│   └── retrieval_analysis.ipynb
└── .env.example
```

## Disclaimer

MediQuery is an educational project. It provides general health information only and does not diagnose, prescribe, or replace professional medical advice.

---

<div align="center">
Built by <a href="https://github.com/pranjalts07">Pranjal Tariga Suresh</a>
</div>