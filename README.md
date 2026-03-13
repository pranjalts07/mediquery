# 🏥 MediQuery — Medical RAG Chatbot

> **Production-ready Retrieval-Augmented Generation system for medical question answering.**
> Combines dense vector retrieval (Pinecone) with remote LLM inference (Hugging Face) to deliver grounded, source-cited medical responses — deployed on Azure Linux App Service.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![Pinecone](https://img.shields.io/badge/Pinecone-vector_db-purple.svg)](https://www.pinecone.io/)
[![Azure App Service](https://img.shields.io/badge/Azure-App_Service-blue.svg)](https://azure.microsoft.com/en-us/products/app-service/)

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────┐
│  FastAPI Backend (Azure)    │
│                             │
│  1. Safety Filter           │  ← Keyword-based emergency/misuse detection
│  2. HF Embed (API)          │  ← sentence-transformers/all-MiniLM-L6-v2
│  3. Pinecone Retrieval      │  ← Top-k cosine similarity search (384-dim)
│  4. Prompt Assembly         │  ← Grounded context + user question
│  5. HF Generate (API)       │  ← Mistral-7B-Instruct via HF router
│  6. Response + Sources      │  ← JSON {answer, sources[]}
└─────────────────────────────┘
    │
    ▼
Chat UI (chat.html, served at /)
```

### Why RAG?

Standard LLMs hallucinate medical facts. RAG grounds every answer in retrieved documents from a curated knowledge base, making responses auditable and source-attributed. This is the same pattern used by production medical AI systems (Epic, Nuance DAX, etc.).

### Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Framework | FastAPI | ASGI, async, automatic OpenAPI docs, Pydantic validation |
| Vector DB | Pinecone | Managed, serverless, sub-50ms retrieval at scale |
| Embeddings | HF Inference API | No local model download; avoids Azure 10GB build limit |
| LLM | HF Inference API | No GPU cost; swappable model without code changes |
| HTTP client | httpx | Async-ready, replaces requests for modern Python |
| No LangChain | ✅ | Eliminated fragile abstraction layer; pure Python is more maintainable |
| No torch/transformers | ✅ | All ML is remote API calls; fast Azure Oryx build |

---

## Project Structure

```
mediquery/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI routes and app setup
│   ├── config.py        # Environment variable loading
│   ├── rag.py           # Embed → Retrieve → Generate pipeline
│   └── safety.py        # Keyword safety filter
├── templates/
│   └── chat.html        # Chat UI (served at GET /)
├── scripts/
│   └── ingest.py        # Knowledge base ingestion script
├── data/
│   └── sample_knowledge.jsonl   # Sample medical knowledge (15 documents)
├── requirements.txt     # Minimal pinned dependencies
├── runtime.txt          # Python 3.11 for Azure Oryx
├── startup.sh           # gunicorn + uvicorn worker startup
├── .env.example         # Environment variable template
├── .gitignore
└── README.md
```

---

## Prerequisites

Before you begin you need:

- **Python 3.11** installed locally
- A **Hugging Face account** with an API token (free at [huggingface.co](https://huggingface.co/settings/tokens))
  - Enable "Inference API" access in your HF account settings
- A **Pinecone account** (free tier works) at [pinecone.io](https://app.pinecone.io)
  - Create an index: **name** = `mediquery`, **dimensions** = `384`, **metric** = `cosine`
- **Azure CLI** (for deployment only) — install from [docs.microsoft.com](https://docs.microsoft.com/cli/azure/install-azure-cli)

---

## Local Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/yourusername/mediquery.git
cd mediquery

python3.11 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```env
HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
PINECONE_API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
PINECONE_INDEX_NAME=mediquery
PINECONE_HOST=https://mediquery-xxxxxxx.svc.aped-xxxx-xxxx.pinecone.io
```

> **Where to find PINECONE_HOST:** Log into [app.pinecone.io](https://app.pinecone.io) → select your index → the host URL is shown in the index details panel.

### 4. Ingest the knowledge base

```bash
python scripts/ingest.py --data data/sample_knowledge.jsonl
```

Expected output:
```
2025-01-01 12:00:00 | INFO | Loaded 15 documents from data/sample_knowledge.jsonl
2025-01-01 12:00:00 | INFO | Connected to Pinecone index: mediquery
2025-01-01 12:00:01 | INFO | Embedding batch 1–15 of 15...
2025-01-01 12:00:03 | INFO | Upserted 15/15 vectors
2025-01-01 12:00:03 | INFO | ✅ Ingestion complete. 15 vectors stored in Pinecone.
```

To ingest your own documents, create a `.jsonl` file where each line is:
```json
{"id": "unique-id", "text": "document content here", "source": "Source Name"}
```

### 5. Run locally

```bash
uvicorn app.main:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000) — the MediQuery chat UI will load.

Health check:
```bash
curl http://localhost:8000/health
```

Test the chat API:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is hypertension and how is it treated?"}'
```

---

## Azure Deployment

### One-time setup

```bash
# Log in to Azure
az login

# Create a resource group (choose your preferred region)
az group create --name mediquery-rg --location eastus

# Create an App Service Plan (B1 is sufficient; free tier F1 also works)
az appservice plan create \
  --name mediquery-plan \
  --resource-group mediquery-rg \
  --sku B1 \
  --is-linux

# Create the Web App with Python 3.11
az webapp create \
  --name mediquery-app \
  --resource-group mediquery-rg \
  --plan mediquery-plan \
  --runtime "PYTHON:3.11"
```

### Set environment variables in Azure

```bash
az webapp config appsettings set \
  --name mediquery-app \
  --resource-group mediquery-rg \
  --settings \
    HF_API_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx" \
    PINECONE_API_KEY="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" \
    PINECONE_INDEX_NAME="mediquery" \
    PINECONE_HOST="https://mediquery-xxxxxxx.svc.aped-xxxx-xxxx.pinecone.io" \
    HF_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2" \
    HF_LLM_MODEL="mistralai/Mistral-7B-Instruct-v0.3" \
    SCM_DO_BUILD_DURING_DEPLOYMENT="true"
```

### Set the startup command

```bash
az webapp config set \
  --name mediquery-app \
  --resource-group mediquery-rg \
  --startup-file "bash startup.sh"
```

### Deploy via ZIP deploy (recommended — fastest, most reliable)

```bash
# Create deployment package (exclude venv and dev files)
zip -r mediquery.zip . \
  --exclude "venv/*" \
  --exclude ".git/*" \
  --exclude "__pycache__/*" \
  --exclude "*.pyc" \
  --exclude ".env"

# Deploy
az webapp deployment source config-zip \
  --name mediquery-app \
  --resource-group mediquery-rg \
  --src mediquery.zip
```

### Verify deployment

```bash
# Stream logs
az webapp log tail --name mediquery-app --resource-group mediquery-rg

# Health check
curl https://mediquery-app.azurewebsites.net/health
```

Your app is live at: `https://mediquery-app.azurewebsites.net`

---

## API Reference

### `GET /`
Returns the MediQuery chat interface (HTML).

### `GET /health`
Returns service health status.

**Response:**
```json
{
  "status": "ok",
  "version": "2.0.0",
  "model": "mistralai/Mistral-7B-Instruct-v0.3",
  "index": "mediquery"
}
```

### `POST /chat`
Runs the RAG pipeline and returns a grounded medical answer.

**Request:**
```json
{ "message": "What are the symptoms of type 2 diabetes?" }
```

**Response:**
```json
{
  "answer": "Type 2 diabetes mellitus is characterised by...",
  "sources": [
    { "source": "Endocrinology Clinical Reference", "score": 0.9231 },
    { "source": "Diabetes Management Guidelines", "score": 0.8847 }
  ]
}
```

Interactive API docs available at `/docs` (Swagger UI) and `/redoc`.

---

## Knowledge Base Options

MediQuery supports three knowledge base sources, in order of impressiveness:

### Option A — PubMed Abstracts (Recommended ⭐)
Real peer-reviewed papers from NIH's National Library of Medicine. No API key needed.

```bash
# Fetches ~800-1000 abstracts across 35 medical topics (takes ~10 mins)
python scripts/fetch_pubmed.py --out data/pubmed_knowledge.jsonl

# Push to Pinecone
python scripts/ingest.py --data data/pubmed_knowledge.jsonl --batch-size 16
```

Chat responses will cite real papers:
> *Source: Efficacy of metformin in type 2 diabetes — PMID 28754998 | NEJM. 2017*

**Interview answer:** *"My knowledge base is built from PubMed abstracts — the same NIH database used by medical researchers worldwide. I query their free E-utilities API across 35 clinical topic areas, pulling ~1000 peer-reviewed abstracts with real PMIDs shown as citations in every response."*

### Option B — Gale Encyclopedia of Medicine (included)
637-page medical encyclopedia, pre-chunked into 857 records. Already in `data/knowledge_base.jsonl`.

```bash
python scripts/ingest.py --data data/knowledge_base.jsonl --batch-size 16
```

### Option C — Both combined (best coverage)
```bash
python scripts/fetch_pubmed.py --out data/pubmed_knowledge.jsonl
cat data/knowledge_base.jsonl data/pubmed_knowledge.jsonl > data/combined_knowledge.jsonl
python scripts/ingest.py --data data/combined_knowledge.jsonl --batch-size 16
```

---

## Extending MediQuery

**Add your own knowledge:** Create a `.jsonl` file and run `python scripts/ingest.py --data your_data.jsonl`

**Swap the LLM:** Set `HF_LLM_MODEL` to any instruction-tuned model available via HF Inference API. The `generate()` function uses the standard OpenAI-compatible chat completions format that HF router supports.

**Scale the knowledge base:** Pinecone's serverless tier handles millions of vectors. Just ingest more documents — no infrastructure changes needed.

---

## Safety & Disclaimers

MediQuery is an educational demonstration project. It:
- Provides general health information only
- Always recommends consulting a qualified healthcare provider
- Includes emergency keyword detection that surfaces crisis resources
- Does **not** diagnose, prescribe, or replace professional medical advice

---

## Recruiter Summary

**GitHub description:** Production RAG chatbot for medical Q&A using Pinecone vector search, HuggingFace LLM inference, and FastAPI — deployed on Azure App Service.

**Resume bullet:** Built MediQuery, a production medical RAG system using Pinecone vector retrieval + Hugging Face LLM inference; achieved grounded, source-cited answers without local ML dependencies, deployed on Azure Linux App Service with gunicorn/uvicorn.

**One-paragraph summary:** MediQuery is a cloud-deployed Retrieval-Augmented Generation system that answers medical questions by combining dense vector search (Pinecone, 384-dim cosine similarity) with a remotely-hosted instruction-tuned LLM (Mistral-7B via Hugging Face Inference API). Built with FastAPI and a custom lightweight RAG pipeline — deliberately avoiding LangChain and local model weights — the system ingests a curated medical knowledge base, embeds each document using `sentence-transformers/all-MiniLM-L6-v2` via the HF API, and grounds every response in retrieved context with source attribution. The backend is containerisation-free, dependency-minimal, and deploys in under two minutes to Azure Linux App Service, demonstrating practical applied NLP, vector database integration, and cloud deployment skills.
