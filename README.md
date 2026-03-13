<div align="center">

# MediQuery
### AI Medical Assistant — Powered by RAG

![Python](https://img.shields.io/badge/Python-3.11-3776ab?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-6c47ff?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Inference_API-ff9d00?style=flat-square&logo=huggingface&logoColor=white)
![Azure](https://img.shields.io/badge/Azure-App_Service-0078d4?style=flat-square&logo=microsoftazure&logoColor=white)
[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-Click_Here-2563eb?style=for-the-badge)](https://mediquery-app.azurewebsites.net)

<div align="center">

<a href="https://mediquery-app.azurewebsites.net" target="_blank">
  <img src="https://img.shields.io/badge/-%F0%9F%8C%90%20Try%20MediQuery%20Live-0f1117?style=for-the-badge&logoColor=white" alt="Live Demo" height="45"/>
</a>

</div>
<div align="center">
  <img src="images/image1.png" alt="MediQuery Screenshot" width="800"/>
</div>

*Answers medical questions grounded in real research — not hallucinations*

</div>

---

## What is MediQuery?

MediQuery is a **production-deployed RAG (Retrieval-Augmented Generation) chatbot** that answers medical questions by searching 915 real medical documents and grounding every response in retrieved evidence — with source citations and confidence scores.

```
User question
     ↓
Safety filter (emergency detection)
     ↓
Embed question → 384-dim vector (HuggingFace API)
     ↓
Pinecone finds top-4 most similar medical passages
     ↓
LLM reads context → generates grounded answer
     ↓
Response + source citations + confidence score
```

---

## Features

| | |
|---|---|
| 🔍 **RAG Pipeline** | 915 medical documents — Gale Encyclopedia + PubMed abstracts |
| 💬 **Conversation Memory** | Remembers last 6 messages for natural follow-ups |
| 📄 **Source Citations** | Every answer shows exactly which documents were used |
| 📊 **Confidence Score** | Visual bar showing how well sources matched the query |
| 🔊 **Voice Input** | Speak your question (Chrome) |
| 📥 **Export PDF** | Download any conversation |
| 🚨 **Safety Layer** | Emergency keyword detection with crisis resources |
| 📱 **Mobile Ready** | Collapsible sidebar, works on any device |

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Backend | FastAPI | Async, Pydantic validation, auto OpenAPI docs |
| Vector DB | Pinecone | Managed, sub-50ms retrieval, free tier = 100k vectors |
| Embeddings | HF Inference API | No local model, no GPU needed |
| LLM | HF Inference API | Zero GPU cost, swappable without code changes |
| Deployment | Azure App Service B1 | Linux container, always-on |
| No LangChain | Pure Python | 8 dependencies total, deploys in 60 seconds |

---

## Local Setup

### 1. Clone & install

```bash
git clone https://github.com/pranjalts07/mediquery.git
cd mediquery
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up environment variables

```bash
cp .env.example .env
```

Fill in `.env`:

```env
HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
PINECONE_API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
PINECONE_INDEX_NAME=mediquery
PINECONE_HOST=https://mediquery-xxxxxxx.svc.aped-xxxx-xxxx.pinecone.io
HF_LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct:cerebras
```

> Get your keys: [HuggingFace](https://huggingface.co/settings/tokens) · [Pinecone](https://app.pinecone.io) (create index: name=`mediquery`, dims=`384`, metric=`cosine`)

### 3. Ingest the knowledge base

```bash
python scripts/ingest.py
```

> ⏱️ ~20-30 min on HuggingFace free tier. Run once — Pinecone stores vectors permanently.

### 4. Run

```bash
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000** 🎉

---

## API

```bash
# Health check
curl https://mediquery-app.azurewebsites.net/health

# Ask a question
curl -X POST https://mediquery-app.azurewebsites.net/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the symptoms of type 2 diabetes?"}'
```

Interactive docs at `/docs`

---

## Project Structure

```
mediquery/
├── app/
│   ├── main.py        # FastAPI routes
│   ├── rag.py         # Embed → Retrieve → Generate pipeline
│   ├── safety.py      # Emergency keyword filter
│   └── config.py      # Environment variables
├── templates/
│   └── chat.html      # Full chat UI
├── scripts/
│   ├── ingest.py      # Load docs → embed → upsert to Pinecone
│   └── fetch_pubmed.py  # Fetch PubMed abstracts from NIH
├── data/
│   └── sample_knowledge.jsonl
├── startup.sh         # gunicorn + uvicorn startup
└── .env.example       # Environment variable template
```

---

## Disclaimer

MediQuery is an educational project. It provides general health information only and does not diagnose, prescribe, or replace professional medical advice. Always consult a qualified healthcare provider.

---

<div align="center">

Built by **[Pranjal Tariga Suresh](https://github.com/pranjalts07)** · Deployed on Azure

⭐ Star this repo if you found it useful!

</div>
