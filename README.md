# IBM_NBA_Agent
# AccrediAI – NBA Accreditation Assistant

A **RAG-based Agentic AI** web application that helps faculty members retrieve and query NBA (National Board of Accreditation) accreditation information from their own uploaded documents using **IBM Granite** as the LLM and **IBM watsonx.ai** as the AI platform.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser (Faculty)                        │
│  Dashboard │ AI Chat │ Documents │ SAR Assist │ CO-PO Mapping  │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP / REST
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Flask Backend  (app.py)                     │
│                                                                 │
│  /api/upload  →  Document Ingestion Pipeline                    │
│  /api/chat    →  RAG Query Pipeline                             │
│  /api/stats   →  Dashboard Statistics                           │
│  /api/sar-assist  →  SAR Section Guidance                       │
│  /api/copo-map    →  CO-PO Mapping Assistant                    │
└──────────┬──────────────────────┬───────────────────────────────┘
           │                      │
           ▼                      ▼
┌──────────────────┐   ┌──────────────────────────────────────────┐
│  RAG Pipeline    │   │          IBM watsonx.ai (Cloud)          │
│  (rag_pipeline)  │   │                                          │
│                  │   │  Model: ibm/granite-4-h-small            │
│  ┌────────────┐  │   │  API: us-south.ml.cloud.ibm.com          │
│  │  LangChain │  │   │                                          │
│  │  Document  │  │   │  • Receives retrieved context + question │
│  │  Loaders   │  │   │  • Returns grounded, cited answer        │
│  └─────┬──────┘  │   └──────────────────────────────────────────┘
│        │         │
│  ┌─────▼──────┐  │
│  │  Sentence  │  │
│  │ Transformers│ │
│  │ (Embeddings)│ │
│  └─────┬──────┘  │
│        │         │
│  ┌─────▼──────┐  │
│  │   FAISS    │  │
│  │  (Vector   │  │
│  │   Store)   │  │
│  └────────────┘  │
└──────────────────┘
```

**Data flow for a query:**
1. Faculty asks a question via the chat UI
2. The question is embedded locally using `all-MiniLM-L6-v2`
3. FAISS returns the top-K most relevant document chunks
4. Chunks + question are sent to **IBM Granite** on IBM watsonx.ai
5. Granite returns a grounded answer (only from uploaded content)
6. The answer and source citations are displayed in the UI

---

## Features

| Feature | Description |
|---|---|
| 📊 Dashboard | Live stats: total docs, indexed docs, total queries, recent activity |
| 📄 Document upload | Upload PDF, DOCX, TXT, or XLSX files (up to 50 MB each) |
| 🔍 Semantic search | FAISS vector store with sentence-transformer embeddings (runs locally) |
| 🤖 IBM Granite | Context-grounded answers via IBM watsonx.ai |
| 📚 Knowledge base | View, manage, and delete indexed documents |
| 💬 Chat interface | Natural language Q&A with source citations |
| 🔗 Source references | Every answer cites the source document and page |
| 📝 SAR Assistance | AI-guided Self-Assessment Report section preparation |
| 🗂️ CO-PO Mapping | AI-suggested CO-PO correlation with NBA standard POs |
| 🚫 No hallucination | Explicitly says "not found" when context is missing |

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | IBM Granite (`ibm/granite-4-h-small`) via IBM watsonx.ai |
| AI Platform | IBM Cloud / IBM watsonx.ai |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, ~90 MB) |
| Vector Store | FAISS (persisted to disk, swappable via `VectorStoreBase` ABC) |
| Document Loaders | LangChain community (PyPDF, Docx2txt, Unstructured, OpenPyXL) |
| Backend | Python 3.11+, Flask |
| Frontend | Vanilla HTML5 / CSS3 / JavaScript (zero framework dependencies) |

---

## Project Structure

```
accrediai/
├── app.py                  # Flask backend – all routes, security headers, error handlers
├── rag_pipeline.py         # Modular RAG: VectorStoreBase ABC, FAISSVectorStore, RAGPipeline
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── README.md               # This file
│
├── uploads/                # Uploaded document files         (git-ignored)
├── knowledge_base/
│   ├── faiss_index/        # Persisted FAISS vector index    (git-ignored)
│   ├── doc_metadata.json   # Document metadata registry      (git-ignored)
│   └── stats.json          # Query statistics                (git-ignored)
│
├── templates/
│   └── index.html          # Main application template (6 sections)
└── static/
    ├── css/style.css       # Application stylesheet
    └── js/app.js           # Frontend logic
```

---

## Prerequisites

- Python 3.11 or later
- An **IBM Cloud** account with **IBM watsonx.ai** access
- An IBM watsonx **Project ID** and **API Key**

---

## IBM Cloud / watsonx.ai Setup

### Step 1 – Create an IBM Cloud account
Go to [https://cloud.ibm.com](https://cloud.ibm.com) and sign up or log in.

### Step 2 – Provision IBM watsonx.ai
1. In the IBM Cloud catalog, search for **Watson Machine Learning** and provision an instance (Lite tier is sufficient for testing).
2. Open **watsonx.ai** at [https://dataplatform.cloud.ibm.com](https://dataplatform.cloud.ibm.com).
3. Create or open a **Project**.

### Step 3 – Get your Project ID
1. Inside your watsonx.ai project, click **Manage** → **General**.
2. Copy the **Project ID** shown at the top.

### Step 4 – Create an API Key
1. In IBM Cloud, go to **Manage** → **Access (IAM)** → **API Keys**.
2. Click **Create an IBM Cloud API key**.
3. Copy the key immediately — it is shown only once.

### Step 5 – Confirm model availability
The application uses `ibm/granite-4-h-small`. You can verify it is available in your region via:
```
https://us-south.ml.cloud.ibm.com/ml/v1/foundation_model_specs?version=2023-05-29&project_id=YOUR_PROJECT_ID
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-org/accrediai.git
cd accrediai
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note**: `faiss-cpu` and `sentence-transformers` will download the embedding model (~90 MB) on first run.
> An internet connection is required for the first launch.

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your IBM watsonx credentials:

```env
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_API_KEY=<your IBM Cloud API key>
WATSONX_PROJECT_ID=<your watsonx project ID>
GRANITE_MODEL_ID=ibm/granite-4-h-small
```

### 5. Run the application

```bash
# The application auto-loads .env via python-dotenv
python app.py
```

The app is available at **http://localhost:5000**

> **Windows note**: If `python` is not found, try `py app.py`

---

## Usage

### Step 1 – Upload Documents
1. Click **Documents** in the sidebar.
2. Select a **Document Category** (e.g. "NBA Guidelines").
3. Drag-and-drop or browse for your PDF/DOCX/TXT/XLSX files.
4. Click **Upload & Index**. The backend extracts, chunks, embeds, and stores the content.

### Step 2 – Ask Questions
1. Click **AI Chat** in the sidebar.
2. Type your question or click one of the sample prompts.
3. AccrediAI retrieves the most relevant passages, sends them to IBM Granite, and returns a grounded answer.
4. The **Sources** panel on the right shows which document and page was used.

### Step 3 – SAR Assistance
1. Click **SAR Assistance** in the sidebar.
2. Describe the SAR section you need help with, or click a quick-topic chip.
3. Click **Generate SAR Guidance**.
4. Review the AI-generated content — it is always based on your uploaded documents only.

### Step 4 – CO-PO Mapping
1. Click **CO-PO Mapping** in the sidebar.
2. Enter Course Outcomes (COs) — one per line.
3. Enter Programme Outcomes (POs), or click **Load NBA Standard POs** to populate the 12 NBA standard POs.
4. Click **Generate CO-PO Mapping** to receive an AI-suggested correlation.

### Step 5 – Monitor the Dashboard
1. Click **Dashboard** in the sidebar.
2. View live stats: total documents, indexed documents, total queries.
3. Review the recent query history and system configuration.

### Step 6 – Manage the Knowledge Base
1. Click **Knowledge Base** to view all indexed documents.
2. Click **Remove** on any document to delete it from the vector index.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve the main application |
| `GET` | `/api/stats` | Dashboard statistics (documents, queries, KB status) |
| `POST` | `/api/upload` | Upload & ingest a document |
| `GET` | `/api/documents` | List all indexed documents (optional `?category=X`) |
| `GET` | `/api/documents/categories` | Category list with document counts |
| `DELETE` | `/api/documents/<id>` | Remove a document from the index |
| `POST` | `/api/chat` | RAG-powered Q&A via IBM Granite |
| `POST` | `/api/sar-assist` | SAR section preparation assistance |
| `POST` | `/api/copo-map` | CO-PO mapping suggestion |
| `GET` | `/api/health` | Service health check |

### `GET /api/stats` response

```json
{
  "total_documents": 4,
  "indexed_documents": 4,
  "total_queries": 12,
  "sar_queries": 3,
  "copo_queries": 1,
  "knowledge_base_ready": true,
  "model": "ibm/granite-4-h-small",
  "recent_queries": [
    { "question": "What are the NBA criteria?", "type": "chat", "time": "2024-08-01T10:32:00Z" }
  ],
  "categories": ["NBA Guidelines", "SAR Document", "CO-PO Mapping", "…"]
}
```

### `POST /api/chat` request body

```json
{
  "question": "What are the NBA eligibility criteria?",
  "top_k": 5
}
```

### `POST /api/chat` response

```json
{
  "question": "What are the NBA eligibility criteria?",
  "answer": "According to the uploaded document…",
  "sources": [
    {
      "document": "NBA_Criteria_2024.pdf",
      "page": 3,
      "excerpt": "Programmes seeking accreditation must have…",
      "relevance_score": 0.3412
    }
  ],
  "timestamp": "2024-08-01T10:32:00Z"
}
```

---

## Important Notes

- **No hallucination**: The system will explicitly state when information is not found in the uploaded documents — it never invents NBA criteria.
- **Local embeddings**: The sentence-transformer model runs locally; only the final LLM call hits IBM Cloud.
- **FAISS persistence**: The index is saved to `knowledge_base/faiss_index/` and reloaded on each restart.
- **Document deletion** triggers a full index rebuild from remaining documents.
- **Security**: Flask security headers (CSP, X-Frame-Options, HSTS, etc.) are applied to every response.

---

## .gitignore Recommendations

Create a `.gitignore` at the project root with:

```
# AccrediAI – files that must not be committed
accrediai/uploads/
accrediai/knowledge_base/
.env

# Python
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
dist/
build/
*.egg-info/

# IDE
.vscode/
.idea/
```

---

## IBM University Engagement

This project is designed for the **IBM University Engagement** programme and demonstrates:

| IBM Capability | Implementation |
|---|---|
| **IBM Granite** | `ibm/granite-4-h-small` — LLM for grounded question-answering |
| **IBM watsonx.ai** | Cloud AI inference platform via `ibm-watsonx-ai` Python SDK |
| **Retrieval-Augmented Generation** | FAISS + sentence-transformers + Granite |
| **Responsible AI** | No hallucination; explicit "not found" response; source citations on every answer |
| **Agentic AI workflow** | Modular RAG pipeline with swappable vector store (`VectorStoreBase` ABC) |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
